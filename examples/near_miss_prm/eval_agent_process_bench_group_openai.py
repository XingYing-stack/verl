import argparse
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from openai import OpenAI

_client_local = threading.local()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    # First, try to extract JSON from ```json ... ``` markdown code block
    json_block_pattern = re.compile(r"```json\\s*([\\s\\S]*?)\\s*```", re.IGNORECASE)
    matches = json_block_pattern.findall(text)
    if matches:
        # Use the last match (in case there are multiple code blocks)
        json_str = matches[-1].strip()
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole text as JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    end = text.rfind("}")
    if end < 0:
        raise ValueError("LLM output is not JSON")

    brace_count = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == "}":
            brace_count += 1
        elif text[i] == "{":
            brace_count -= 1
            if brace_count == 0:
                start = i
                break
    if start < 0:
        raise ValueError("LLM output is not JSON")

    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object")
    return obj


def _coerce_int_label(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("label must be -1/0/1, got bool")
    if isinstance(val, int):
        out = val
    elif isinstance(val, str) and val.strip() in {"-1", "0", "1"}:
        out = int(val.strip())
    else:
        raise ValueError(f"label must be -1/0/1, got {val!r}")
    if out not in (-1, 0, 1):
        raise ValueError(f"label must be -1/0/1, got {out}")
    return out


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    client: OpenAI | None = getattr(_client_local, "client", None)
    cached_base = getattr(_client_local, "base_url", None)
    cached_key = getattr(_client_local, "api_key", None)
    if client is not None and cached_base == base_url and cached_key == api_key:
        return client
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    _client_local.client = client
    _client_local.base_url = base_url
    _client_local.api_key = api_key
    return client


def _call_llm(*, base_url: str, api_key: str, model: str, messages: list[dict[str, Any]], max_tokens: int) -> str:
    client = _get_openai_client(base_url, api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if not resp.choices:
        return ""
    msg = resp.choices[0].message
    return msg.content or ""


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _iter_row_trajectories(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    if "trajectories" in extra_info:
        return list(extra_info["trajectories"])
    return [extra_info]


def _score_one_trajectory(
    *,
    pred_traj: dict[str, Any] | None,
    assistant_indices: list[int],
    gt_step_labels: dict[str, Any],
    gt_final_label: int,
) -> tuple[float, float, int]:
    if pred_traj is None:
        return 0.0, 0.0, 0

    step_labels_raw = pred_traj.get("step_labels")
    final_label_raw = pred_traj.get("final_label")
    if not isinstance(step_labels_raw, dict):
        return 0.0, 0.0, 0

    try:
        pred_final = _coerce_int_label(final_label_raw)
    except Exception:
        return 0.0, 0.0, 0

    # Outcome label: map gt (-1/0)->-1, 1->1
    if gt_final_label in (-1, 0):
        target = -1
    elif gt_final_label == 1:
        target = 1
    else:
        raise ValueError(f"unexpected final_label: {gt_final_label}")
    orm_score = 1.0 if pred_final == target else 0.0

    # Process label accuracy over assistant indices (drop None in gt)
    gt_steps = {str(k): int(v) for k, v in gt_step_labels.items() if v is not None}
    expected_keys = {str(i) for i in assistant_indices}
    assert gt_steps.keys() == expected_keys, "gt step_labels keys must match assistant_indices"

    pred_steps: dict[str, int] = {}
    for i in assistant_indices:
        key = str(i)
        if key not in step_labels_raw:
            return 0.0, orm_score, 0
        try:
            pred_steps[key] = _coerce_int_label(step_labels_raw[key])
        except Exception:
            return 0.0, orm_score, 0

    correct = sum(1 for k in expected_keys if pred_steps[k] == gt_steps[k])
    prm_score = correct / len(expected_keys) if expected_keys else 0.0
    return prm_score, orm_score, 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True, help="Parquet file path.")
    parser.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible base_url.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API key.")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--out_dir", type=str, default="/workspace/verl/rollout_data", help="Output directory (default: parquet dir).")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("api_key is required (or set OPENAI_API_KEY).")

    parquet_path = Path(args.parquet)
    out_dir = Path(args.out_dir) if args.out_dir else parquet_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url_tag = _sanitize_filename(args.base_url)
    model_tag = _sanitize_filename(args.model)

    out_jsonl = out_dir / f"{parquet_path.stem}.{base_url_tag}.{model_tag}.responses.jsonl"
    out_summary = out_dir / f"{parquet_path.stem}.{base_url_tag}.{model_tag}.summary.json"

    df = pd.read_parquet(parquet_path)
    prompts: list[list[dict[str, Any]]] = df["prompt"].tolist()
    extra_infos: list[dict[str, Any]] = df["extra_info"].tolist()
    num_rows = len(df)

    if out_jsonl.exists():
        records = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != num_rows:
            raise ValueError(f"{out_jsonl} exists but has {len(records)} lines; expected {num_rows}.")
    else:
        records: list[dict[str, Any]] = [{"row_id": i} for i in range(num_rows)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(
                    _call_llm,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=prompts[i],
                    max_tokens=args.max_tokens,
                ): i
                for i in range(num_rows)
            }
            done = 0
            for fut in as_completed(futures):
                row_id = futures[fut]
                try:
                    content = fut.result()
                    records[row_id] = {"row_id": row_id, "response": content, "error": None}
                except Exception as e:
                    records[row_id] = {"row_id": row_id, "response": "", "error": str(e)}
                done += 1
                if done % 50 == 0 or done == num_rows:
                    print(f"completed {done}/{num_rows}")

        with out_jsonl.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prm_scores: dict[int, list[float]] = defaultdict(list)
    orm_scores: dict[int, list[float]] = defaultdict(list)
    format_scores: dict[int, list[int]] = defaultdict(list)

    for row_id, rec in enumerate(records):
        extra_info = extra_infos[row_id]
        trajectories = _iter_row_trajectories(extra_info)
        raw_text = rec.get("response", "")

        parsed: dict[str, Any] | None
        try:
            parsed = _extract_json_object(raw_text)
        except Exception:
            parsed = None

        pred_trajectories: dict[str, Any] | None = None
        if parsed is not None:
            if "trajectories" in extra_info:
                pred_trajectories = parsed.get("trajectories") if isinstance(parsed.get("trajectories"), dict) else None
            else:
                pred_trajectories = {"0": parsed}

        for traj in trajectories:
            traj_id = str(traj.get("trajectory_id", "0"))
            traj_index = int(traj.get("index", traj.get("total_index")))
            assistant_indices = list(traj.get("assistant_indices"))
            gt_final_label = int(traj.get("final_label"))
            gt_step_labels = traj.get("step_labels")
            assert isinstance(gt_step_labels, dict), "step_labels must be a dict"

            pred_traj = pred_trajectories.get(traj_id) if pred_trajectories is not None else None
            prm, orm, fmt = _score_one_trajectory(
                pred_traj=pred_traj,
                assistant_indices=assistant_indices,
                gt_step_labels=gt_step_labels,
                gt_final_label=gt_final_label,
            )
            prm_scores[traj_index].append(prm)
            orm_scores[traj_index].append(orm)
            format_scores[traj_index].append(fmt)

    prm_macro = mean(mean(v) for v in prm_scores.values())
    orm_macro = mean(mean(v) for v in orm_scores.values())
    fmt_macro = mean(mean(v) for v in format_scores.values())

    summary = {
        "parquet": str(parquet_path),
        "base_url": args.base_url,
        "model": args.model,
        "responses_path": str(out_jsonl),
        "num_rows": num_rows,
        "num_unique_trajectories": len(prm_scores),
        "macro": {
            "prm_score": prm_macro,
            "orm_score": orm_macro,
            "format_pass_rate": fmt_macro,
        },
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary["macro"], ensure_ascii=False, indent=2))
    print(f"saved: {out_jsonl}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
