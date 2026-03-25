from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from tqdm import tqdm

try:
    from .apb_alignment import (
        build_trajectory_prompt_messages,
        extract_json_object,
        normalize_trajectory_annotation,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from Future_Evidence_PRM.apb_alignment import (
        build_trajectory_prompt_messages,
        extract_json_object,
        normalize_trajectory_annotation,
    )


TARGET_DATASETS = ["bfcl", "hotpotqa", "tau2", "gaia_dev"]
DATASET_TOTAL_INDEX_OFFSET = {
    "hotpotqa": 0,
    "gaia_dev": 250,
    "bfcl": 500,
    "tau2": 750,
}

_CLIENT_LOCAL = threading.local()


class _NullProgress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.current = 0

    def update(self, n: int = 1) -> None:
        self.current += n
        print(f"[{self.desc}] {self.current}/{self.total}", flush=True)

    def close(self) -> None:
        return


def _create_progress(*, total: int, desc: str):
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(obj)!r}")
            yield obj


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_record_id(dataset_name: str, obj: dict[str, Any]) -> str:
    data_source = str(obj.get("data_source") or dataset_name)
    query_index = obj.get("query_index")
    sample_index = obj.get("sample_index")
    if query_index is not None and sample_index is not None:
        return f"{data_source}:{query_index}:{sample_index}"
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{data_source}:{digest}"


def _build_future_evidence_messages(item: dict[str, Any]) -> tuple[list[dict[str, str]], list[int]]:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("item['messages'] must be a non-empty list")
    return build_trajectory_prompt_messages(
        question=item.get("question"),
        task_description=item.get("task_description"),
        tools=item.get("tools"),
        messages=messages,
    )


def _to_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def _infer_total_index(dataset: str, item: dict[str, Any]) -> int | None:
    ti = _to_int_or_none(item.get("total_index"))
    if ti is not None:
        return ti
    qi = _to_int_or_none(item.get("query_index"))
    si = _to_int_or_none(item.get("sample_index"))
    offset = DATASET_TOTAL_INDEX_OFFSET.get(dataset)
    if qi is None or si is None or offset is None:
        return None
    return offset + qi * 5 + si


def _normalize_step_labels(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for k, v in value.items():
            iv = _to_int_or_none(v)
            if iv is None:
                continue
            out[str(k)] = iv
        return out
    if isinstance(value, list):
        out: dict[str, int] = {}
        for i, v in enumerate(value):
            iv = _to_int_or_none(v)
            if iv is None:
                continue
            out[str(i)] = iv
        return out
    raise TypeError(f"Unsupported step_labels type: {type(value)!r}")


def _normalize_final_label(value: Any) -> int | None:
    iv = _to_int_or_none(value)
    if iv in (-1, 1):
        return iv
    return None


def _first_neg1_index(step_labels: dict[str, int]) -> int:
    idxs: list[int] = []
    for k, v in step_labels.items():
        if v != -1:
            continue
        try:
            idxs.append(int(k))
        except ValueError:
            continue
    return min(idxs) if idxs else -1


def _record_key(record: dict[str, Any], dataset: str) -> str:
    ti = _to_int_or_none(record.get("total_index"))
    if ti is not None:
        return f"ti:{ti}"

    qi = _to_int_or_none(record.get("query_index"))
    si = _to_int_or_none(record.get("sample_index"))
    if qi is not None and si is not None:
        off = DATASET_TOTAL_INDEX_OFFSET.get(dataset)
        if off is not None:
            return f"ti:{off + qi * 5 + si}"
        return f"qs:{qi}:{si}"

    rid = record.get("record_id")
    if isinstance(rid, str) and rid.strip():
        return f"rid:{rid.strip()}"

    raise KeyError(
        "Cannot infer record key (need total_index or query_index+sample_index or record_id). "
        f"keys={sorted(record.keys())}"
    )


@dataclass(frozen=True)
class ConsistencyMetrics:
    dataset: str
    ref_records: int
    pred_records: int
    compared_records: int
    missing_or_failed: int
    step_matches: int
    step_total: int
    step_exact_matches: int
    first_neg1_index_matches: int
    final_matches: int

    @property
    def missing_or_failed_ratio(self) -> float:
        return self.missing_or_failed / self.compared_records if self.compared_records else 0.0

    @property
    def step_micro_accuracy(self) -> float:
        return self.step_matches / self.step_total if self.step_total else 0.0

    @property
    def step_exact_accuracy(self) -> float:
        return self.step_exact_matches / self.compared_records if self.compared_records else 0.0

    @property
    def first_neg1_index_accuracy(self) -> float:
        return self.first_neg1_index_matches / self.compared_records if self.compared_records else 0.0

    @property
    def final_accuracy(self) -> float:
        return self.final_matches / self.compared_records if self.compared_records else 0.0


def _compute_metrics(
    *,
    dataset: str,
    ref_by_key: dict[str, dict[str, Any]],
    pred_by_key: dict[str, dict[str, Any]],
) -> ConsistencyMetrics:
    keys = list(ref_by_key.keys())
    compared_records = len(keys)

    missing_or_failed = 0
    step_matches = 0
    step_total = 0
    step_exact_matches = 0
    first_neg1_index_matches = 0
    final_matches = 0

    for key in keys:
        ref = ref_by_key[key]
        pred = pred_by_key.get(key)

        failed = False
        if pred is None:
            failed = True
        else:
            comment = pred.get("comment")
            if isinstance(comment, str) and comment.strip().startswith("llm_annotate_failed:"):
                failed = True
        if failed:
            missing_or_failed += 1

        ref_steps = _normalize_step_labels(ref.get("step_labels"))
        pred_steps = _normalize_step_labels(pred.get("step_labels")) if pred is not None else {}
        ref_final = _normalize_final_label(ref.get("final_label"))
        pred_final = _normalize_final_label(pred.get("final_label")) if pred is not None else None

        if _first_neg1_index(pred_steps) == _first_neg1_index(ref_steps):
            first_neg1_index_matches += 1
        if pred_final == ref_final:
            final_matches += 1

        step_total += len(ref_steps)
        for sk, sv in ref_steps.items():
            if pred_steps.get(sk) == sv:
                step_matches += 1

        if pred_steps == ref_steps:
            step_exact_matches += 1

    return ConsistencyMetrics(
        dataset=dataset,
        ref_records=len(ref_by_key),
        pred_records=len(pred_by_key),
        compared_records=compared_records,
        missing_or_failed=missing_or_failed,
        step_matches=step_matches,
        step_total=step_total,
        step_exact_matches=step_exact_matches,
        first_neg1_index_matches=first_neg1_index_matches,
        final_matches=final_matches,
    )


def _aggregate_metrics(metrics_list: list[ConsistencyMetrics]) -> ConsistencyMetrics:
    if not metrics_list:
        raise ValueError("empty metrics list")
    return ConsistencyMetrics(
        dataset="AVG",
        ref_records=sum(m.ref_records for m in metrics_list),
        pred_records=sum(m.pred_records for m in metrics_list),
        compared_records=sum(m.compared_records for m in metrics_list),
        missing_or_failed=sum(m.missing_or_failed for m in metrics_list),
        step_matches=sum(m.step_matches for m in metrics_list),
        step_total=sum(m.step_total for m in metrics_list),
        step_exact_matches=sum(m.step_exact_matches for m in metrics_list),
        first_neg1_index_matches=sum(m.first_neg1_index_matches for m in metrics_list),
        final_matches=sum(m.final_matches for m in metrics_list),
    )


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_metrics_table(metrics_list: list[ConsistencyMetrics], run_name: str) -> None:
    headers = [
        "dataset",
        "ref_n",
        "pred_n",
        "missing_or_failed_pct",
        "step_micro_acc",
        "step_exact_acc",
        "first_neg1_idx_acc",
        "final_acc",
    ]
    rows = []
    for m in metrics_list:
        rows.append(
            [
                m.dataset,
                str(m.ref_records),
                str(m.pred_records),
                _format_pct(m.missing_or_failed_ratio),
                _format_pct(m.step_micro_accuracy),
                _format_pct(m.step_exact_accuracy),
                _format_pct(m.first_neg1_index_accuracy),
                _format_pct(m.final_accuracy),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    numeric_cols = {
        "ref_n",
        "pred_n",
        "missing_or_failed_pct",
        "step_micro_acc",
        "step_exact_acc",
        "first_neg1_idx_acc",
        "final_acc",
    }
    print(f"MODEL  {run_name}")
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    for row in rows:
        cells: list[str] = []
        for i, cell in enumerate(row):
            if headers[i] in numeric_cols:
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        print("  ".join(cells))
    print()


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    client: OpenAI | None = getattr(_CLIENT_LOCAL, "client", None)
    cached_base = getattr(_CLIENT_LOCAL, "base_url", None)
    cached_key = getattr(_CLIENT_LOCAL, "api_key", None)
    if client is not None and cached_base == base_url and cached_key == api_key:
        return client
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    _CLIENT_LOCAL.client = client
    _CLIENT_LOCAL.base_url = base_url
    _CLIENT_LOCAL.api_key = api_key
    return client


def _call_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    max_attempts: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_openai_client(base_url, api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            if not response.choices:
                return ""
            message = response.choices[0].message
            return message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def _load_existing_raw_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for rec in _iter_jsonl(path):
        row_id = _to_int_or_none(rec.get("row_id"))
        if row_id is None:
            continue
        out[row_id] = rec
    return out


def _normalize_prediction(
    *,
    parsed: dict[str, Any],
    assistant_indices: list[int],
) -> tuple[dict[str, str], dict[str, int], dict[str, dict[str, Any]], int, dict[str, str]]:
    return normalize_trajectory_annotation(parsed, assistant_indices=assistant_indices)


def _build_success_record(
    *,
    dataset: str,
    run_name: str,
    model_name: str,
    index_in_dataset: int,
    item: dict[str, Any],
    raw_text: str,
    parsed: dict[str, Any],
    assistant_indices: list[int],
) -> dict[str, Any]:
    step_assessments, step_labels, future_implications, final_label, explanations = _normalize_prediction(
        parsed=parsed,
        assistant_indices=assistant_indices,
    )
    record: dict[str, Any] = {
        "dataset": dataset,
        "record_id": str(item.get("record_id") or _stable_record_id(dataset, item)),
        "annotator": run_name,
        "username": run_name,
        "model": model_name,
        "index_in_dataset": index_in_dataset,
        "data_source": item.get("data_source"),
        "query_index": item.get("query_index"),
        "sample_index": item.get("sample_index"),
        "step_assessments": {
            "steps": step_assessments,
        },
        "future_implications": {
            "steps": future_implications,
        },
        "explanations": {
            "steps": explanations,
        },
        "step_labels": step_labels,
        "final_label": final_label,
        "status": "done",
        "comment": "",
        "updated_at": _utc_now_iso(),
    }
    total_index = _infer_total_index(dataset, item)
    if total_index is not None:
        record["total_index"] = total_index
    return record


def _build_failure_record(
    *,
    dataset: str,
    run_name: str,
    model_name: str,
    index_in_dataset: int,
    item: dict[str, Any],
    comment: str,
    raw_text: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dataset": dataset,
        "record_id": str(item.get("record_id") or _stable_record_id(dataset, item)),
        "annotator": run_name,
        "username": run_name,
        "model": model_name,
        "index_in_dataset": index_in_dataset,
        "data_source": item.get("data_source"),
        "query_index": item.get("query_index"),
        "sample_index": item.get("sample_index"),
        "step_assessments": {
            "steps": {},
        },
        "future_implications": {
            "steps": {},
        },
        "explanations": {
            "steps": {},
        },
        "step_labels": {},
        "final_label": None,
        "status": "failed",
        "comment": f"llm_annotate_failed: {comment}",
        "updated_at": _utc_now_iso(),
    }
    total_index = _infer_total_index(dataset, item)
    if total_index is not None:
        record["total_index"] = total_index
    return record


def _evaluate_dataset(
    *,
    dataset: str,
    data_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    run_name: str,
    model_name: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    max_attempts: int,
    start: int,
    end: int,
    save_every: int,
) -> tuple[Path, ConsistencyMetrics]:
    input_path = data_dir / f"{dataset}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    source_records = list(_iter_jsonl(input_path))
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end >= 0:
        source_records = source_records[start:end]
    else:
        source_records = source_records[start:]
    if not source_records:
        raise ValueError(f"No records selected from {input_path}")

    print(
        f"[{dataset}] loaded selected_records={len(source_records)} start={start} end={'ALL' if end < 0 else end}",
        flush=True,
    )

    raw_path = raw_dir / f"{dataset}__{run_name}.responses.jsonl"
    existing_raw = _load_existing_raw_records(raw_path)
    raw_records: list[dict[str, Any]] = [{"row_id": i} for i in range(len(source_records))]

    pending: list[tuple[int, list[dict[str, str]]]] = []
    for row_id, item in enumerate(source_records):
        cached = existing_raw.get(row_id)
        if cached is not None and isinstance(cached.get("response"), str) and cached.get("response", "").strip():
            raw_records[row_id] = cached
            continue
        prompt, _assistant_indices = _build_future_evidence_messages(item)
        pending.append((row_id, prompt))

    cached_count = len(source_records) - len(pending)
    print(f"[{dataset}] cache_hit={cached_count} pending_requests={len(pending)}", flush=True)

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_row = {
                executor.submit(
                    _call_llm,
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    max_attempts=max_attempts,
                ): row_id
                for row_id, prompt in pending
            }
            finished = 0
            total = len(pending)
            progress = _create_progress(total=total, desc=f"{dataset}:request")
            for future in concurrent.futures.as_completed(future_to_row):
                row_id = future_to_row[future]
                try:
                    content = future.result()
                    raw_records[row_id] = {"row_id": row_id, "response": content, "error": None}
                except Exception as exc:
                    raw_records[row_id] = {"row_id": row_id, "response": "", "error": str(exc)}
                finished += 1
                progress.update(1)
                if finished % save_every == 0 or finished == total:
                    _write_jsonl(raw_path, raw_records)
                    print(f"[{dataset}] raw_saved progress={finished}/{total} path={raw_path}", flush=True)
            progress.close()

        for row_id, rec in existing_raw.items():
            if 0 <= row_id < len(raw_records) and raw_records[row_id].keys() == {"row_id"}:
                raw_records[row_id] = rec
        _write_jsonl(raw_path, raw_records)
        print(f"[{dataset}] raw_complete path={raw_path}", flush=True)
    else:
        for row_id, rec in existing_raw.items():
            if 0 <= row_id < len(raw_records):
                raw_records[row_id] = rec
        if raw_records:
            _write_jsonl(raw_path, raw_records)
        print(f"[{dataset}] using cached raw responses path={raw_path}", flush=True)

    print(f"[{dataset}] start postprocess total={len(source_records)}", flush=True)
    pred_records: list[dict[str, Any]] = []
    for row_id, item in enumerate(source_records):
        prompt, assistant_indices = _build_future_evidence_messages(item)
        _ = prompt
        rec = raw_records[row_id]
        raw_text = str(rec.get("response") or "")
        error = rec.get("error")
        if error:
            pred_records.append(
                _build_failure_record(
                    dataset=dataset,
                    run_name=run_name,
                    model_name=model_name,
                    index_in_dataset=start + row_id,
                    item=item,
                    comment=str(error),
                    raw_text=raw_text,
                )
            )
            continue

        try:
            parsed = extract_json_object(raw_text)
            pred_records.append(
                _build_success_record(
                    dataset=dataset,
                    run_name=run_name,
                    model_name=model_name,
                    index_in_dataset=start + row_id,
                    item=item,
                    raw_text=raw_text,
                    parsed=parsed,
                    assistant_indices=assistant_indices,
                )
            )
        except Exception as exc:
            pred_records.append(
                _build_failure_record(
                    dataset=dataset,
                    run_name=run_name,
                    model_name=model_name,
                    index_in_dataset=start + row_id,
                    item=item,
                    comment=str(exc),
                    raw_text=raw_text,
                )
            )

    output_path = out_dir / f"{dataset}__{run_name}.jsonl"
    _write_jsonl(output_path, pred_records)
    print(f"[{dataset}] predictions_written path={output_path}", flush=True)

    ref_by_key: dict[str, dict[str, Any]] = {}
    for rec in source_records:
        key = _record_key(rec, dataset)
        if key in ref_by_key:
            raise ValueError(f"Duplicate reference key {key} in selected slice of {input_path}")
        ref_by_key[key] = rec
    pred_by_key = {_record_key(rec, dataset): rec for rec in pred_records}
    metrics = _compute_metrics(dataset=dataset, ref_by_key=ref_by_key, pred_by_key=pred_by_key)
    print(
        (
            f"[{dataset}] done step_micro_acc={metrics.step_micro_accuracy:.4f} "
            f"step_exact_acc={metrics.step_exact_accuracy:.4f} "
            f"first_neg1_idx_acc={metrics.first_neg1_index_accuracy:.4f} "
            f"final_acc={metrics.final_accuracy:.4f} "
            f"missing_or_failed={metrics.missing_or_failed}/{metrics.compared_records}"
        ),
        flush=True,
    )
    return output_path, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate future-evidence SFT models on raw tool-use trajectories via an "
            "OpenAI-compatible API with the structured payload and output schema used by this pipeline."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing <dataset>.jsonl files.")
    parser.add_argument("--results_root", type=Path, required=True, help="Root directory for prediction outputs.")
    parser.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible base URL.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API key.")
    parser.add_argument("--model", type=str, required=True, help="Model name served by the endpoint.")
    parser.add_argument("--run_name", type=str, default="", help="Output run name. Defaults to sanitized model name.")
    parser.add_argument("--datasets", type=str, default=",".join(TARGET_DATASETS), help="Comma-separated dataset list.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--start", type=int, default=0, help="Inclusive dataset row start.")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive dataset row end; -1 means full dataset.")
    parser.add_argument("--save_every", type=int, default=20, help="Flush raw responses every N completed requests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("api_key is required (or set OPENAI_API_KEY)")

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise ValueError("No datasets provided")

    run_name = args.run_name or f"blind_{_sanitize_filename(args.model)}"
    run_dir = args.results_root / run_name
    raw_dir = args.results_root / "_raw" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics: list[ConsistencyMetrics] = []
    output_paths: list[Path] = []
    for dataset in datasets:
        print(f"[run] start dataset={dataset}", flush=True)
        output_path, metrics = _evaluate_dataset(
            dataset=dataset,
            data_dir=args.data_dir,
            raw_dir=raw_dir,
            out_dir=run_dir,
            run_name=run_name,
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            max_attempts=args.max_attempts,
            start=args.start,
            end=args.end,
            save_every=args.save_every,
        )
        output_paths.append(output_path)
        dataset_metrics.append(metrics)

    avg = _aggregate_metrics(dataset_metrics)
    metrics_with_avg = [*dataset_metrics, avg]
    _print_metrics_table(metrics_with_avg, run_name)

    print(f"FINAL_SCORE[step_micro_acc]={avg.step_micro_accuracy * 100:.2f}%", flush=True)
    print(f"FINAL_SCORE_RAW={avg.step_micro_accuracy:.6f}", flush=True)

    summary = {
        "run_name": run_name,
        "model": args.model,
        "base_url": args.base_url,
        "datasets": datasets,
        "start": args.start,
        "end": args.end,
        "outputs": [str(path) for path in output_paths],
        "metrics": {
            m.dataset: {
                "ref_records": m.ref_records,
                "pred_records": m.pred_records,
                "compared_records": m.compared_records,
                "missing_or_failed_ratio": m.missing_or_failed_ratio,
                "step_micro_accuracy": m.step_micro_accuracy,
                "step_exact_accuracy": m.step_exact_accuracy,
                "first_neg1_index_accuracy": m.first_neg1_index_accuracy,
                "final_accuracy": m.final_accuracy,
            }
            for m in metrics_with_avg
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
