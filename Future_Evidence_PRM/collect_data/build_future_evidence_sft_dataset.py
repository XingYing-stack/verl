from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

try:
    from ..apb_alignment import (
        build_trajectory_prompt_messages,
        build_trajectory_response_json,
        extract_json_object,
        normalize_step_only_annotation,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from Future_Evidence_PRM.apb_alignment import (
        build_trajectory_prompt_messages,
        build_trajectory_response_json,
        extract_json_object,
        normalize_step_only_annotation,
    )


_CLIENT: Optional[OpenAI] = None
_CLIENT_BASE_URL: Optional[str] = None
_CLIENT_API_KEY: Optional[str] = None
_CLIENT_TIMEOUT: Optional[int] = None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _safe_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_safe_native(item) for item in obj]
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        return _safe_native(obj.as_py())
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        return _safe_native(obj.to_pylist())
    if isinstance(obj, tuple):
        return [_safe_native(item) for item in obj]
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return _safe_native(obj.tolist())
    return obj


def _derive_auxiliary_paths(output_path: Path) -> Tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_step_outputs{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}"),
    )


def _get_client(base_url: str, api_key: str, timeout: int) -> OpenAI:
    global _CLIENT, _CLIENT_BASE_URL, _CLIENT_API_KEY, _CLIENT_TIMEOUT
    if (
        _CLIENT is not None
        and _CLIENT_BASE_URL == base_url
        and _CLIENT_API_KEY == api_key
        and _CLIENT_TIMEOUT == timeout
    ):
        return _CLIENT
    _CLIENT = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)
    _CLIENT_BASE_URL = base_url
    _CLIENT_API_KEY = api_key
    _CLIENT_TIMEOUT = timeout
    return _CLIENT


def _call_teacher(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    client = _get_client(
        base_url=str(cfg["base_url"]),
        api_key=str(cfg["api_key"]),
        timeout=int(cfg["timeout"]),
    )
    response = client.chat.completions.create(
        model=str(cfg["model"]),
        messages=sample["messages"],
        temperature=float(cfg["temperature"]),
        max_tokens=int(cfg["max_tokens"]),
    )
    if not response.choices:
        raise ValueError("Empty completion choices")

    content = response.choices[0].message.content or ""
    parsed = extract_json_object(content)
    normalized = normalize_step_only_annotation(parsed)
    normalized["assistant_message_index"] = int(sample["assistant_message_index"])
    return {
        "query_index": sample["query_index"],
        "tree_id": sample["tree_id"],
        "trajectory_sample_id": sample["trajectory_sample_id"],
        "trajectory_terminal_node_id": sample["trajectory_terminal_node_id"],
        "assistant_node_id": sample["assistant_node_id"],
        "prefix_node_id": sample["prefix_node_id"],
        "assistant_message_index": int(sample["assistant_message_index"]),
        "question": sample["question"],
        "task_description": sample.get("task_description"),
        "tools": sample.get("tools"),
        "trajectory_messages": sample["trajectory_messages"],
        "assistant_message_indices": sample["assistant_message_indices"],
        "trajectory_final_label": int(sample["trajectory_final_label"]),
        "teacher_messages": sample["messages"],
        "teacher_raw_output": content,
        "teacher_parsed_output": normalized,
    }


def _worker(task: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    sample, cfg = task
    try:
        result = _call_teacher(sample, cfg)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "query_index": sample.get("query_index"),
            "tree_id": sample.get("tree_id"),
            "trajectory_sample_id": sample.get("trajectory_sample_id"),
            "assistant_node_id": sample.get("assistant_node_id"),
            "prefix_node_id": sample.get("prefix_node_id"),
            "assistant_message_index": sample.get("assistant_message_index"),
        }


def _build_expected_step_indices(teacher_samples: List[Dict[str, Any]]) -> Dict[str, set[int]]:
    expected: Dict[str, set[int]] = defaultdict(set)
    for sample in teacher_samples:
        trajectory_sample_id = str(sample["trajectory_sample_id"])
        expected[trajectory_sample_id].add(int(sample["assistant_message_index"]))
    return dict(expected)


def _aggregate_to_trajectory_pairs(
    step_outputs: List[Dict[str, Any]],
    expected_step_indices_by_trajectory: Dict[str, set[int]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in step_outputs:
        grouped[str(item["trajectory_sample_id"])].append(item)

    trajectory_pairs: List[Dict[str, Any]] = []
    dropped_incomplete_trajectory_count = 0
    dropped_incomplete_step_count = 0
    trajectory_ids = (
        sorted(set(expected_step_indices_by_trajectory) | set(grouped))
        if expected_step_indices_by_trajectory
        else sorted(grouped)
    )
    for trajectory_sample_id in trajectory_ids:
        items = grouped.get(trajectory_sample_id, [])
        items = sorted(items, key=lambda item: int(item["assistant_message_index"]))
        actual_step_indices = [int(item["assistant_message_index"]) for item in items]
        expected_step_indices = expected_step_indices_by_trajectory.get(trajectory_sample_id)
        if expected_step_indices is not None:
            if len(actual_step_indices) != len(expected_step_indices) or set(actual_step_indices) != expected_step_indices:
                dropped_incomplete_trajectory_count += 1
                dropped_incomplete_step_count += len(items)
                continue
        if not items:
            continue
        last_item = items[-1]
        trajectory_messages = _safe_native(last_item["trajectory_messages"])
        assistant_message_indices = [int(idx) for idx in _safe_native(last_item["assistant_message_indices"])]
        step_assessments = {
            str(item["teacher_parsed_output"]["assistant_message_index"]): str(item["teacher_parsed_output"]["step_assessment"])
            for item in items
        }
        step_labels = {
            str(item["teacher_parsed_output"]["assistant_message_index"]): int(item["teacher_parsed_output"]["step_label"])
            for item in items
        }
        step_future_implications = {
            str(item["teacher_parsed_output"]["assistant_message_index"]): _safe_native(
                item["teacher_parsed_output"]["future_implication"]
            )
            for item in items
        }
        step_explanations = {
            str(item["teacher_parsed_output"]["assistant_message_index"]): str(item["teacher_parsed_output"]["explanation"])
            for item in items
        }
        final_label = int(last_item["trajectory_final_label"])
        response_json = build_trajectory_response_json(
            step_assessments=step_assessments,
            step_labels=step_labels,
            step_future_implications=step_future_implications,
            final_label=final_label,
            step_explanations=step_explanations,
        )
        prompt_messages, prompt_assistant_indices = build_trajectory_prompt_messages(
            question=last_item["question"],
            task_description=last_item.get("task_description"),
            tools=last_item.get("tools"),
            messages=trajectory_messages,
        )
        if prompt_assistant_indices != assistant_message_indices:
            raise ValueError(
                f"Assistant indices mismatch for {trajectory_sample_id}: "
                f"expected {assistant_message_indices}, got {prompt_assistant_indices}"
            )
        trajectory_pairs.append(
            {
                "query_index": last_item["query_index"],
                "tree_id": last_item["tree_id"],
                "trajectory_sample_id": trajectory_sample_id,
                "trajectory_terminal_node_id": last_item["trajectory_terminal_node_id"],
                "question": last_item["question"],
                "task_description": last_item.get("task_description"),
                "tools": last_item.get("tools"),
                "messages": prompt_messages,
                "trajectory_messages": trajectory_messages,
                "assistant_message_indices": assistant_message_indices,
                "response": json.dumps(response_json, ensure_ascii=False, indent=2),
                "response_json": response_json,
                "step_node_ids": [item["assistant_node_id"] for item in items],
                "step_annotations": [
                    {
                        "assistant_message_index": item["teacher_parsed_output"]["assistant_message_index"],
                        "step_assessment": item["teacher_parsed_output"]["step_assessment"],
                        "step_label": item["teacher_parsed_output"]["step_label"],
                        "future_implication": _safe_native(item["teacher_parsed_output"]["future_implication"]),
                        "explanation": item["teacher_parsed_output"]["explanation"],
                    }
                    for item in items
                ],
            }
        )
    stats = {
        "complete_trajectory_pair_count": len(trajectory_pairs),
        "dropped_incomplete_trajectory_count": dropped_incomplete_trajectory_count,
        "dropped_incomplete_step_count": dropped_incomplete_step_count,
    }
    return trajectory_pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Call teacher model on step-level future-evidence data and build trajectory SFT pairs")
    parser.add_argument("--input", type=str, required=True, help="Teacher-input JSONL built by build_future_evidence_teacher_requests.py")
    parser.add_argument("--output", type=str, required=True, help="Trajectory-level SFT pair JSONL output path")
    parser.add_argument("--existing_step_outputs", type=str, default="", help="Optional existing *_step_outputs.jsonl to rebuild trajectory SFT pairs without re-calling the teacher model")
    parser.add_argument("--llm_base_url", type=str, default="")
    parser.add_argument("--llm_model", type=str, default="")
    parser.add_argument("--openai_api_key", type=str, default="dada")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step_output_path, failed_output_path = _derive_auxiliary_paths(output_path)

    teacher_samples = [_safe_native(record) for record in _iter_jsonl(input_path)]
    expected_step_indices_by_trajectory = _build_expected_step_indices(teacher_samples)

    step_outputs: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    if args.existing_step_outputs:
        step_outputs = [_safe_native(record) for record in _iter_jsonl(Path(args.existing_step_outputs))]
    else:
        if not args.llm_base_url or not args.llm_model:
            raise ValueError("`--llm_base_url` and `--llm_model` are required unless `--existing_step_outputs` is provided.")
        cfg = {
            "base_url": args.llm_base_url,
            "model": args.llm_model,
            "api_key": args.openai_api_key,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        }

        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
            futures = [executor.submit(_worker, (sample, cfg)) for sample in teacher_samples]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Teacher annotation", unit="step", dynamic_ncols=True):
                result = future.result()
                if result["ok"]:
                    step_outputs.append(result["result"])
                else:
                    failed.append(result)

    step_outputs = sorted(
        step_outputs,
        key=lambda item: (
            int(item["query_index"]),
            str(item["trajectory_sample_id"]),
            int(item["assistant_message_index"]),
        ),
    )
    trajectory_pairs, aggregate_stats = _aggregate_to_trajectory_pairs(step_outputs, expected_step_indices_by_trajectory)

    _write_jsonl(step_output_path, step_outputs)
    _write_jsonl(failed_output_path, failed)
    _write_jsonl(output_path, trajectory_pairs)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "step_output_path": str(step_output_path),
        "failed_output_path": str(failed_output_path),
        "existing_step_outputs": str(args.existing_step_outputs) if args.existing_step_outputs else "",
        "teacher_sample_count": len(teacher_samples),
        "successful_step_count": len(step_outputs),
        "failed_step_count": len(failed),
        "trajectory_pair_count": len(trajectory_pairs),
        **aggregate_stats,
        "max_workers": int(args.max_workers),
        "llm_model": str(args.llm_model),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
