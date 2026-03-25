from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    from ..utils import SamplingConfig, _extract_question_from_messages, _to_native, openai_chat_completions, read_parquet_dataset
    from .tree_sampling_all_nodes_data import _normalize_prompt_messages, sample_tree_for_question
except ImportError:  # pragma: no cover - fallback for direct execution
    from Future_Evidence_PRM.utils import (
        SamplingConfig,
        _extract_question_from_messages,
        _to_native,
        openai_chat_completions,
        read_parquet_dataset,
    )
    from Future_Evidence_PRM.collect_data.tree_sampling_all_nodes_data import (
        _normalize_prompt_messages,
        sample_tree_for_question,
    )


def _filter_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "hotpotqa":
        return df[df["data_source"] == "searchR1_hotpotqa"]
    if dataset == "nq":
        return df[df["data_source"] == "searchR1_nq"]
    return df


def _inspect_rows(df: pd.DataFrame, start: int, limit: int) -> Optional[Dict[str, Any]]:
    inspected = 0
    first_valid: Optional[Dict[str, Any]] = None

    for idx in range(start, min(len(df), start + limit)):
        row = df.iloc[idx]
        prompt_raw = row.get("prompt")
        prompt_native = _to_native(prompt_raw)
        base_messages = _normalize_prompt_messages(prompt_raw)
        extra_info = _to_native(row.get("extra_info", {}))
        question = extra_info.get("question") if isinstance(extra_info, dict) else None
        if not question:
            question = _extract_question_from_messages(base_messages)

        print(
            json.dumps(
                {
                    "row_iloc": idx,
                    "prompt_raw_type": type(prompt_raw).__name__,
                    "prompt_native_type": type(prompt_native).__name__,
                    "normalized_message_count": len(base_messages),
                    "has_question": bool(question),
                    "data_source": row.get("data_source", ""),
                },
                ensure_ascii=False,
            )
        )

        if base_messages and question and first_valid is None:
            first_valid = {
                "idx": idx,
                "row": row,
                "base_messages": base_messages,
                "question": question,
            }
        inspected += 1

    print(f"[dataset] inspected_rows={inspected}, first_valid_found={first_valid is not None}")
    return first_valid


def _probe_llm(args: argparse.Namespace) -> None:
    print("[llm] probing chat completions...")
    try:
        outputs = openai_chat_completions(
            base_url=args.llm_base_url,
            model=args.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with exactly: <answer> ok </answer>"},
            ],
            n=1,
            temperature=0.0,
            max_tokens=32,
            timeout=args.llm_timeout_s,
            api_key=args.openai_api_key,
            tools=None,
        )
        content = outputs[0].get("content", "") if outputs else ""
        print(json.dumps({"llm_ok": True, "sample_output": content[:200]}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"llm_ok": False, "error": str(exc)}, ensure_ascii=False))


def _probe_retriever(args: argparse.Namespace) -> None:
    print("[retriever] probing HTTP endpoint...")
    try:
        response = requests.post(
            args.retriever_url,
            json={"queries": ["python programming language"], "topk": 1, "return_scores": False},
            timeout=args.retriever_timeout_s,
        )
        preview = response.text[:500]
        print(
            json.dumps(
                {
                    "retriever_ok": response.ok,
                    "status_code": response.status_code,
                    "response_preview": preview,
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        print(json.dumps({"retriever_ok": False, "error": str(exc)}, ensure_ascii=False))


def _dry_run_sample(args: argparse.Namespace, first_valid: Dict[str, Any]) -> None:
    print("[sample] running one-row dry run...")
    row = first_valid["row"]
    extra_info = _to_native(row.get("extra_info", {}))
    ground_truth = None
    try:
        tools_kwargs = extra_info.get("tools_kwargs", {}) if isinstance(extra_info, dict) else {}
        ground_truth = tools_kwargs.get("search", {}).get("create_kwargs", {}).get("ground_truth")
        if isinstance(ground_truth, dict) and "target" in ground_truth:
            ground_truth["target"] = list(ground_truth["target"])
    except Exception:
        ground_truth = None

    cfg = SamplingConfig(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout_s=args.llm_timeout_s,
        branching_factor=args.dry_run_branching_factor,
        max_depth=args.dry_run_max_depth,
        concurrency=1,
        retriever_url=args.retriever_url,
        retriever_topk=args.retriever_topk,
        retriever_timeout_s=args.retriever_timeout_s,
        openai_api_key=args.openai_api_key,
    )

    try:
        tree = sample_tree_for_question(
            base_messages=first_valid["base_messages"],
            question=first_valid["question"],
            query_index=first_valid["idx"],
            data_source=row.get("data_source", ""),
            ground_truth=ground_truth,
            cfg=cfg,
        )
        print(
            json.dumps(
                {
                    "sample_ok": True,
                    "tree_id": tree["tree_id"],
                    "node_count": len(tree["nodes"]),
                    "terminal_count": len(tree["terminal_node_ids"]),
                    "root_continuation_count": len(tree["continuation_leaf_ids_by_node"].get(tree["root_id"], [])),
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        print(json.dumps({"sample_ok": False, "error": str(exc)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check for compact tree sampling")
    parser.add_argument("--dataset_path", type=str, required=True, help="Processed Search-R1 parquet path")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset subset: nq / hotpotqa / all")
    parser.add_argument("--start", type=int, default=0, help="Start row index to inspect")
    parser.add_argument("--inspect_rows", type=int, default=5, help="How many rows to inspect")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8888/v1")
    parser.add_argument("--llm_model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=64)
    parser.add_argument("--llm_timeout_s", type=int, default=30)
    parser.add_argument("--retriever_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retriever_topk", type=int, default=1)
    parser.add_argument("--retriever_timeout_s", type=int, default=15)
    parser.add_argument("--openai_api_key", type=str, default="dada")
    parser.add_argument("--dry_run_max_depth", type=int, default=1)
    parser.add_argument("--dry_run_branching_factor", type=int, default=1)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--skip_retriever", action="store_true")
    parser.add_argument("--skip_sample", action="store_true")
    args = parser.parse_args()

    df = read_parquet_dataset(args.dataset_path)
    print(f"[dataset] raw_rows={len(df)}")
    df = _filter_dataset(df, args.dataset)
    print(f"[dataset] filtered_rows={len(df)}, dataset={args.dataset}")

    if len(df) == 0:
        print("[dataset] no rows after filtering")
        return

    first_valid = _inspect_rows(df, args.start, args.inspect_rows)

    if not args.skip_llm:
        _probe_llm(args)
    if not args.skip_retriever:
        _probe_retriever(args)
    if not args.skip_sample:
        if first_valid is None:
            print("[sample] skipped because no valid prompt/question row was found")
        else:
            _dry_run_sample(args, first_valid)


if __name__ == "__main__":
    main()
