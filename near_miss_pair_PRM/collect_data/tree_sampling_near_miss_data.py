"""
Concurrent tree sampling of multi-turn, tool-using trajectories for PRM/ORM data.

Overview
- Reads processed Search-R1-like parquet produced by examples/data_preprocess/preprocess_search_r1_dataset.py
- For each question, drives a local LLM server (OpenAI-compatible, e.g., SGLang) on port 8888
- The model emits <tool_call> ... </tool_call> and <answer> ... </answer> blocks
- On <tool_call>, this script queries the local retriever service (port 8000) and injects
  the results between <tool_response> ... </tool_response>, then continues sampling
- Performs breadth-first tree sampling with configurable branching factor and depth
- Saves all trajectories (every root-to-node path) to a JSONL file; near-miss pairing is not generated here

Assumptions
- LLM server exposes OpenAI-compatible /v1/chat/completions at http://127.0.0.1:8888
- Retriever exposes POST /retrieve at http://127.0.0.1:8000/retrieve

Usage example
    python near_miss_pair_PRM/tree_sampling_near_miss_data.py \
        --dataset_path /workspace/fanshengda/verl/input_data/searchR1_processed_direct/train.parquet \
        --output_dir ./rollout_data/tree_sampling \
        --max_depth 4 --branching_factor 3 --concurrency 32

Notes
- We do not compute near-miss pairs here; only full trajectories are saved.
- To keep dependency-light, we call both LLM and retriever via requests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from queue import Queue

import pandas as pd
import requests
from openai import OpenAI
import atexit
from tqdm import tqdm

from ..utils import *

def sample_tree_for_question(
    base_messages: List[Dict[str, Any]],
    question: str,
    query_index: int,
    data_source: str,
    ground_truth: Any,
    cfg: SamplingConfig,
) -> List[TrajectoryRecord]:
    """Breadth-first tree sampling for a single question.

    At every depth, for each frontier node, sample `branching_factor` completions.
    If a completion emits <tool_call>, we fetch <tool_response> and continue.
    If a completion emits <answer>, we mark it terminated.
    We save every root-to-node transcript as a trajectory record.
    """
    results: List[TrajectoryRecord] = []

    # Each frontier item is a tuple of (messages_so_far)
    frontier: List[List[Dict[str, Any]]] = [list(base_messages)]

    for depth in range(cfg.max_depth):
        next_frontier: List[List[Dict[str, Any]]] = []

        for node_messages in frontier:
            # Call LLM for branching_factor samples from this node
            try:
                amsgs = openai_chat_completions(
                    base_url=cfg.llm_base_url,
                    model=cfg.llm_model,
                    messages=node_messages,
                    n=cfg.branching_factor,
                    temperature=cfg.llm_temperature,
                    max_tokens=cfg.llm_max_tokens,
                    timeout=cfg.llm_timeout_s,
                    api_key=cfg.openai_api_key,
                    tools=cfg.tools,
                )
            except Exception as e:
                # Record a failed leaf for observability and skip branching further
                fail_msg = {"role": "assistant", "content": f"<error> LLM error: {e} </error>"}
                msgs = node_messages + [fail_msg]
                results.append(
                    TrajectoryRecord(
                        query_index=query_index,
                        question=question,
                        data_source=data_source,
                        ground_truth=ground_truth,
                        messages=msgs,
                        depth=depth,
                        terminated=True,
                        leaf_output=fail_msg["content"],
                    )
                )
                continue

            for assistant in amsgs:
                content = assistant.get("content", "")
                tool_calls = assistant.get("tool_calls", []) or []
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                traj_messages: List[Dict[str, Any]] = node_messages + [assistant_msg]

                # Save immediate node as a trajectory
                results.append(
                    TrajectoryRecord(
                        query_index=query_index,
                        question=question,
                        data_source=data_source,
                        ground_truth=ground_truth,
                        messages=traj_messages,
                        depth=depth,
                        terminated=contains_answer(content),
                        leaf_output=content,
                    )
                )

                if contains_answer(content):
                    # Do not expand further
                    continue

                # Prefer structured tool_calls. If none, fallback to parsing <tool_call> from content.
                structured_calls: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    structured_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments", {}),
                        }
                    )

                legacy_mode = False
                if not structured_calls:
                    legacy = parse_tool_call(content)
                    if legacy:
                        legacy_mode = True
                        structured_calls = [
                            {"id": None, "name": legacy.get("name"), "arguments": legacy.get("arguments", {})}
                        ]

                for call in structured_calls:
                    if call.get("name") != "search":
                        continue
                    args = call.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    qlist = args.get("query_list")
                    if not qlist and "query" in args:
                        qlist = [args["query"]]
                    if isinstance(qlist, str):
                        query_list = [qlist]
                    elif isinstance(qlist, list):
                        query_list = [str(x) for x in qlist]
                    else:
                        query_list = []
                    if not query_list:
                        continue

                    try:
                        retrieved = call_retriever(
                            retrieval_url=cfg.retriever_url,
                            query_list=query_list,
                            topk=cfg.retriever_topk,
                            timeout=cfg.retriever_timeout_s,
                        )
                    except Exception as e:
                        retrieved = f"Search error: {e}"

                    if legacy_mode:
                        # Legacy format: inject as user content with <tool_response> block
                        tool_resp_block = f"<tool_response>\n{retrieved}\n</tool_response>"
                        env_msg: Dict[str, Any] = {"role": "user", "content": tool_resp_block}
                        extended = traj_messages + [env_msg]
                    else:
                        tool_msg: Dict[str, Any] = {
                            "role": "tool",
                            "content": retrieved,
                            "name": "search",
                        }
                        if call.get("id"):
                            tool_msg["tool_call_id"] = call["id"]
                        extended = traj_messages + [tool_msg]

                    next_frontier.append(extended)
                    results.append(
                        TrajectoryRecord(
                            query_index=query_index,
                            question=question,
                            data_source=data_source,
                            ground_truth=ground_truth,
                            messages=extended,
                            depth=depth,
                            terminated=False,
                            leaf_output=content,
                        )
                    )
                    if cfg.pause_between_calls_s > 0:
                        time.sleep(cfg.pause_between_calls_s)

        if not next_frontier:
            break
        frontier = next_frontier

    return results

def process_rows_in_parallel(
    df: pd.DataFrame,
    row_indices: Iterable[int],
    cfg: SamplingConfig,
    output_fp: str,
) -> None:
    """Process a subset of rows concurrently and append JSONL trajectories to output file.

    We use a file-level lock to serialize writes across threads.
    """
    # Single-writer: avoid opening the file repeatedly from many threads
    lines_q: Queue[str | None] = Queue(maxsize=10000)

    def _writer_worker():
        with open(output_fp, "a", encoding="utf-8") as f:
            while True:
                item = lines_q.get()
                if item is None:
                    lines_q.task_done()
                    break
                f.write(item)
                lines_q.task_done()

    writer_thread = threading.Thread(target=_writer_worker, name="writer", daemon=True)
    writer_thread.start()

    def worker(idx: int) -> Tuple[int, int]:
        row = df.iloc[idx]
        prompt = list(row.get("prompt"))
        extra_info = _to_native(row.get("extra_info", {}))
        data_source = row.get("data_source", "")
        question = extra_info.get("question") if isinstance(extra_info, dict) else None
        # For SearchR1-like preprocess, ground_truth is stored under tools_kwargs.search.create_kwargs
        gt = None
        try:
            tools_kwargs = extra_info.get("tools_kwargs", {}) if isinstance(extra_info, dict) else {}
            gt = tools_kwargs.get("search", {}).get("create_kwargs", {}).get("ground_truth")
            gt['target'] = list(gt['target'])
        except Exception:
            gt = None

        # If prompt is still not a list, try to coerce from JSON string
        if not isinstance(prompt, list):
            return (idx, 0)

        # Recover question from messages if missing
        if not question:
            question = _extract_question_from_messages(prompt)
            if not question:
                return (idx, 0)

        # Normalize to OpenAI messages structure directly
        base_messages: List[Dict[str, Any]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(role, str):
                if not isinstance(content, str):
                    content = str(content)
                base_messages.append({"role": role, "content": content})

        if not base_messages:
            return (idx, 0)

        trajs = sample_tree_for_question(
            base_messages=base_messages,
            question=question,
            query_index=idx,
            data_source=data_source,
            ground_truth=gt,
            cfg=cfg,
        )

        # Enqueue lines for single writer
        for t in trajs:
            if t.terminated:
                rec = {
                    "query_index": t.query_index,
                    "question": t.question,
                    "data_source": t.data_source,
                    "ground_truth": t.ground_truth,
                    "depth": t.depth,
                    "terminated": t.terminated,
                    "leaf_output": t.leaf_output,
                    "messages": t.messages,
                }
                lines_q.put(json.dumps(rec, ensure_ascii=False) + "\n")
        return (idx, len(trajs))

    # Run threaded
    indices = list(row_indices)
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futures = {ex.submit(worker, i): i for i in indices}
        with tqdm(total=len(indices), desc="Sampling", unit="q", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    pbar.write(f"Error on idx {i}: {e}")
                finally:
                    pbar.update(1)

    # Ensure all lines are written and close the file once
    lines_q.join()
    lines_q.put(None)
    writer_thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent tree sampling for SearchR1-like data")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/fanshengda/verl/input_data/searchR1_processed_direct/train.parquet",
        help="Path to processed parquet with prompt/extra_info fields.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rollout_data/tree_sampling",
        help="Directory to save JSONL trajectories.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--end", type=int, default=20000, help="End row index (exclusive); -1 means all")
    parser.add_argument("--max_depth", type=int, default=4, help="Max tree depth per question")
    parser.add_argument("--branching_factor", type=int, default=2, help="Branching factor per node")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent questions")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://localhost:8888/v1",
        help="OpenAI-compatible base URL for LLM server (e.g., SGLang)",
    )
    parser.add_argument("--llm_model", type=str, default="Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--llm_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--llm_max_tokens", type=int, default=4096, help="Max new tokens per completion")
    parser.add_argument("--llm_timeout_s", type=int, default=120, help="LLM request timeout seconds")
    parser.add_argument(
        "--retriever_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retriever HTTP endpoint (Search-R1-like)",
    )
    parser.add_argument("--retriever_topk", type=int, default=3, help="Retriever topk")
    parser.add_argument("--retriever_timeout_s", type=int, default=30, help="Retriever timeout seconds")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="dada",
        help="API key for OpenAI-compatible LLM server Authorization header.",
    )
    parser.add_argument("--dataset", type=str, default="nq", help="LLM model name")


    args = parser.parse_args()

    df = read_parquet_dataset(args.dataset_path)

    if args.dataset == "hotpotqa":
        df = df[df['data_source'] == 'searchR1_hotpotqa']
    elif args.dataset == "nq":
        df = df[df['data_source'] == 'searchR1_nq']
    start = max(0, int(args.start))
    end = int(args.end)
    if end < 0 or end > len(df):
        end = len(df)
    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={len(df)}")

    ensure_dir(args.output_dir)
    out_fp = os.path.join(
        args.output_dir, f"tree_sampling_{os.path.basename(args.dataset_path).split('.')[0]}_{args.dataset}_{start}_{end}_{_now_str()}.jsonl"
    )
    print(f"Writing trajectories to: {out_fp}")

    cfg = SamplingConfig(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_timeout_s=int(args.llm_timeout_s),
        branching_factor=int(args.branching_factor),
        max_depth=int(args.max_depth),
        concurrency=int(args.concurrency),
        retriever_url=args.retriever_url,
        retriever_topk=int(args.retriever_topk),
        retriever_timeout_s=int(args.retriever_timeout_s),
        openai_api_key=args.openai_api_key,
    )

    indices = list(range(start, end))
    process_rows_in_parallel(df, indices, cfg, out_fp)

    print("Done.")


if __name__ == "__main__":
    main()
