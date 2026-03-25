"""
Concurrent tree sampling that stores the full search tree in a compact node/edge format.

Compared with `tree_sampling_near_miss_data.py`, this script:
- stores every sampled node, not only terminated leaves
- stores shared `base_messages` once per question
- stores each node as a delta from its parent to avoid repeating full paths
- explicitly saves `continuation_leaf_ids_by_node`, so every prefix can be mapped to
  all future continuations without duplicating the continuation text

Output format
- One JSON record per question tree
- `base_messages` contains the shared prompt prefix
- `nodes` contains per-node deltas (`delta_messages`) plus metadata
- `child_ids_by_node` stores the explicit tree edges
- `continuation_leaf_ids_by_node` stores all reachable terminal leaf ids for each prefix

This keeps the tree explicit while being more storage-efficient than saving every
root-to-node transcript repeatedly.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from queue import Queue
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from ..utils import (
        SamplingConfig,
        _extract_question_from_messages,
        _now_str,
        _to_native,
        call_retriever,
        contains_answer,
        ensure_dir,
        openai_chat_completions,
        parse_tool_call,
        read_parquet_dataset,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from Future_Evidence_PRM.utils import (
        SamplingConfig,
        _extract_question_from_messages,
        _now_str,
        _to_native,
        call_retriever,
        contains_answer,
        ensure_dir,
        openai_chat_completions,
        parse_tool_call,
        read_parquet_dataset,
    )


@dataclass
class CompactTreeNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    node_type: str
    delta_messages: List[Dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    termination_reason: Optional[str] = None
    answer_found: bool = False
    branch_index: Optional[int] = None
    tool_call_index: Optional[int] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    query_list: Optional[List[str]] = None
    leaf_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompactTreeBuilder:
    def __init__(
        self,
        *,
        query_index: int,
        question: str,
        data_source: str,
        ground_truth: Any,
        base_messages: List[Dict[str, Any]],
        cfg: SamplingConfig,
    ) -> None:
        self.tree_id = f"q{query_index}"
        self.query_index = query_index
        self.question = question
        self.data_source = data_source
        self.ground_truth = ground_truth
        self.base_messages = base_messages
        self.cfg = cfg
        self.root_id = f"{self.tree_id}:root"

        self._counter = 0
        self.nodes: List[CompactTreeNode] = []
        self.node_lookup: Dict[str, CompactTreeNode] = {}
        self.child_ids_by_node: DefaultDict[str, List[str]] = defaultdict(list)

        self._register_node(
            CompactTreeNode(
                node_id=self.root_id,
                parent_id=None,
                depth=-1,
                node_type="root",
                delta_messages=[],
                terminated=False,
                termination_reason=None,
            )
        )

    def _register_node(self, node: CompactTreeNode) -> None:
        self.nodes.append(node)
        self.node_lookup[node.node_id] = node
        self.child_ids_by_node.setdefault(node.node_id, [])
        if node.parent_id is not None:
            self.child_ids_by_node[node.parent_id].append(node.node_id)

    def add_node(
        self,
        *,
        parent_id: str,
        depth: int,
        node_type: str,
        delta_messages: Optional[List[Dict[str, Any]]] = None,
        terminated: bool = False,
        termination_reason: Optional[str] = None,
        answer_found: bool = False,
        branch_index: Optional[int] = None,
        tool_call_index: Optional[int] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        query_list: Optional[List[str]] = None,
        leaf_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._counter += 1
        node_id = f"{self.tree_id}:n{self._counter}"
        self._register_node(
            CompactTreeNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                node_type=node_type,
                delta_messages=delta_messages or [],
                terminated=terminated,
                termination_reason=termination_reason,
                answer_found=answer_found,
                branch_index=branch_index,
                tool_call_index=tool_call_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                query_list=query_list,
                leaf_output=leaf_output,
                metadata=metadata or {},
            )
        )
        return node_id

    def mark_terminated(self, node_id: str, reason: str) -> None:
        node = self.node_lookup[node_id]
        node.terminated = True
        node.termination_reason = reason

    def _validate_tree_shape(self) -> None:
        for node in self.nodes:
            child_ids = self.child_ids_by_node[node.node_id]

            if node.node_type in {"root", "tool"}:
                if len(child_ids) > self.cfg.branching_factor:
                    raise AssertionError(
                        f"State node {node.node_id} has {len(child_ids)} children, "
                        f"exceeding branching_factor={self.cfg.branching_factor}"
                    )
                invalid_children = [
                    child_id
                    for child_id in child_ids
                    if self.node_lookup[child_id].node_type not in {"assistant", "error"}
                ]
                if invalid_children:
                    raise AssertionError(
                        f"State node {node.node_id} has invalid child types: {invalid_children[:5]}"
                    )

            elif node.node_type == "assistant":
                if len(child_ids) > 1:
                    raise AssertionError(
                        f"Assistant node {node.node_id} has {len(child_ids)} children; "
                        "one assistant completion must expand to at most one child state"
                    )
                invalid_children = [child_id for child_id in child_ids if self.node_lookup[child_id].node_type != "tool"]
                if invalid_children:
                    raise AssertionError(
                        f"Assistant node {node.node_id} has non-tool children: {invalid_children[:5]}"
                    )

            elif child_ids:
                raise AssertionError(f"Leaf-like node {node.node_id} of type {node.node_type} should not have children")

    def finalize(self) -> Dict[str, Any]:
        terminal_node_ids: List[str] = []

        for node in self.nodes:
            self.child_ids_by_node.setdefault(node.node_id, [])

        for node in self.nodes:
            if node.node_type == "root":
                continue

            if self.child_ids_by_node[node.node_id]:
                continue

            if not node.terminated:
                if node.node_type == "tool":
                    node.terminated = True
                    node.termination_reason = "max_depth"
                else:
                    node.terminated = True
                    node.termination_reason = node.termination_reason or "dead_end"

            terminal_node_ids.append(node.node_id)

        continuation_leaf_ids_by_node: Dict[str, List[str]] = {}

        def collect_leaf_ids(node_id: str) -> List[str]:
            if node_id in continuation_leaf_ids_by_node:
                return continuation_leaf_ids_by_node[node_id]

            child_ids = self.child_ids_by_node[node_id]
            if not child_ids:
                leaf_ids = [] if node_id == self.root_id else [node_id]
            else:
                leaf_ids = []
                for child_id in child_ids:
                    leaf_ids.extend(collect_leaf_ids(child_id))

            continuation_leaf_ids_by_node[node_id] = leaf_ids
            return leaf_ids

        collect_leaf_ids(self.root_id)
        self._validate_tree_shape()

        max_nodes = _max_possible_nodes(self.cfg.branching_factor, self.cfg.max_depth)
        actual_nodes = len(self.nodes)
        if actual_nodes > max_nodes:
            raise AssertionError(
                f"Tree {self.tree_id} has {actual_nodes} nodes, exceeding theoretical max {max_nodes} "
                f"for branching_factor={self.cfg.branching_factor}, max_depth={self.cfg.max_depth}"
            )

        return {
            "tree_id": self.tree_id,
            "query_index": self.query_index,
            "question": self.question,
            "data_source": self.data_source,
            "ground_truth": self.ground_truth,
            "base_messages": self.base_messages,
            "root_id": self.root_id,
            "max_depth": self.cfg.max_depth,
            "branching_factor": self.cfg.branching_factor,
            "nodes": [asdict(node) for node in self.nodes],
            "child_ids_by_node": dict(self.child_ids_by_node),
            "continuation_leaf_ids_by_node": continuation_leaf_ids_by_node,
            "terminal_node_ids": terminal_node_ids,
        }


def _normalize_prompt_messages(prompt: Any) -> List[Dict[str, Any]]:
    prompt = _to_native(prompt)
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    elif not isinstance(prompt, list):
        try:
            prompt = list(prompt)
        except TypeError:
            return []

    base_messages: List[Dict[str, Any]] = []
    for msg in prompt:
        msg = _to_native(msg)
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = str(content)
        normalized = {"role": role, "content": content}
        if "tool_calls" in msg and msg["tool_calls"]:
            normalized["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            normalized["tool_call_id"] = msg["tool_call_id"]
        if "name" in msg:
            normalized["name"] = msg["name"]
        base_messages.append(normalized)
    return base_messages


def _normalize_query_list(arguments: Any) -> List[str]:
    args = arguments or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}

    if not isinstance(args, dict):
        return []

    query_list = args.get("query_list")
    if not query_list and "query" in args:
        query_list = [args["query"]]

    if isinstance(query_list, str):
        return [query_list]
    if isinstance(query_list, list):
        return [str(item) for item in query_list if str(item).strip()]
    return []


def _max_possible_nodes(branching_factor: int, max_depth: int) -> int:
    if branching_factor < 0 or max_depth < 0:
        raise ValueError(f"Invalid tree config: branching_factor={branching_factor}, max_depth={max_depth}")
    if branching_factor == 0 or max_depth == 0:
        return 1
    if branching_factor == 1:
        return 1 + (2 * max_depth)
    total_per_type = sum(branching_factor**depth for depth in range(1, max_depth + 1))
    return 1 + (2 * total_per_type)


def _json_safe(obj: Any) -> Any:
    obj = _to_native(obj)

    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_safe(item) for item in obj]

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass

    return obj


def sample_tree_for_question(
    base_messages: List[Dict[str, Any]],
    question: str,
    query_index: int,
    data_source: str,
    ground_truth: Any,
    cfg: SamplingConfig,
) -> Dict[str, Any]:
    builder = CompactTreeBuilder(
        query_index=query_index,
        question=question,
        data_source=data_source,
        ground_truth=ground_truth,
        base_messages=base_messages,
        cfg=cfg,
    )

    frontier: List[Tuple[str, List[Dict[str, Any]]]] = [(builder.root_id, list(base_messages))]

    for depth in range(cfg.max_depth):
        next_frontier: List[Tuple[str, List[Dict[str, Any]]]] = []

        for parent_state_id, node_messages in frontier:
            try:
                assistant_messages = openai_chat_completions(
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
            except Exception as exc:
                fail_msg = {"role": "assistant", "content": f"<error> LLM error: {exc} </error>"}
                builder.add_node(
                    parent_id=parent_state_id,
                    depth=depth,
                    node_type="error",
                    delta_messages=[fail_msg],
                    terminated=True,
                    termination_reason="llm_error",
                    leaf_output=fail_msg["content"],
                    metadata={"error": str(exc)},
                )
                continue

            for branch_index, assistant in enumerate(assistant_messages):
                content = assistant.get("content", "")
                tool_calls = assistant.get("tool_calls", []) or []
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                answer_found = contains_answer(content)
                assistant_node_id = builder.add_node(
                    parent_id=parent_state_id,
                    depth=depth,
                    node_type="assistant",
                    delta_messages=[assistant_msg],
                    terminated=answer_found,
                    termination_reason="answer" if answer_found else None,
                    answer_found=answer_found,
                    branch_index=branch_index,
                    leaf_output=content,
                    metadata={"structured_tool_call_count": len(tool_calls)},
                )

                if answer_found:
                    continue

                traj_messages = node_messages + [assistant_msg]

                structured_calls: List[Dict[str, Any]] = []
                for tool_call in tool_calls:
                    fn = (tool_call.get("function") or {})
                    structured_calls.append(
                        {
                            "id": tool_call.get("id"),
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
                            {
                                "id": None,
                                "name": legacy.get("name"),
                                "arguments": legacy.get("arguments", {}),
                            }
                        ]

                saw_any_tool_syntax = bool(structured_calls)
                saw_search_call = False
                saw_queryless_search = False
                tool_messages: List[Dict[str, Any]] = []
                tool_call_summaries: List[Dict[str, Any]] = []
                combined_query_list: List[str] = []

                for tool_call_index, call in enumerate(structured_calls):
                    if call.get("name") != "search":
                        continue

                    saw_search_call = True
                    query_list = _normalize_query_list(call.get("arguments"))
                    if not query_list:
                        saw_queryless_search = True
                        continue

                    retriever_error = None
                    try:
                        retrieved = call_retriever(
                            retrieval_url=cfg.retriever_url,
                            query_list=query_list,
                            topk=cfg.retriever_topk,
                            timeout=cfg.retriever_timeout_s,
                        )
                    except Exception as exc:
                        retrieved = f"Search error: {exc}"
                        retriever_error = str(exc)

                    if legacy_mode:
                        tool_msg: Dict[str, Any] = {
                            "role": "user",
                            "content": f"<tool_response>\n{retrieved}\n</tool_response>",
                        }
                    else:
                        tool_msg = {"role": "tool", "content": retrieved, "name": "search"}
                        if call.get("id"):
                            tool_msg["tool_call_id"] = call["id"]

                    tool_messages.append(tool_msg)
                    combined_query_list.extend(query_list)
                    tool_call_summaries.append(
                        {
                            "tool_call_index": tool_call_index,
                            "tool_name": "search",
                            "tool_call_id": call.get("id"),
                            "query_list": query_list,
                            "retriever_error": retriever_error,
                        }
                    )

                    if cfg.pause_between_calls_s > 0:
                        time.sleep(cfg.pause_between_calls_s)

                if tool_messages:
                    tool_node_id = builder.add_node(
                        parent_id=assistant_node_id,
                        depth=depth,
                        node_type="tool",
                        delta_messages=tool_messages,
                        terminated=False,
                        termination_reason=None,
                        tool_call_index=tool_call_summaries[0]["tool_call_index"] if len(tool_call_summaries) == 1 else None,
                        tool_name="search",
                        tool_call_id=tool_call_summaries[0]["tool_call_id"] if len(tool_call_summaries) == 1 else None,
                        query_list=combined_query_list,
                        metadata={
                            "legacy_mode": legacy_mode,
                            "tool_calls": tool_call_summaries,
                            "valid_search_call_count": len(tool_call_summaries),
                        },
                    )

                    next_frontier.append((tool_node_id, traj_messages + tool_messages))
                    continue

                if saw_any_tool_syntax:
                    if saw_search_call:
                        reason = "empty_search_query" if saw_queryless_search else "invalid_search_call"
                    else:
                        reason = "unsupported_tool"
                else:
                    reason = "no_tool_call"
                builder.mark_terminated(assistant_node_id, reason)

        if not next_frontier:
            break
        frontier = next_frontier

    return builder.finalize()


def process_rows_in_parallel(
    df: pd.DataFrame,
    row_indices: Iterable[int],
    cfg: SamplingConfig,
    output_fp: str,
) -> None:
    lines_q: Queue[Optional[str]] = Queue(maxsize=4096)

    def writer_worker() -> None:
        with open(output_fp, "a", encoding="utf-8") as fout:
            while True:
                item = lines_q.get()
                if item is None:
                    lines_q.task_done()
                    break
                fout.write(item)
                lines_q.task_done()

    writer_thread = threading.Thread(target=writer_worker, name="writer", daemon=True)
    writer_thread.start()

    def worker(idx: int) -> Tuple[int, int]:
        row = df.iloc[idx]
        prompt = row.get("prompt")
        extra_info = _to_native(row.get("extra_info", {}))
        data_source = row.get("data_source", "")
        question = extra_info.get("question") if isinstance(extra_info, dict) else None

        ground_truth = None
        try:
            tools_kwargs = extra_info.get("tools_kwargs", {}) if isinstance(extra_info, dict) else {}
            ground_truth = tools_kwargs.get("search", {}).get("create_kwargs", {}).get("ground_truth")
            if isinstance(ground_truth, dict) and isinstance(ground_truth.get("target"), (list, tuple, set)):
                ground_truth["target"] = list(ground_truth["target"])
        except Exception:
            ground_truth = None

        base_messages = _normalize_prompt_messages(prompt)
        if not base_messages:
            return (idx, 0)

        if not question:
            question = _extract_question_from_messages(base_messages)
            if not question:
                return (idx, 0)

        tree_record = sample_tree_for_question(
            base_messages=base_messages,
            question=question,
            query_index=idx,
            data_source=data_source,
            ground_truth=ground_truth,
            cfg=cfg,
        )

        lines_q.put(json.dumps(_json_safe(tree_record), ensure_ascii=False) + "\n")
        return (idx, len(tree_record["nodes"]))

    indices = list(row_indices)
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
        futures = {executor.submit(worker, idx): idx for idx in indices}
        with tqdm(total=len(indices), desc="Sampling", unit="q", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    pbar.write(f"Error on idx {idx}: {exc}")
                finally:
                    pbar.update(1)

    lines_q.join()
    lines_q.put(None)
    writer_thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent tree sampling with compact all-node storage")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/fanshengda/verl/input_data/searchR1_processed_direct/train.parquet",
        help="Path to processed parquet with prompt/extra_info fields.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rollout_data/tree_sampling_compact",
        help="Directory to save compact tree JSONL records.",
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
        help="OpenAI-compatible base URL for LLM server.",
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
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset subset: `nq` or `hotpotqa`")

    args = parser.parse_args()

    df = read_parquet_dataset(args.dataset_path)
    if args.dataset == "hotpotqa":
        df = df[df["data_source"] == "searchR1_hotpotqa"]
    elif args.dataset == "nq":
        df = df[df["data_source"] == "searchR1_nq"]

    start = max(0, int(args.start))
    end = int(args.end)
    if end < 0 or end > len(df):
        end = len(df)
    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={len(df)}")

    ensure_dir(args.output_dir)
    output_fp = os.path.join(
        args.output_dir,
        f"tree_sampling_compact_{os.path.basename(args.dataset_path).split('.')[0]}_{args.dataset}_{start}_{end}_{_now_str()}.jsonl",
    )
    print(f"Writing compact trees to: {output_fp}")

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

    process_rows_in_parallel(df, range(start, end), cfg, output_fp)
    print("Done.")


if __name__ == "__main__":
    main()
