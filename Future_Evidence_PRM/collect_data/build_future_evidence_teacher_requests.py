from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..apb_alignment import (
        DEFAULT_SEARCH_TOOLS,
        HINDSIGHT_STEP_SYSTEM_PROMPT,
        HINDSIGHT_STEP_ICL_EXAMPLES,
        HINDSIGHT_STEP_USER_INSTRUCTIONS,
        ensure_tools,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from Future_Evidence_PRM.apb_alignment import (
        DEFAULT_SEARCH_TOOLS,
        HINDSIGHT_STEP_SYSTEM_PROMPT,
        HINDSIGHT_STEP_ICL_EXAMPLES,
        HINDSIGHT_STEP_USER_INSTRUCTIONS,
        ensure_tools,
    )


def _to_native(obj: Any) -> Any:
    if isinstance(obj, (list, dict)):
        return obj
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        return obj.as_py()
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        return obj.to_pylist()
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    return obj


def _safe_native(obj: Any) -> Any:
    obj = _to_native(obj)
    if isinstance(obj, dict):
        return {str(key): _safe_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_safe_native(item) for item in obj]
    return obj


def _format_reference_annotation_examples(examples: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for example_index, example in enumerate(examples, start=1):
        blocks.append(
            f"REFERENCE_ANNOTATION_EXAMPLE_{example_index}:\n"
            + json.dumps(_safe_native(example), ensure_ascii=False, indent=2)
        )
    return "\n\n".join(blocks)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_check(prediction: str, golden_answers: Sequence[str] | str) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(str(golden_answer)) == normalized_prediction:
            return 1
    return 0


def subem_check(prediction: str, golden_answers: Sequence[str] | str) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(str(golden_answer)) in normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_ground_truth(ground_truth: Any) -> Optional[Dict[str, Any]]:
    ground_truth = _safe_native(ground_truth)
    if ground_truth is None:
        return None
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target")
        if target is None:
            return None
        if isinstance(target, str):
            target = [target]
        elif not isinstance(target, list):
            target = [str(target)]
        return {"target": [str(item) for item in target]}
    if isinstance(ground_truth, str):
        return {"target": [ground_truth]}
    if isinstance(ground_truth, list):
        return {"target": [str(item) for item in ground_truth]}
    return {"target": [str(ground_truth)]}


def _extract_leaf_output(node: Dict[str, Any]) -> str:
    if isinstance(node.get("leaf_output"), str):
        return node["leaf_output"]
    for message in reversed(node.get("delta_messages", []) or []):
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _score_leaf(node: Dict[str, Any], ground_truth: Dict[str, Any], score_method: str) -> int:
    answer = extract_solution(_extract_leaf_output(node))
    if answer is None:
        return 0
    if score_method == "subem":
        return int(subem_check(answer, ground_truth["target"]))
    return int(em_check(answer, ground_truth["target"]))


def _build_node_map(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(node["node_id"]): _safe_native(node) for node in record.get("nodes", [])}


def _chain_to_root(node_map: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
    chain: List[str] = []
    current_id = node_id
    while True:
        node = node_map[current_id]
        chain.append(current_id)
        parent_id = node.get("parent_id")
        if parent_id is None or str(parent_id).endswith(":root"):
            break
        current_id = str(parent_id)
    chain.reverse()
    return chain


def _reconstruct_messages(
    base_messages: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    node_chain: List[str],
) -> List[Dict[str, Any]]:
    messages = [_safe_native(message) for message in base_messages]
    for node_id in node_chain:
        node = node_map[node_id]
        for message in node.get("delta_messages", []) or []:
            messages.append(_safe_native(message))
    return messages


def _reconstruct_trajectory_with_indices(
    base_messages: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    node_chain: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[int]]:
    messages = [_safe_native(message) for message in base_messages]
    assistant_message_index_by_node_id: Dict[str, int] = {}

    for node_id in node_chain:
        node = node_map[node_id]
        delta_messages = [_safe_native(message) for message in node.get("delta_messages", []) or []]

        if node.get("node_type") == "assistant":
            assistant_index: Optional[int] = None
            for message in delta_messages:
                if assistant_index is None and isinstance(message, dict) and message.get("role") == "assistant":
                    assistant_index = len(messages)
                messages.append(message)
            if assistant_index is None:
                raise ValueError(f"Assistant node {node_id} does not contain an assistant message")
            assistant_message_index_by_node_id[str(node_id)] = assistant_index
            continue

        messages.extend(delta_messages)

    assistant_indices = [
        assistant_message_index_by_node_id[str(node_id)]
        for node_id in node_chain
        if node_map[node_id].get("node_type") == "assistant"
    ]
    return messages, assistant_message_index_by_node_id, assistant_indices


def _future_messages_after_prefix(
    node_map: Dict[str, Dict[str, Any]],
    prefix_chain: List[str],
    leaf_chain: List[str],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    suffix_node_ids = leaf_chain[len(prefix_chain):]
    for node_id in suffix_node_ids:
        for message in node_map[node_id].get("delta_messages", []) or []:
            messages.append(_safe_native(message))
    return messages


def _current_step_context(
    trajectory_messages: List[Dict[str, Any]],
    assistant_message_index: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    current_messages = [_safe_native(trajectory_messages[assistant_message_index])]

    next_index = assistant_message_index + 1
    if next_index < len(trajectory_messages):
        next_message = _safe_native(trajectory_messages[next_index])
        if isinstance(next_message, dict) and next_message.get("role") == "tool":
            current_messages.append(next_message)

    return current_messages, len(current_messages) > 1


def _trim_future_messages_for_current_step(
    future_messages: List[Dict[str, Any]],
    *,
    include_immediate_tool: bool,
) -> List[Dict[str, Any]]:
    trimmed_messages = [_safe_native(message) for message in future_messages]
    if (
        include_immediate_tool
        and trimmed_messages
        and isinstance(trimmed_messages[0], dict)
        and trimmed_messages[0].get("role") == "tool"
    ):
        return trimmed_messages[1:]
    return trimmed_messages


def _parse_tool_call_queries(content: str) -> List[str]:
    queries: List[str] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        payload = match.group(1).strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        arguments = _safe_native(data.get("arguments", {}))
        queries.extend(_normalize_query_list(arguments))
    return queries


def _normalize_query_list(arguments: Any) -> List[str]:
    arguments = _safe_native(arguments) or {}
    if not isinstance(arguments, dict):
        return []
    query_list = arguments.get("query_list")
    if isinstance(query_list, list):
        return [str(item).strip() for item in query_list if str(item).strip()]
    query = arguments.get("query")
    if query is None:
        return []
    query_text = str(query).strip()
    return [query_text] if query_text else []


def _extract_queries_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    queries: List[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = _safe_native(message.get("tool_calls")) or []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = _safe_native(tool_call.get("function", {}))
            if not isinstance(function, dict) or function.get("name") != "search":
                continue
            queries.extend(_normalize_query_list(function.get("arguments", {})))
        content = message.get("content")
        if isinstance(content, str):
            queries.extend(_parse_tool_call_queries(content))
    return queries


def _leaf_outcome_bucket(node: Dict[str, Any], leaf_success: int) -> str:
    if leaf_success == 1:
        return "success"
    termination_reason = str(node.get("termination_reason") or "other")
    if termination_reason == "answer":
        return "wrong_answer"
    return termination_reason


def _leaf_signature(
    leaf_id: str,
    *,
    node_map: Dict[str, Dict[str, Any]],
    prefix_chain: List[str],
    leaf_success_by_id: Dict[str, int],
) -> Dict[str, Any]:
    leaf_node = node_map[leaf_id]
    leaf_chain = _chain_to_root(node_map, leaf_id)
    future_messages = _future_messages_after_prefix(node_map, prefix_chain, leaf_chain)
    final_answer = extract_solution(_extract_leaf_output(leaf_node)) or ""
    queries = _extract_queries_from_messages(future_messages)
    return {
        "leaf_id": leaf_id,
        "bucket": _leaf_outcome_bucket(leaf_node, leaf_success_by_id.get(leaf_id, 0)),
        "termination_reason": str(leaf_node.get("termination_reason") or "other"),
        "final_answer": normalize_answer(final_answer) if final_answer else "",
        "query_set": sorted(set(normalize_answer(query) for query in queries if query)),
        "future_text_key": normalize_answer(_extract_leaf_output(leaf_node))[:200],
    }


def _signature_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    distance = 0.0
    if a["bucket"] != b["bucket"]:
        distance += 4.0
    if a["termination_reason"] != b["termination_reason"]:
        distance += 2.0
    if a["final_answer"] != b["final_answer"]:
        distance += 2.0
    distance += float(len(set(a["query_set"]).symmetric_difference(set(b["query_set"]))))
    if a["future_text_key"] != b["future_text_key"]:
        distance += 1.0
    return distance


def _select_diverse_leaf_ids(
    candidate_leaf_ids: List[str],
    *,
    num_future_trajs_per_node: int,
    node_map: Dict[str, Dict[str, Any]],
    prefix_chain: List[str],
    leaf_success_by_id: Dict[str, int],
) -> List[str]:
    if len(candidate_leaf_ids) <= num_future_trajs_per_node:
        return sorted(candidate_leaf_ids)

    signatures = {
        leaf_id: _leaf_signature(
            leaf_id,
            node_map=node_map,
            prefix_chain=prefix_chain,
            leaf_success_by_id=leaf_success_by_id,
        )
        for leaf_id in candidate_leaf_ids
    }

    selected: List[str] = []
    remaining = set(candidate_leaf_ids)

    bucket_to_leaf_ids: Dict[str, List[str]] = {}
    for leaf_id in sorted(candidate_leaf_ids):
        bucket = signatures[leaf_id]["bucket"]
        bucket_to_leaf_ids.setdefault(bucket, []).append(leaf_id)

    for bucket in sorted(bucket_to_leaf_ids):
        if len(selected) >= num_future_trajs_per_node:
            break
        leaf_id = bucket_to_leaf_ids[bucket][0]
        selected.append(leaf_id)
        remaining.remove(leaf_id)

    while len(selected) < num_future_trajs_per_node and remaining:
        if not selected:
            next_leaf_id = sorted(remaining)[0]
        else:
            next_leaf_id = max(
                sorted(remaining),
                key=lambda leaf_id: (
                    min(
                        _signature_distance(signatures[leaf_id], signatures[selected_leaf_id])
                        for selected_leaf_id in selected
                    ),
                    leaf_id,
                ),
            )
        selected.append(next_leaf_id)
        remaining.remove(next_leaf_id)

    return selected


def _format_message_text(message: Dict[str, Any]) -> str:
    role = str(message.get("role"))
    parts: List[str] = [f"[{role}]"]

    if message.get("name") is not None:
        parts.append(f"name={message['name']}")

    tool_calls = _safe_native(message.get("tool_calls")) or []
    if tool_calls:
        parts.append("tool_calls=" + json.dumps(tool_calls, ensure_ascii=False))

    content = message.get("content")
    if content is not None and str(content).strip():
        parts.append(str(content))

    return "\n".join(parts)


def _format_messages_block(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return "[No additional messages]"
    return "\n\n".join(_format_message_text(message) for message in messages)


def _build_teacher_messages(
    *,
    question: Any,
    ground_truth: Dict[str, Any],
    task_description: Any,
    tools: List[Dict[str, Any]],
    trajectory_messages: List[Dict[str, Any]],
    assistant_message_index: int,
    current_step_messages: List[Dict[str, Any]],
    selected_futures: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    if assistant_message_index < 0 or assistant_message_index >= len(trajectory_messages):
        raise ValueError(f"assistant_message_index out of range: {assistant_message_index}")

    payload = {
        "question": question,
        "task_description": task_description,
        "ground_truth": list(ground_truth["target"]),
        "tools": ensure_tools(tools),
        "messages_before_current_step": list(enumerate(trajectory_messages[:assistant_message_index])),
        "current_step_messages": _safe_native(current_step_messages),
        "hidden_future_trajectories": [
            {
                "messages": _safe_native(future["future_messages"]),
                "success": int(future["success"]),
                "termination_reason": str(future["termination_reason"]),
            }
            for future in selected_futures
        ],
        "notes": {
            "step_definition": (
                "Treat the tool response as part of the same local step context. "
            ),
            "hindsight": (
                "Ground truth and hidden_future_trajectories are hindsight evidence for teacher distillation. "
                "Use them to infer latent future implications of the current step, but do not mention them "
                "explicitly in explanations unless absolutely necessary. Future implications should describe "
                "1 or 2 likely future trends rather than repeating the current observation."
            ),
            "output_requirements": (
                "Return one local step_assessment, one three-way step_label, one structured "
                "future_implication object with 1 or 2 future directions, and one trajectory-grounded "
                "explanation for the current step."
            ),
        },
    }

    return [
        {"role": "system", "content": HINDSIGHT_STEP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                HINDSIGHT_STEP_USER_INSTRUCTIONS
                + "\n\n"
                + _format_reference_annotation_examples(HINDSIGHT_STEP_ICL_EXAMPLES)
                + "\n\nHINDSIGHT_STEP_JSON:\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def _split_dataframe_by_query(
    dataframe: pd.DataFrame,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")
    if dataframe.empty:
        return dataframe.copy(), dataframe.copy()

    unique_queries = dataframe["query_index"].drop_duplicates().tolist()
    if len(unique_queries) <= 1:
        return dataframe.copy(), dataframe.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_queries)
    validation_size = min(max(1, int(round(len(unique_queries) * validation_ratio))), len(unique_queries) - 1)
    validation_queries = set(unique_queries[:validation_size])

    validation_mask = dataframe["query_index"].isin(validation_queries)
    validation_df = dataframe[validation_mask].copy().reset_index(drop=True)
    train_df = dataframe[~validation_mask].copy().reset_index(drop=True)
    return train_df, validation_df


def _derive_split_paths(output_path: Path) -> Tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_train{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_validation{output_path.suffix}"),
    )


def build_dataset(
    input_path: Path,
    *,
    score_method: str,
    num_future_trajs_per_node: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "tree_count": 0,
        "sampled_trajectory_count": 0,
        "teacher_sample_count": 0,
        "supervised_assistant_step_count": 0,
        "skipped_no_ground_truth": 0,
        "skipped_no_first_layer_assistant": 0,
        "skipped_no_leaf_under_first_layer_assistant": 0,
        "total_candidate_futures": 0,
        "total_selected_futures": 0,
    }

    for record in tqdm(_iter_jsonl(input_path), desc="Building teacher data", unit="tree", dynamic_ncols=True):
        stats["tree_count"] += 1
        record = _safe_native(record)
        ground_truth = _normalize_ground_truth(record.get("ground_truth"))
        if ground_truth is None:
            stats["skipped_no_ground_truth"] += 1
            continue

        node_map = _build_node_map(record)
        terminal_node_ids = [str(node_id) for node_id in record.get("terminal_node_ids", []) or []]
        continuation_leaf_ids_by_node = _safe_native(record.get("continuation_leaf_ids_by_node", {}))
        child_ids_by_node = _safe_native(record.get("child_ids_by_node", {}))
        base_messages = _safe_native(record.get("base_messages", []))
        root_id = str(record.get("root_id"))

        leaf_success_by_id = {
            leaf_id: _score_leaf(node_map[leaf_id], ground_truth, score_method)
            for leaf_id in terminal_node_ids
            if leaf_id in node_map
        }

        first_layer_assistant_ids = [
            str(node_id)
            for node_id in child_ids_by_node.get(root_id, [])
            if str(node_id) in node_map and node_map[str(node_id)]["node_type"] == "assistant"
        ]
        if not first_layer_assistant_ids:
            stats["skipped_no_first_layer_assistant"] += 1
            continue

        for first_assistant_id in first_layer_assistant_ids:
            candidate_leaf_ids = [
                str(node_id)
                for node_id in continuation_leaf_ids_by_node.get(first_assistant_id, [])
                if str(node_id) in node_map and str(node_id) in leaf_success_by_id
            ]
            if not candidate_leaf_ids:
                stats["skipped_no_leaf_under_first_layer_assistant"] += 1
                continue

            sampled_leaf_id = candidate_leaf_ids[0]
            trajectory_sample_id = f"{record.get('tree_id')}::{sampled_leaf_id}"
            stats["sampled_trajectory_count"] += 1

            node_chain = _chain_to_root(node_map, sampled_leaf_id)
            trajectory_messages, assistant_message_index_by_node_id, assistant_message_indices = (
                _reconstruct_trajectory_with_indices(base_messages, node_map, node_chain)
            )
            tools = ensure_tools(record.get("tools") or DEFAULT_SEARCH_TOOLS)
            task_description = record.get("task_description")
            trajectory_final_label = 1 if int(leaf_success_by_id.get(sampled_leaf_id, 0)) == 1 else -1

            for assistant_node_id in node_chain:
                if node_map[assistant_node_id]["node_type"] != "assistant":
                    continue

                prefix_node_id = assistant_node_id
                prefix_chain = _chain_to_root(node_map, prefix_node_id)
                assistant_message_index = assistant_message_index_by_node_id[str(assistant_node_id)]
                current_step_messages, include_immediate_tool = _current_step_context(
                    trajectory_messages,
                    assistant_message_index,
                )

                future_candidate_leaf_ids = [
                    str(node_id)
                    for node_id in continuation_leaf_ids_by_node.get(prefix_node_id, [])
                    if str(node_id) in node_map and str(node_id) in leaf_success_by_id
                ]
                if not future_candidate_leaf_ids and prefix_node_id in leaf_success_by_id:
                    future_candidate_leaf_ids = [prefix_node_id]
                if not future_candidate_leaf_ids:
                    raise ValueError(f"Node {prefix_node_id} has no continuation leaves")

                selected_leaf_ids = _select_diverse_leaf_ids(
                    future_candidate_leaf_ids,
                    num_future_trajs_per_node=num_future_trajs_per_node,
                    node_map=node_map,
                    prefix_chain=prefix_chain,
                    leaf_success_by_id=leaf_success_by_id,
                )

                termination_reason_count = Counter(
                    str(node_map[leaf_id].get("termination_reason") or "other") for leaf_id in future_candidate_leaf_ids
                )
                success_count = sum(leaf_success_by_id.get(leaf_id, 0) for leaf_id in future_candidate_leaf_ids)
                future_stats = {
                    "total_future_count": len(future_candidate_leaf_ids),
                    "success_count": success_count,
                    "failure_count": len(future_candidate_leaf_ids) - success_count,
                    "success_rate": round(success_count / len(future_candidate_leaf_ids), 6),
                    "termination_reason_count": dict(sorted(termination_reason_count.items())),
                }

                selected_futures: List[Dict[str, Any]] = []
                for selected_leaf_id in selected_leaf_ids:
                    leaf_chain = _chain_to_root(node_map, selected_leaf_id)
                    future_messages = _trim_future_messages_for_current_step(
                        _future_messages_after_prefix(node_map, prefix_chain, leaf_chain),
                        include_immediate_tool=include_immediate_tool,
                    )
                    leaf_node = node_map[selected_leaf_id]
                    selected_futures.append(
                        {
                            "leaf_id": selected_leaf_id,
                            "success": int(leaf_success_by_id.get(selected_leaf_id, 0)),
                            "outcome_bucket": _leaf_outcome_bucket(
                                leaf_node, leaf_success_by_id.get(selected_leaf_id, 0)
                            ),
                            "termination_reason": str(leaf_node.get("termination_reason") or "other"),
                            "final_answer": extract_solution(_extract_leaf_output(leaf_node)),
                            "future_messages": future_messages,
                            "future_text": _format_messages_block(future_messages),
                        }
                    )

                teacher_messages = _build_teacher_messages(
                    question=record.get("question"),
                    ground_truth=ground_truth,
                    task_description=task_description,
                    tools=tools,
                    trajectory_messages=trajectory_messages,
                    assistant_message_index=assistant_message_index,
                    current_step_messages=current_step_messages,
                    selected_futures=selected_futures,
                )

                rows.append(
                    {
                        "messages": teacher_messages,
                        "query_index": record.get("query_index"),
                        "tree_id": record.get("tree_id"),
                        "trajectory_sample_id": trajectory_sample_id,
                        "trajectory_terminal_node_id": sampled_leaf_id,
                        "assistant_node_id": assistant_node_id,
                        "prefix_node_id": prefix_node_id,
                        "assistant_message_index": assistant_message_index,
                        "assistant_message_indices": assistant_message_indices,
                        "data_source": record.get("data_source"),
                        "question": record.get("question"),
                        "task_description": task_description,
                        "tools": tools,
                        "ground_truth": ground_truth,
                        "trajectory_messages": trajectory_messages,
                        "trajectory_final_label": trajectory_final_label,
                        "selected_future_leaf_ids": selected_leaf_ids,
                        "selected_future_trajectories": selected_futures,
                        "future_stats": future_stats,
                    }
                )
                # print('da')

                stats["teacher_sample_count"] += 1
                stats["supervised_assistant_step_count"] += 1
                stats["total_candidate_futures"] += len(future_candidate_leaf_ids)
                stats["total_selected_futures"] += len(selected_leaf_ids)

    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build future-evidence teacher-input JSONL from compact tree JSONL")
    parser.add_argument("--input", type=str, required=True, help="Compact tree JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--score_method", type=str, default="strict", choices=["strict", "subem"])
    parser.add_argument(
        "--num_future_trajs_per_node",
        type=int,
        default=4,
        help="How many diverse future continuations to keep for each supervised node",
    )
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, stats = build_dataset(
        input_path,
        score_method=args.score_method,
        num_future_trajs_per_node=int(args.num_future_trajs_per_node),
    )
    dataframe = pd.DataFrame(rows)
    train_df, validation_df = _split_dataframe_by_query(
        dataframe,
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
    )
    train_output_path, validation_output_path = _derive_split_paths(output_path)

    _write_jsonl(train_output_path, train_df.to_dict(orient="records"))
    _write_jsonl(validation_output_path, validation_df.to_dict(orient="records"))

    summary = {
        **stats,
        "avg_steps_per_sampled_trajectory": (
            round(stats["teacher_sample_count"] / stats["sampled_trajectory_count"], 6)
            if stats["sampled_trajectory_count"]
            else 0.0
        ),
        "avg_candidate_futures_per_node": (
            round(stats["total_candidate_futures"] / stats["teacher_sample_count"], 6)
            if stats["teacher_sample_count"]
            else 0.0
        ),
        "avg_selected_futures_per_node": (
            round(stats["total_selected_futures"] / stats["teacher_sample_count"], 6)
            if stats["teacher_sample_count"]
            else 0.0
        ),
        "num_future_trajs_per_node": int(args.num_future_trajs_per_node),
        "train_sample_count": int(len(train_df)),
        "validation_sample_count": int(len(validation_df)),
        "validation_ratio": float(args.validation_ratio),
        "seed": int(args.seed),
        "input": str(input_path),
        "train_output": str(train_output_path),
        "validation_output": str(validation_output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
