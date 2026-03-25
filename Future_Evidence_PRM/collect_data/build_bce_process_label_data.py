"""
Build trajectory-level BCE process-label data from compact tree JSONL.

One terminal trajectory becomes one training sample.
If a trajectory has 5 assistant steps, the sample contains 5 step labels.

For each assistant step t on that trajectory:
- label[t] = 1 if that prefix has at least one successful continuation leaf
- label[t] = 0 otherwise
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches for relevant information based on queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries",
                    },
                    "query": {"type": "string", "description": "Single query"},
                },
                "required": ["query_list"],
            },
        },
    }
]


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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


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


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    messages = _safe_native(messages)
    if not isinstance(messages, list):
        raise ValueError(f"prompt must be a list, got {type(messages).__name__}")

    normalized: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"prompt item must be a dict, got {type(message).__name__}")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"message.role must be str, got {type(role).__name__}")

        item: Dict[str, Any] = {"role": role}
        if "content" in message:
            content = message.get("content")
            item["content"] = content if content is None or isinstance(content, str) else str(content)
        if "name" in message and message.get("name") is not None:
            item["name"] = str(message.get("name"))
        if "tool_call_id" in message and message.get("tool_call_id") is not None:
            item["tool_call_id"] = str(message.get("tool_call_id"))
        if "tool_calls" in message and message.get("tool_calls") is not None:
            tool_calls = _safe_native(message.get("tool_calls"))
            if not isinstance(tool_calls, list):
                raise ValueError(f"message.tool_calls must be list, got {type(tool_calls).__name__}")
            item["tool_calls"] = tool_calls
        normalized.append(item)
    return normalized


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    tools = _safe_native(tools)
    if not isinstance(tools, list):
        raise ValueError(f"tools must be a list, got {type(tools).__name__}")

    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"tool item must be a dict, got {type(tool).__name__}")
        normalized.append(_safe_native(tool))
    return normalized


def _json_dumps(obj: Any) -> str:
    return json.dumps(_safe_native(obj), ensure_ascii=False)


def build_dataset(input_path: Path, score_method: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stats = {
        "tree_count": 0,
        "sample_count": 0,
        "first_layer_assistant_count": 0,
        "positive_step_count": 0,
        "negative_step_count": 0,
        "skipped_no_ground_truth": 0,
        "skipped_no_first_layer_assistant": 0,
        "skipped_no_leaf_under_first_layer_assistant": 0,
        "query_count": 0,
        "queries_with_any_positive_trajectory": 0,
        "trajectories_with_any_positive_step": 0,
        "trajectories_with_mixed_step_labels": 0,
    }
    query_to_has_positive_trajectory: Dict[Any, bool] = {}

    for record in tqdm(_iter_jsonl(input_path), desc="Building BCE data", unit="tree", dynamic_ncols=True):
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

        stats["first_layer_assistant_count"] += len(first_layer_assistant_ids)
        query_index = record.get("query_index")
        query_to_has_positive_trajectory.setdefault(query_index, False)


        # 遍历第一层的所有assistant node
        for first_assistant_id in first_layer_assistant_ids:
            candidate_leaf_ids = [
                str(node_id)
                for node_id in continuation_leaf_ids_by_node.get(first_assistant_id, [])
                if str(node_id) in node_map and str(node_id) in leaf_success_by_id
            ]
            if not candidate_leaf_ids:
                stats["skipped_no_leaf_under_first_layer_assistant"] += 1
                continue

            leaf_id = candidate_leaf_ids[0]
            node_chain = _chain_to_root(node_map, leaf_id)
            assistant_node_ids = [node_id for node_id in node_chain if node_map[node_id]["node_type"] == "assistant"]

            step_labels: List[int] = []
            step_mc_scores: List[float] = []
            for assistant_node_id in assistant_node_ids:
                continuation_leaf_ids = [str(x) for x in continuation_leaf_ids_by_node.get(assistant_node_id, [])]
                if not continuation_leaf_ids:
                    raise ValueError(f"Node {assistant_node_id} has no continuation leaves")
                success_count = sum(leaf_success_by_id.get(x, 0) for x in continuation_leaf_ids)
                total_count = len(continuation_leaf_ids)
                step_labels.append(1 if success_count > 0 else 0)
                step_mc_scores.append(success_count / total_count)

            messages = _reconstruct_messages(base_messages, node_map, node_chain)
            leaf_success = int(leaf_success_by_id.get(leaf_id, 0))
            has_positive_step = any(step_labels)
            has_negative_step = any(label == 0 for label in step_labels)

            rows.append(
                {
                    "data_source": record.get("data_source"),
                    "ability": "search",
                    "prompt": _json_dumps(_normalize_messages(messages)),
                    "tools": _json_dumps(_normalize_tools(DEFAULT_SEARCH_TOOLS)),
                    "step_labels": _json_dumps([int(label) for label in step_labels]),
                    "step_signed_labels": _json_dumps([1 if label == 1 else -1 for label in step_labels]),
                    "step_mc_scores": _json_dumps([float(score) for score in step_mc_scores]),
                    "step_node_ids": _json_dumps([str(node_id) for node_id in assistant_node_ids]),
                    "query_index": record.get("query_index"),
                    "tree_id": record.get("tree_id"),
                    "terminal_node_id": leaf_id,
                    "terminal_node_type": node_map[leaf_id].get("node_type"),
                    "terminal_success": leaf_success,
                    "question": record.get("question"),
                    "ground_truth": _json_dumps(_safe_native(ground_truth)),
                    "reward_model": _json_dumps({
                            "style": "bce_process",
                            "ground_truth": _safe_native(ground_truth),
                        }),
                    }
                )
            stats["sample_count"] += 1
            stats["positive_step_count"] += sum(step_labels)
            stats["negative_step_count"] += len(step_labels) - sum(step_labels)
            stats["trajectories_with_any_positive_step"] += int(has_positive_step)
            stats["trajectories_with_mixed_step_labels"] += int(has_positive_step and has_negative_step)
            query_to_has_positive_trajectory[query_index] = query_to_has_positive_trajectory[query_index] or has_positive_step

    stats["query_count"] = len(query_to_has_positive_trajectory)
    stats["queries_with_any_positive_trajectory"] = sum(
        int(has_positive_trajectory) for has_positive_trajectory in query_to_has_positive_trajectory.values()
    )

    return pd.DataFrame(rows), stats


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    for row_idx, value in enumerate(dataframe["prompt"].tolist()):
        if not isinstance(value, str):
            raise ValueError(f"row {row_idx} column prompt must be str, got {type(value).__name__}")
    for row_idx, value in enumerate(dataframe["tools"].tolist()):
        if not isinstance(value, str):
            raise ValueError(f"row {row_idx} column tools must be str, got {type(value).__name__}")
    for row_idx, value in enumerate(dataframe["step_labels"].tolist()):
        if not isinstance(value, str):
            raise ValueError(f"row {row_idx} column step_labels must be str, got {type(value).__name__}")


def _split_dataframe_by_query(
    dataframe: pd.DataFrame,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")

    if "query_index" not in dataframe.columns:
        indices = np.arange(len(dataframe))
        if len(indices) <= 1:
            return dataframe, dataframe.iloc[0:0].copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        validation_size = min(max(1, int(round(len(indices) * validation_ratio))), len(indices) - 1)
        validation_indices = set(indices[:validation_size].tolist())
        validation_df = dataframe.iloc[[idx for idx in range(len(dataframe)) if idx in validation_indices]].copy()
        train_df = dataframe.iloc[[idx for idx in range(len(dataframe)) if idx not in validation_indices]].copy()
        return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)

    unique_queries = dataframe["query_index"].drop_duplicates().tolist()
    if len(unique_queries) <= 1:
        return dataframe, dataframe.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_queries)
    validation_size = min(max(1, int(round(len(unique_queries) * validation_ratio))), len(unique_queries) - 1)
    validation_queries = set(unique_queries[:validation_size])

    validation_mask = dataframe["query_index"].isin(validation_queries)
    validation_df = dataframe[validation_mask].copy().reset_index(drop=True)
    train_df = dataframe[~validation_mask].copy().reset_index(drop=True)
    return train_df, validation_df


def _derive_split_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_train{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_validation{output_path.suffix}"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trajectory-level BCE process-label parquet from tree JSONL")
    parser.add_argument("--input", type=str, required=True, help="Compact tree JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--score_method", type=str, default="strict", choices=["strict", "subem"])
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe, stats = build_dataset(input_path=input_path, score_method=args.score_method)
    _validate_dataframe(dataframe)
    train_df, validation_df = _split_dataframe_by_query(
        dataframe,
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
    )
    train_output_path, validation_output_path = _derive_split_paths(output_path)
    train_df.to_parquet(train_output_path, index=False)
    validation_df.to_parquet(validation_output_path, index=False)

    total_steps = stats["positive_step_count"] + stats["negative_step_count"]
    summary = {
        **stats,
        "positive_step_rate": round(stats["positive_step_count"] / total_steps, 6) if total_steps else 0.0,
        "query_rate_with_any_positive_trajectory": (
            round(stats["queries_with_any_positive_trajectory"] / stats["query_count"], 6) if stats["query_count"] else 0.0
        ),
        "trajectory_rate_with_any_positive_step": (
            round(stats["trajectories_with_any_positive_step"] / stats["sample_count"], 6) if stats["sample_count"] else 0.0
        ),
        "trajectory_rate_with_mixed_step_labels": (
            round(stats["trajectories_with_mixed_step_labels"] / stats["sample_count"], 6) if stats["sample_count"] else 0.0
        ),
        "train_sample_count": int(len(train_df)),
        "validation_sample_count": int(len(validation_df)),
        "validation_ratio": float(args.validation_ratio),
        "seed": int(args.seed),
        "input": str(input_path),
        "train_output": str(train_output_path),
        "validation_output": str(validation_output_path),
        "metric_notes": {
            "tree_count": "树的数量，也就是 query 的数量。",
            "sample_count": "最终保留下来的训练样本数；当前等于第一层 assistant 节点数。",
            "first_layer_assistant_count": "所有 query 的第一层 assistant 子节点总数。",
            "positive_step_count": "所有样本里标签为 +1 的 step 总数。",
            "negative_step_count": "所有样本里标签为 -1/0 的 step 总数。",
            "positive_step_rate": "所有 step 中，正标签 step 的比例。",
            "query_count": "参与统计的 query 数量。",
            "queries_with_any_positive_trajectory": "至少存在一条轨迹包含 +1 step 的 query 数量。",
            "query_rate_with_any_positive_trajectory": "各个 query 的多跳轨迹中，至少存在一条包含 +1 step 的比例。",
            "trajectories_with_any_positive_step": "至少包含一个 +1 step 的轨迹数量。",
            "trajectory_rate_with_any_positive_step": "所有轨迹中，至少包含一个 +1 step 的比例。",
            "trajectories_with_mixed_step_labels": "同时包含 +1 和 -1/0 step 的轨迹数量。",
            "trajectory_rate_with_mixed_step_labels": "所有轨迹中，同时包含不同标签（+1 和 -1/0）的比例。",
            "train_sample_count": "切分后的训练集样本数。",
            "validation_sample_count": "切分后的验证集样本数。",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
