"""
Summarize compact tree-sampling JSONL outputs.

This script reports:
- tree-level coverage and size statistics
- node counts by stored depth and node type
- terminal leaf distributions and termination reasons
- leaf success rates using Search-R1-like EM / SubEM scoring

Example
    python -m Future_Evidence_PRM.collect_data.tree_sampling_stats \
        --input rollout_data/tree_sampling_compact/example.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm


def _to_native(obj: Any) -> Any:
    if isinstance(obj, (list, dict)):
        return obj
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        try:
            return obj.as_py()
        except Exception:
            pass
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        try:
            return obj.to_pylist()
        except Exception:
            pass
    if isinstance(obj, (str, bytes)):
        try:
            text = obj.decode("utf-8") if isinstance(obj, bytes) else obj
            return json.loads(text)
        except Exception:
            return obj
    return obj


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_check(prediction: str, golden_answers: List[str] | str) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def subem_check(prediction: str, golden_answers: List[str] | str) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def count_answer_tags(text: str) -> tuple[int, int]:
    return text.count("<answer>"), text.count("</answer>")


def compute_score(solution_str: str, ground_truth: Dict[str, Any], format_score: float = 0.0, score: float = 1.0) -> float:
    answer = extract_solution(solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    if answer is None:
        return 0.0
    if em_check(answer, ground_truth["target"]):
        if open_count > 10 or close_count > 10:
            return score / 4
        return score
    return format_score


def compute_score_subem(
    solution_str: str,
    ground_truth: Dict[str, Any],
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    if subem_check(answer, ground_truth["target"]):
        return score
    return format_score


def _safe_native(obj: Any) -> Any:
    obj = _to_native(obj)
    if isinstance(obj, dict):
        return {str(key): _safe_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_safe_native(item) for item in obj]
    if isinstance(obj, tuple):
        return [_safe_native(item) for item in obj]
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return _safe_native(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _safe_native(obj.item())
        except Exception:
            pass
    return obj


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
        elif isinstance(target, list):
            target = [str(item) for item in target]
        else:
            target = [str(target)]
        normalized = dict(ground_truth)
        normalized["target"] = target
        return normalized
    if isinstance(ground_truth, str):
        return {"target": [ground_truth]}
    if isinstance(ground_truth, list):
        return {"target": [str(item) for item in ground_truth]}
    return {"target": [str(ground_truth)]}


def _extract_leaf_solution(node: Dict[str, Any]) -> str:
    leaf_output = node.get("leaf_output")
    if isinstance(leaf_output, str):
        return leaf_output

    delta_messages = node.get("delta_messages", [])
    if isinstance(delta_messages, list):
        for msg in reversed(delta_messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


def _score_leaf(node: Dict[str, Any], ground_truth: Optional[Dict[str, Any]], score_method: str) -> Optional[float]:
    if ground_truth is None:
        return None
    solution_str = _extract_leaf_solution(node)
    if score_method == "subem":
        return float(compute_score_subem(solution_str=solution_str, ground_truth=ground_truth))
    return float(compute_score(solution_str=solution_str, ground_truth=ground_truth))


def _format_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _format_counter(counter: Counter[Any]) -> Dict[str, int]:
    items = sorted(counter.items(), key=lambda item: str(item[0]))
    return {str(key): int(value) for key, value in items}


def _format_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _iter_jsonl(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if limit is not None and line_idx >= limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def analyze_file(path: Path, score_method: str, limit: Optional[int]) -> Dict[str, Any]:
    tree_count = 0
    trees_with_ground_truth = 0
    trees_with_any_success = 0
    trees_with_any_answer_leaf = 0

    nodes_per_tree: List[int] = []
    leaves_per_tree: List[int] = []
    successful_leaves_per_tree: List[int] = []
    root_continuations_per_tree: List[int] = []

    node_type_counter: Counter[str] = Counter()
    terminal_reason_counter: Counter[str] = Counter()
    leaf_node_type_counter: Counter[str] = Counter()

    depth_node_counter: Counter[int] = Counter()
    depth_terminal_counter: Counter[int] = Counter()
    depth_success_counter: Counter[int] = Counter()
    depth_answer_leaf_counter: Counter[int] = Counter()
    depth_node_type_counter: Dict[int, Counter[str]] = defaultdict(Counter)
    depth_termination_reason_counter: Dict[int, Counter[str]] = defaultdict(Counter)
    depth_child_counts: Dict[int, List[int]] = defaultdict(list)
    depth_continuation_counts: Dict[int, List[int]] = defaultdict(list)

    total_terminal_leaves = 0
    total_answer_leaves = 0
    total_successful_leaves = 0
    total_scored_leaves = 0

    leaf_reason_scored_counter: Counter[str] = Counter()
    leaf_reason_success_counter: Counter[str] = Counter()
    leaf_node_type_scored_counter: Counter[str] = Counter()
    leaf_node_type_success_counter: Counter[str] = Counter()

    progress = tqdm(_iter_jsonl(path, limit), desc="Analyzing", unit="tree", dynamic_ncols=True)
    for record in progress:
        tree_count += 1

        root_id = record["root_id"]
        nodes = record.get("nodes", [])
        child_ids_by_node = record.get("child_ids_by_node", {})
        continuation_leaf_ids_by_node = record.get("continuation_leaf_ids_by_node", {})
        terminal_node_ids = record.get("terminal_node_ids", [])
        ground_truth = _normalize_ground_truth(record.get("ground_truth"))

        if ground_truth is not None:
            trees_with_ground_truth += 1

        nodes_per_tree.append(len(nodes))
        leaves_per_tree.append(len(terminal_node_ids))
        root_continuations_per_tree.append(len(continuation_leaf_ids_by_node.get(root_id, [])))

        node_map = {node["node_id"]: node for node in nodes}

        for node in nodes:
            node_type = str(node.get("node_type"))
            depth = int(node.get("depth", -999))

            node_type_counter[node_type] += 1
            depth_node_counter[depth] += 1
            depth_node_type_counter[depth][node_type] += 1
            depth_child_counts[depth].append(len(child_ids_by_node.get(node["node_id"], [])))
            depth_continuation_counts[depth].append(len(continuation_leaf_ids_by_node.get(node["node_id"], [])))

        successful_leaves_this_tree = 0
        answer_leaves_this_tree = 0

        for leaf_id in terminal_node_ids:
            leaf = node_map[leaf_id]
            depth = int(leaf.get("depth", -999))
            node_type = str(leaf.get("node_type"))
            reason = str(leaf.get("termination_reason"))

            total_terminal_leaves += 1
            leaf_node_type_counter[node_type] += 1
            terminal_reason_counter[reason] += 1
            depth_terminal_counter[depth] += 1
            depth_termination_reason_counter[depth][reason] += 1

            answer_found = bool(leaf.get("answer_found"))
            if answer_found:
                total_answer_leaves += 1
                answer_leaves_this_tree += 1
                depth_answer_leaf_counter[depth] += 1

            score = _score_leaf(leaf, ground_truth, score_method)
            if score is None:
                continue

            total_scored_leaves += 1
            leaf_reason_scored_counter[reason] += 1
            leaf_node_type_scored_counter[node_type] += 1

            if score > 0:
                total_successful_leaves += 1
                successful_leaves_this_tree += 1
                depth_success_counter[depth] += 1
                leaf_reason_success_counter[reason] += 1
                leaf_node_type_success_counter[node_type] += 1

        if answer_leaves_this_tree > 0:
            trees_with_any_answer_leaf += 1
        if successful_leaves_this_tree > 0:
            trees_with_any_success += 1

        successful_leaves_per_tree.append(successful_leaves_this_tree)

    depth_stats: Dict[str, Any] = {}
    all_depths = sorted(depth_node_counter.keys())
    for depth in all_depths:
        terminal_count = depth_terminal_counter[depth]
        success_count = depth_success_counter[depth]
        answer_leaf_count = depth_answer_leaf_counter[depth]
        depth_stats[str(depth)] = {
            "node_count": depth_node_counter[depth],
            "node_type_count": _format_counter(depth_node_type_counter[depth]),
            "terminal_leaf_count": terminal_count,
            "answer_leaf_count": answer_leaf_count,
            "successful_leaf_count": success_count,
            "successful_leaf_rate_among_terminal": _format_rate(success_count, terminal_count),
            "successful_leaf_rate_among_answer_leaves": _format_rate(success_count, answer_leaf_count),
            "termination_reason_count": _format_counter(depth_termination_reason_counter[depth]),
            "avg_children_per_node": round(mean(depth_child_counts[depth]), 6) if depth_child_counts[depth] else 0.0,
            "avg_continuation_leaves_per_node": (
                round(mean(depth_continuation_counts[depth]), 6) if depth_continuation_counts[depth] else 0.0
            ),
        }

    reason_success_stats = {}
    for reason, total_count in terminal_reason_counter.items():
        scored_count = leaf_reason_scored_counter[reason]
        success_count = leaf_reason_success_counter[reason]
        reason_success_stats[str(reason)] = {
            "terminal_leaf_count": total_count,
            "scored_leaf_count": scored_count,
            "successful_leaf_count": success_count,
            "successful_leaf_rate_among_scored": _format_rate(success_count, scored_count),
        }

    leaf_node_type_success_stats = {}
    for node_type, total_count in leaf_node_type_counter.items():
        scored_count = leaf_node_type_scored_counter[node_type]
        success_count = leaf_node_type_success_counter[node_type]
        leaf_node_type_success_stats[str(node_type)] = {
            "terminal_leaf_count": total_count,
            "scored_leaf_count": scored_count,
            "successful_leaf_count": success_count,
            "successful_leaf_rate_among_scored": _format_rate(success_count, scored_count),
        }

    summary = {
        "input_path": str(path),
        "score_method": score_method,
        "tree_count": tree_count,
        "trees_with_ground_truth": trees_with_ground_truth,
        "trees_with_any_answer_leaf": trees_with_any_answer_leaf,
        "trees_with_any_successful_leaf": trees_with_any_success,
        "tree_rate_with_any_answer_leaf": _format_rate(trees_with_any_answer_leaf, tree_count),
        "tree_rate_with_any_successful_leaf": _format_rate(trees_with_any_success, tree_count),
        "nodes_per_tree": _format_stats(nodes_per_tree),
        "terminal_leaves_per_tree": _format_stats(leaves_per_tree),
        "successful_leaves_per_tree": _format_stats(successful_leaves_per_tree),
        "root_continuations_per_tree": _format_stats(root_continuations_per_tree),
        "total_terminal_leaves": total_terminal_leaves,
        "total_answer_leaves": total_answer_leaves,
        "total_scored_leaves": total_scored_leaves,
        "total_successful_leaves": total_successful_leaves,
        "leaf_success_rate_among_terminal": _format_rate(total_successful_leaves, total_terminal_leaves),
        "leaf_success_rate_among_answer_leaves": _format_rate(total_successful_leaves, total_answer_leaves),
        "leaf_success_rate_among_scored_leaves": _format_rate(total_successful_leaves, total_scored_leaves),
    }

    return {
        "summary": summary,
        "node_type_count": _format_counter(node_type_counter),
        "terminal_reason_count": _format_counter(terminal_reason_counter),
        "leaf_node_type_count": _format_counter(leaf_node_type_counter),
        "depth_stats": depth_stats,
        "termination_reason_success_stats": reason_success_stats,
        "leaf_node_type_success_stats": leaf_node_type_success_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compact tree-sampling JSONL outputs")
    parser.add_argument("--input", type=str, required=True, help="Path to compact tree JSONL file")
    parser.add_argument(
        "--score_method",
        type=str,
        default="strict",
        choices=["strict", "subem"],
        help="Leaf success metric based on Search-R1-like scoring",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of trees to analyze")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save summary JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    result = analyze_file(input_path, score_method=args.score_method, limit=args.limit)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
