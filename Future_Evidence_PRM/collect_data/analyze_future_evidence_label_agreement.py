from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


StepKey = Tuple[str, int]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def _extract_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(record.get("step_annotations"), list):
        return [step for step in record["step_annotations"] if isinstance(step, dict)]

    response_json = record.get("response_json")
    if isinstance(response_json, dict):
        step_assessments = ((response_json.get("step_assessments") or {}).get("steps") or {})
        step_labels = response_json.get("step_labels")
        step_future_implications = ((response_json.get("future_implications") or {}).get("steps") or {})
        step_explanations = ((response_json.get("explanations") or {}).get("steps") or {})
        if isinstance(step_labels, dict) and isinstance(step_explanations, dict):
            return [
                {
                    "assistant_message_index": int(key),
                    "step_assessment": str(step_assessments.get(key, "")),
                    "step_label": int(value),
                    "future_implication": step_future_implications.get(key, {}),
                    "explanation": str(step_explanations.get(key, "")),
                }
                for key, value in sorted(step_labels.items(), key=lambda item: int(item[0]))
            ]

    response = record.get("response")
    if isinstance(response, str) and response.strip():
        response_data = json.loads(response)
        if isinstance(response_data, dict):
            step_assessments = ((response_data.get("step_assessments") or {}).get("steps") or {})
            step_labels = response_data.get("step_labels")
            step_future_implications = ((response_data.get("future_implications") or {}).get("steps") or {})
            step_explanations = ((response_data.get("explanations") or {}).get("steps") or {})
            if isinstance(step_labels, dict) and isinstance(step_explanations, dict):
                return [
                    {
                        "assistant_message_index": int(key),
                        "step_assessment": str(step_assessments.get(key, "")),
                        "step_label": int(value),
                        "future_implication": step_future_implications.get(key, {}),
                        "explanation": str(step_explanations.get(key, "")),
                    }
                    for key, value in sorted(step_labels.items(), key=lambda item: int(item[0]))
                ]

    teacher_parsed_output = record.get("teacher_parsed_output")
    if isinstance(teacher_parsed_output, dict):
        return [teacher_parsed_output]

    if "assistant_message_index" in record and "step_label" in record:
        return [record]

    return []


def _load_step_map(path: Path) -> Dict[StepKey, Dict[str, Any]]:
    step_map: Dict[StepKey, Dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        trajectory_sample_id = str(record.get("trajectory_sample_id", "")).strip()
        if not trajectory_sample_id:
            continue

        for step in _extract_steps(record):
            if "assistant_message_index" not in step:
                continue
            key = (trajectory_sample_id, int(step["assistant_message_index"]))
            if key in step_map:
                raise ValueError(f"Duplicate step key found in {path}: {key}")
            step_map[key] = {
                "step_assessment": str(step.get("step_assessment", "")).strip(),
                "step_label": int(step.get("step_label", 0)),
                "future_implication": step.get("future_implication", {}),
                "explanation": str(step.get("explanation", "")).strip(),
            }
    return step_map


def _extract_mc_label(record: Dict[str, Any]) -> Dict[str, Any]:
    future_stats = record.get("future_stats")
    if isinstance(future_stats, dict):
        success_count = int(future_stats.get("success_count", 0) or 0)
        total_future_count = int(future_stats.get("total_future_count", 0) or 0)
        success_rate = float(future_stats.get("success_rate", 0.0) or 0.0)
        return {
            "mc_label": int(success_count > 0 or success_rate > 0.0),
            "success_count": success_count,
            "total_future_count": total_future_count,
            "success_rate": success_rate,
        }

    selected_future_trajectories = record.get("selected_future_trajectories")
    if isinstance(selected_future_trajectories, list):
        success_values = [int(item.get("success", 0) or 0) for item in selected_future_trajectories if isinstance(item, dict)]
        success_count = sum(success_values)
        total_future_count = len(success_values)
        success_rate = success_count / total_future_count if total_future_count else 0.0
        return {
            "mc_label": int(success_count > 0),
            "success_count": success_count,
            "total_future_count": total_future_count,
            "success_rate": success_rate,
        }

    raise ValueError("MC source record must contain `future_stats` or `selected_future_trajectories`")


def _load_mc_map(path: Path) -> Dict[StepKey, Dict[str, Any]]:
    mc_map: Dict[StepKey, Dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        trajectory_sample_id = str(record.get("trajectory_sample_id", "")).strip()
        if not trajectory_sample_id:
            continue
        if "assistant_message_index" not in record:
            continue
        key = (trajectory_sample_id, int(record["assistant_message_index"]))
        if key in mc_map:
            raise ValueError(f"Duplicate MC step key found in {path}: {key}")
        mc_map[key] = _extract_mc_label(record)
    return mc_map


def _counter_to_dict(counter: Counter[Any]) -> Dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _confusion_to_dict(confusion: Dict[str, Counter[str]]) -> Dict[str, Dict[str, int]]:
    return {
        label_a: dict(sorted(counter_b.items(), key=lambda item: item[0]))
        for label_a, counter_b in sorted(confusion.items(), key=lambda item: item[0])
    }


def _agreement_rate(labels_a: List[Any], labels_b: List[Any]) -> float:
    if not labels_a:
        return 0.0
    matches = sum(label_a == label_b for label_a, label_b in zip(labels_a, labels_b))
    return round(matches / len(labels_a), 6)


def _cohen_kappa_from_lists(labels_a: List[Any], labels_b: List[Any]) -> float:
    if not labels_a:
        return 0.0

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    observed = sum(label_a == label_b for label_a, label_b in zip(labels_a, labels_b)) / len(labels_a)
    expected = 0.0
    for label in set(counts_a) | set(counts_b):
        expected += (counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b))

    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1.0 - expected), 6)


def _binary_metrics(predictions: List[int], targets: List[int]) -> Dict[str, Any]:
    if not predictions:
        return {
            "count": 0,
            "prediction_positive_rate": 0.0,
            "target_positive_rate": 0.0,
            "agreement": 0.0,
            "cohen_kappa": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }

    tp = sum(pred == 1 and target == 1 for pred, target in zip(predictions, targets))
    fp = sum(pred == 1 and target == 0 for pred, target in zip(predictions, targets))
    tn = sum(pred == 0 and target == 0 for pred, target in zip(predictions, targets))
    fn = sum(pred == 0 and target == 1 for pred, target in zip(predictions, targets))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    agreement = (tp + tn) / len(predictions)

    return {
        "count": len(predictions),
        "prediction_positive_rate": round(sum(predictions) / len(predictions), 6),
        "target_positive_rate": round(sum(targets) / len(targets), 6),
        "agreement": round(agreement, 6),
        "cohen_kappa": _cohen_kappa_from_lists(predictions, targets),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _model_vs_mc(step_map: Dict[StepKey, Dict[str, Any]], mc_map: Dict[StepKey, Dict[str, Any]]) -> Dict[str, Any]:
    overlap_keys = sorted(set(step_map) & set(mc_map))
    mc_labels = [int(mc_map[key]["mc_label"]) for key in overlap_keys]

    label_is_positive = [int(step_map[key]["step_label"] == 1) for key in overlap_keys]
    label_is_non_negative = [int(step_map[key]["step_label"] >= 0) for key in overlap_keys]

    return {
        "mc_overlap_step_count": len(overlap_keys),
        "only_in_model_step_count": len(set(step_map) - set(mc_map)),
        "only_in_mc_step_count": len(set(mc_map) - set(step_map)),
        "mc_label_distribution": _counter_to_dict(Counter(mc_labels)),
        "label_is_positive_vs_mc": _binary_metrics(label_is_positive, mc_labels),
        "label_is_non_negative_vs_mc": _binary_metrics(label_is_non_negative, mc_labels),
    }


def analyze(file_a: Path, file_b: Path, mc_file: Path | None) -> Dict[str, Any]:
    step_map_a = _load_step_map(file_a)
    step_map_b = _load_step_map(file_b)

    keys_a = set(step_map_a)
    keys_b = set(step_map_b)
    overlap_keys = sorted(keys_a & keys_b)
    only_a_keys = keys_a - keys_b
    only_b_keys = keys_b - keys_a

    overlap_records_a = [step_map_a[key] for key in overlap_keys]
    overlap_records_b = [step_map_b[key] for key in overlap_keys]

    labels_a = [record["step_label"] for record in overlap_records_a]
    labels_b = [record["step_label"] for record in overlap_records_b]

    label_confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for label_a, label_b in zip(labels_a, labels_b):
        label_confusion[str(label_a)][str(label_b)] += 1

    summary: Dict[str, Any] = {
        "file_a": str(file_a),
        "file_b": str(file_b),
        "step_count_a": len(step_map_a),
        "step_count_b": len(step_map_b),
        "overlap_step_count": len(overlap_keys),
        "only_in_a_step_count": len(only_a_keys),
        "only_in_b_step_count": len(only_b_keys),
        "label_distribution_a": _counter_to_dict(Counter(record["step_label"] for record in step_map_a.values())),
        "label_distribution_b": _counter_to_dict(Counter(record["step_label"] for record in step_map_b.values())),
        "label_agreement": _agreement_rate(labels_a, labels_b),
        "label_cohen_kappa": _cohen_kappa_from_lists(labels_a, labels_b),
        "label_confusion_matrix": _confusion_to_dict(label_confusion),
    }

    if mc_file is not None:
        mc_map = _load_mc_map(mc_file)
        summary["mc_file"] = str(mc_file)
        summary["file_a_vs_mc"] = _model_vs_mc(step_map_a, mc_map)
        summary["file_b_vs_mc"] = _model_vs_mc(step_map_b, mc_map)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze label agreement between two future-evidence annotation JSONL files")
    parser.add_argument("--file_a", type=str, required=True, help="First JSONL file")
    parser.add_argument("--file_b", type=str, required=True, help="Second JSONL file")
    parser.add_argument(
        "--mc_file",
        type=str,
        default="",
        help="Optional teacher-input JSONL with `future_stats` or `selected_future_trajectories` for MC consistency analysis",
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    mc_file = Path(args.mc_file) if args.mc_file else None
    summary = analyze(Path(args.file_a), Path(args.file_b), mc_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
