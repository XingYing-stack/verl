from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable




def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    # First, try to extract JSON from ```json ... ``` markdown code block
    json_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
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

    # Fallback: find the last JSON object in the text
    # Use rfind to locate the last '{' to handle reasoning text before JSON
    end = text.rfind("}")
    if end < 0:
        raise ValueError("LLM output is not JSON")

    # Find the matching '{' for this '}'
    # We need to find the correct opening brace by counting braces
    brace_count = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == '}':
            brace_count += 1
        elif text[i] == '{':
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



def _normalize_judge_output(
    raw: dict[str, Any],
    *,
    assistant_indices: list[int],
) -> tuple[dict[str, int], int, dict[str, Any]]:
    if "step_labels" not in raw or "final_label" not in raw:
        raise ValueError("missing step_labels/final_label in judge output")
    step_labels_raw = raw.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError("step_labels must be an object")

    #
    step_labels_raw = {str(k): v for k, v in step_labels_raw.items()}

    expected_keys = [str(i) for i in assistant_indices]
    step_labels: dict[str, int] = {}
    for k in expected_keys:
        if k not in step_labels_raw:
            raise ValueError(f"missing step_labels[{k}]")
        step_labels[k] = _coerce_int_label(step_labels_raw[k])

    final_label = _coerce_int_label(raw.get("final_label"))

    explanations = raw.get("explanations") or {}
    if not isinstance(explanations, dict):
        raise ValueError("explanations must be an object")
    steps_expl = explanations.get("steps") or {}
    if not isinstance(steps_expl, dict):
        raise ValueError("explanations.steps must be an object")
    steps_expl = {str(k): v for k, v in steps_expl.items()}
    for k in expected_keys:
        if k not in steps_expl:
            raise ValueError(f"missing explanations.steps[{k}]")
        if not isinstance(steps_expl[k], str):
            steps_expl[k] = str(steps_expl[k])
    final_expl = explanations.get("final")
    if not isinstance(final_expl, str):
        final_expl = "" if final_expl is None else str(final_expl)
    explanations_out = {"steps": steps_expl, "final": final_expl}

    return step_labels, final_label, explanations_out


def calculate_PRM_score(predict_step_labels: dict[str, int], step_labels: dict[str, int]) -> float:
    assert isinstance(predict_step_labels, dict)
    assert isinstance(step_labels, dict)

    if predict_step_labels.keys() != step_labels.keys():
        return -1.0

    correct_num = 0
    for k in predict_step_labels.keys():
        if predict_step_labels[k] == step_labels[k]:
            correct_num += 1
    return correct_num / len(predict_step_labels)





def compute_score(solution_str: str, ground_truth: str, *, extra_info: dict | None = None, **_) -> float:
    assert 'assistant_indices' in extra_info
    assert 'reward_type' in extra_info

    assistant_indices = extra_info['assistant_indices']


    result = {'score': -1 , "ORM_score": -1, "PRM_score": -1, "format": -1}
    try:
        raw = _extract_json_object(solution_str)
        predict_step_labels, predict_final_label, explanations = _normalize_judge_output(
            raw, assistant_indices=assistant_indices
        )
    except Exception as e:
        return result

    result['format'] = 1

    # ===================================
    print('extra_info:', extra_info)
    gt = int(extra_info["final_label"])
    if gt in (-1, 0):
        target = -1
    elif gt == 1:
        target = 1
    else:
        raise ValueError(f"unexpected final_label: {gt}")

    ORM_score = float(predict_final_label == target)
    # ===================================
    # 去掉value为None的子项
    step_labels = {str(k): int(v) for k, v in extra_info['step_labels'].items() if v is not None}

    assert step_labels.keys() == {str(i) for i in assistant_indices}

    #
    PRM_score = calculate_PRM_score(predict_step_labels,  step_labels)


    result['ORM_score'] = ORM_score
    result['PRM_score'] = PRM_score

    reward_type = extra_info['reward_type']

    if reward_type == 'outcome':
        result['score'] = ORM_score

    elif reward_type == "process":
        result['score'] = PRM_score
    else:
        raise ValueError(f"reward_type must be 'outcome' or 'process'")

    return result
