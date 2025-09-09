"""
GAIA scoring aligned with the official scorer.py:
https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py

Steps:
- Extract model_answer from <answer>...</answer> in solution_str.
- Apply question_scorer(model_answer, ground_truth) with the same rules:
  * If ground_truth is numeric -> compare floats (model side normalized by removing $, %, ,).
  * Else if ground_truth contains ',' or ';' -> split list; compare element-wise (numeric vs. normalized string w/ remove_punct=False).
  * Else -> compare normalized strings (remove whitespace, lowercase, remove punctuation).
"""

import re
import string


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_number_str(number_str: str) -> float:
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        # follow upstream: return inf to ensure mismatch
        return float("inf")


def _split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
    # Remove all whitespace
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _question_scorer(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"

    # number ground truth
    if _is_float(ground_truth):
        normalized_answer = _normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # list ground truth
    elif any(ch in ground_truth for ch in [",", ";"]):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                comparisons.append(_normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    _normalize_str(ma_elem, remove_punct=False)
                    == _normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # string ground truth
    else:
        return _normalize_str(model_answer) == _normalize_str(ground_truth)


def compute_score(solution_str: str, ground_truth: str, **_) -> float:
    pred = extract_answer(solution_str)
    return 1.0 if _question_scorer(pred or "", str(ground_truth)) else 0.0