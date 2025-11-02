import json

import pytest

from verl.utils.reward_score import default_compute_score, pair_comparison


def _build_solution(step_count: int, *, final_t1: bool, final_t2: bool, helps_t1: bool = True, helps_t2: bool = False) -> str:
    step_evaluations = {
        "trajectory_1": [
            {"step_id": idx, "reason": f"t1 step {idx}", "helps": helps_t1}
            for idx in range(step_count)
        ],
        "trajectory_2": [
            {"step_id": idx, "reason": f"t2 step {idx}", "helps": helps_t2}
            for idx in range(step_count)
        ],
    }
    final_comparison = {
        "reason": "analysis",
        "trajectory_1_final_correct": final_t1,
        "trajectory_2_final_correct": final_t2,
    }
    payload = {"step_evaluations": step_evaluations, "final_comparison": final_comparison}
    return f"judgement\n```json\n{json.dumps(payload)}\n```"


def _build_extra_info(step_count: int) -> dict[str, str]:
    trajectory = {str(idx): f"content {idx}" for idx in range(step_count)}
    traj_json = json.dumps(trajectory)
    return {"trajectory_1": traj_json, "trajectory_2": traj_json}


def test_pair_comparison_full_credit():
    solution = _build_solution(2, final_t1=True, final_t2=False)
    extra_info = _build_extra_info(2)
    ground_truth = {"trajectory_1": True, "trajectory_2": False}

    res = pair_comparison.compute_score(solution, ground_truth, extra_info=extra_info)
    assert res["score"] == pytest.approx(1.0)

    default_score = default_compute_score(
        "asearcher_near_miss_prm", solution, ground_truth, extra_info=extra_info
    )
    assert isinstance(default_score, dict)
    assert default_score["score"] == pytest.approx(1.0)


def test_pair_comparison_partial_credit():
    solution = _build_solution(3, final_t1=True, final_t2=True)
    extra_info = _build_extra_info(3)
    ground_truth = {"trajectory_1": True, "trajectory_2": False}

    res = pair_comparison.compute_score(solution, ground_truth, extra_info=extra_info)
    assert res["score"] == pytest.approx(0.5)


def test_pair_comparison_format_error_returns_negative_one():
    extra_info = _build_extra_info(1)
    ground_truth = {"trajectory_1": True, "trajectory_2": False}

    res = pair_comparison.compute_score("no json block", ground_truth, extra_info=extra_info)
    assert res["score"] == -1.0
    assert res["pc_err_stage"] == "extract_json"

    malformed = _build_solution(2, final_t1=True, final_t2=False)
    malformed = malformed.replace("\"step_evaluations\"", "\"step_eval\"")
    res_malformed = pair_comparison.compute_score(malformed, ground_truth, extra_info=extra_info)
    assert res_malformed["score"] == -1.0
    assert res_malformed["pc_err_stage"] != ""

if __name__ == "__main__":
    test_pair_comparison_full_credit()
    test_pair_comparison_partial_credit()
    test_pair_comparison_format_error_returns_negative_one()
