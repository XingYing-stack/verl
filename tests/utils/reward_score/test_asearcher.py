from verl.utils.reward_score import default_compute_score


def _wrap_answer(answer: str) -> str:
    return f"<answer> <answer>{answer}<answer></answer>"


def test_asearcher_scores_augmented_answer_match():
    aug_answers = "Oscar Hammerstein, O. G. Clendenning Hammerstein II"
    solution_str = _wrap_answer("O. G. Clendenning Hammerstein II")

    score = default_compute_score(
        "asearcher",
        solution_str,
        "Oscar Hammerstein II",
        extra_info={"aug_answers": aug_answers},
    )

    assert score == 1.0


def test_asearcher_scores_incorrect_answer_zero():
    aug_answers = "Oscar Hammerstein, O. G. Clendenning Hammerstein II"
    solution_str = _wrap_answer("Composer Name")

    score = default_compute_score(
        "asearcher",
        solution_str,
        "Oscar Hammerstein II",
        extra_info={"aug_answers": aug_answers},
    )

    assert score == 0.0


def test_asearcher_accepts_augmented_list_input():
    aug_answers = ["Jodie", "Jodie Comer"]
    solution_str = _wrap_answer("Jodie Comer")

    score = default_compute_score(
        "asearcher",
        solution_str,
        "Jodie Comer",
        extra_info={"aug_answers": aug_answers},
    )

    assert score == 1.0

if __name__ == "__main__":
    test_asearcher_scores_augmented_answer_match()
    test_asearcher_scores_incorrect_answer_zero()
    test_asearcher_accepts_augmented_list_input()