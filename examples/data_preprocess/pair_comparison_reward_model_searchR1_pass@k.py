"""
Build pair-comparison reward-model data from Search-R1 pass@k rollouts.

Input:
- `--metadata_path`: JSONL produced by `Future_Evidence_PRM/eval/run_pass@k.py`

We group rollouts by `query_index` (typically k=8 per question), pick a partner rollout per sample
(`random` or `nearest` by `tool_call_similarity`), then label the better trajectory as `positive`
and the worse as `negative` using ground-truth exact match (Search-R1 EM).

Output:
- Parquet files with `prompt` (system+user) and `reward_model.ground_truth` for pair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import pandas as pd
from transformers import AutoTokenizer

from Future_Evidence_PRM.utils import tool_call_similarity
from verl.utils.reward_score.search_r1_like_qa_em import compute_score as qa_em_score


def split_trajectory(ori_traj: str) -> str:
    """Split a rendered ChatML trajectory string into a JSON list of steps."""
    traj_list = [chunk.strip() for chunk in (ori_traj or "").split("<|im_start|>assistant\n") if chunk.strip()]
    traj_list = [
        chunk.replace("<|im_start|>", "").replace("<|im_end|>", "")
        for chunk in traj_list
        if "<|im_start|>system" not in chunk
    ]
    steps = [{"step_id": idx, "content": content} for idx, content in enumerate(traj_list)]
    return json.dumps(steps, indent=2, ensure_ascii=False)



# todo: 可能需要添加thought, 去掉是不太对的
SYSTEM_PROMPT_TEMPLATE = """# General Objective

You are an experienced and impartial judge. I will provide a *question* and two *trajectories* (trajectory_1 and trajectory_2). Each trajectory is a JSON object whose keys are stringified integers ("0","1",...) and whose values are step texts.  
A step text may contain tool calls (`<tool_call>...</tool_call>`), tool responses (`<tool_response>...</tool_response>`), and a answer tag `<answer>...</answer>`.


Your Goals:
1) For **every step** in each trajectory, give `step_evaluations`, i.e., deciding whether that step helped move toward the correct final answer (`helps`: true/false) and provide a short `reason`.  
2) Decide which trajectory's **final answer** is correct (trajectory_1, trajectory_2, both, or neither) in `final_comparison` and explain why.


# Guidelines

- The trajectory with the correct final answer does **not necessarily** have all correct intermediate steps.  
- Likewise, a trajectory with an incorrect final answer may still contain helpful intermediate steps.  
- For each step, explain *why* it is or is not helpful **before** giving the boolean label (`reason` before `helps`).  
- The number of steps you evaluate must exactly match the number of steps in each input trajectory, including the final answer step.
- Your judgments must focus on *factual correctness* and *logical contribution*, not on style or phrasing.

# Format Guidelines

## Input Format

You will receive a query and two trajectories of the format { "0": "<step text>", "1": "<step text>", "...": "<step text>" }.

## Output Format

Your output **must** be in strict JSON format within 
```json\nyour_answer\n``` 
tags.

Example output:

```json
{
  "meta": {
    "trajectory_1_expected_steps": <int>,      // N1
    "trajectory_2_expected_steps": <int>,      // N2
  },
  "step_evaluations": {
    "trajectory_1": [
      { "step_id": 0, "reason": "<string>", "helps": true|false },
      { "step_id": 1, "reason": "<string>", "helps": true|false },
      ...
      { "step_id": N1-1, "reason": "<string>", "helps": true|false } 
    ],
    "trajectory_2": [
      { "step_id": 0, "reason": "<string>", "helps": true|false },
      ...
      { "step_id": N2-1, "reason": "<string>", "helps": true|false }
    ]
  },
  "final_comparison": {
    "reason": "<why the final answer of one/both/neither is correct>",
    "trajectory_1_final_correct": true|false,
    "trajectory_2_final_correct": true|false
  }
}
```

###  Step Evaluations Parsing Rules

- Let N1 = the number of keys in trajectory_1 (after sorting keys as integers: 0,1,2,...).
- Let N2 = the number of keys in trajectory_2 (same sorting rule).
- For trajectory_1, you must output exactly N1 step evaluations with step_id equal to those integer indices (0..N1-1).
- For trajectory_2, do the same: exactly N2 integer steps.
- If any step contains an <answer> tag, treat that step as part of the sequence (not a separate step).
- Do not fabricate or drop steps. The total number of step_evaluations per trajectory must be:
    - N1 for trajectory_1
    - N2 for trajectory_2
"""

USER_PROMPT_TEMPLATE = """# Task Input

Below is a question and two trajectories attempting to solve it.
Each trajectory contains multiple steps, including tool calls, tool responses, and a final answer.

Please carefully read both trajectories in full before making any judgment.

---

**Question:**
{question}

---

**trajectory_1:**
```json
{trajectory_1}
```

---

**trajectory_2:**
```json
{trajectory_2}
```

---

Now, based on the two trajectories and the question, produce your JSON answer following the required output format.
"""

@dataclass(frozen=True)
class RolloutSample:
    query_index: int
    sample_index: int
    question: str
    data_source: str
    answer_text: str
    ground_truth: Optional[dict[str, Any]]
    em_score: float
    messages: list[dict[str, Any]]
    output: str


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _render_output_from_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(messages[2:], tokenize=False)
    if "<|im_start|>assistant" in rendered:
        rendered = rendered[rendered.index("<|im_start|>assistant") :]
    return rendered


def _choose_partner_random(n: int, i: int, rng: random.Random) -> int:
    j = rng.randrange(0, n - 1)
    return j if j < i else j + 1


def _choose_partner_nearest_answer_aware(
    outputs: list[str],
    answers: list[str],
    em_scores: list[float],
    i: int,
    rng: random.Random,
) -> tuple[int, bool, bool]:
    """Prefer a near-miss partner (answer differs and EM differs), then maximize tool_call_similarity.

    Returns:
        (partner_index, used_different_answer_pool, used_near_miss_pool)
    """

    def _norm_answer(a: str) -> str:
        return (a or "").strip().lower()

    target_answer = _norm_answer(answers[i])
    target_score = float(em_scores[i])

    def _best_from(candidates: list[int]) -> Optional[int]:
        best_sim = -1.0
        best_js: list[int] = []
        for j in candidates:
            if outputs[i] == outputs[j]:
                continue
            sim = float(tool_call_similarity(outputs[i], outputs[j]))
            if sim > best_sim:
                best_sim = sim
                best_js = [j]
            elif sim == best_sim:
                best_js.append(j)
        if best_js:
            return rng.choice(best_js)
        return None

    # Build candidate pools.
    near_miss_candidates: list[int] = []
    diff_answer_candidates: list[int] = []
    any_candidates: list[int] = []
    for j in range(len(outputs)):
        if i == j:
            continue
        any_candidates.append(j)
        if _norm_answer(answers[j]) != target_answer:
            diff_answer_candidates.append(j)
            if float(em_scores[j]) != target_score:
                near_miss_candidates.append(j)

    # Prefer near-miss: similar tool usage, different answer, different correctness.
    picked = _best_from(near_miss_candidates)
    if picked is not None:
        return picked, True, True

    # Next, allow different answer even if correctness matches.
    picked = _best_from(diff_answer_candidates)
    if picked is not None:
        return picked, True, False

    picked = _best_from(any_candidates)
    if picked is not None:
        return picked, False, False

    # All outputs identical (or only self): fall back to any other sample.
    return _choose_partner_random(len(outputs), i, rng), False, False


def _pair_to_row(
    *,
    idx: int,
    data_source: str,
    system_message: str,
    question: str,
    t1: RolloutSample,
    t2: RolloutSample,
    rng: random.Random,
) -> dict[str, Any]:
    if (t1.em_score, -t1.sample_index) >= (t2.em_score, -t2.sample_index):
        tag1, tag2 = "positive", "negative"
    else:
        tag1, tag2 = "negative", "positive"

    traj1 = split_trajectory(t1.output)
    traj2 = split_trajectory(t2.output)

    # Randomize order to reduce positional bias (but keep tags aligned).
    if rng.random() < 0.5:
        traj1, traj2 = traj2, traj1
        tag1, tag2 = tag2, tag1
        t1, t2 = t2, t1

    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, trajectory_1=traj1, trajectory_2=traj2)
    solution = {"trajectory_1": tag1, "trajectory_2": tag2}

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        "ability": "pair_comparison",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": "train",
            "index": idx,
            "question": question,
            "trajectory_1": traj1,
            "trajectory_2": traj2,
            "need_tools_kwargs": False,
            "dataset_name": t1.data_source,
            "query_index": t1.query_index,
            "sample_index_1": t1.sample_index,
            "sample_index_2": t2.sample_index,
            "em_score_1": float(t1.em_score),
            "em_score_2": float(t2.em_score),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/fanshengda/verl/input_data/PRM_from_ORM")
    parser.add_argument(
        "--metadata_path",
        default="/nfsdata/fanshengda/verl/Future_Evidence_PRM/eval/output/pass_at_k/pass_at_k_rollouts_test_0_7405_k8_20251219_114335.jsonl",
        help="JSONL produced by Future_Evidence_PRM/eval/run_pass@k.py"
    )
    parser.add_argument("--k", type=int, default=8, help="Expected rollouts per query (used for sanity stats only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_select_mode", choices=["random", "nearest"], default="nearest")
    parser.add_argument("--tokenizer_path", default="/nfsdata/fanshengda/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_filename", default=None)
    parser.add_argument("--val_filename", default=None)
    args = parser.parse_args()

    data_source = "searchR1_near_miss_prm"

    rng = random.Random(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and filter rollouts
    raw_records: list[dict[str, Any]] = []
    for rec in _iter_jsonl(args.metadata_path):
        if "error" in rec:
            continue
        if "query_index" not in rec:
            continue
        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        raw_records.append(rec)
    if not raw_records:
        raise ValueError(f"No valid records found in {args.metadata_path}")

    # Render outputs and group by query_index
    grouped: dict[int, list[RolloutSample]] = defaultdict(list)
    for rec in raw_records:
        qidx = int(rec["query_index"])
        sidx = int(rec.get("sample_index", -1))
        question = str(rec.get("question") or "")
        ds = str(rec.get("data_source") or "unknown")
        answer_text = str(rec.get("answer_text") or "")
        ground_truth = rec.get("ground_truth") if isinstance(rec.get("ground_truth"), dict) else None
        messages = rec.get("messages") or []
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        output = _render_output_from_messages(tokenizer, messages)
        em_score = 0.0
        if ground_truth is not None and answer_text.strip():
            try:
                em_score = float(qa_em_score(solution_str=f"<answer>{answer_text}</answer>", ground_truth=ground_truth))
            except Exception:
                em_score = 0.0
        grouped[qidx].append(
            RolloutSample(
                query_index=qidx,
                sample_index=sidx,
                question=question,
                data_source=ds,
                answer_text=answer_text,
                ground_truth=ground_truth,
                em_score=em_score,
                messages=messages,
                output=output,
            )
        )

    # Build pair-comparison rows: one pair per sample (ensures each sample has a partner).
    system_message = deepcopy(SYSTEM_PROMPT_TEMPLATE)
    rows: list[dict[str, Any]] = []
    skipped_groups = 0
    nearest_fallbacks = 0
    near_miss_selections = 0
    diff_answer_selections = 0
    for _, samples in sorted(grouped.items(), key=lambda kv: kv[0]):
        if len(samples) < 2:
            skipped_groups += 1
            continue
        outputs = [s.output for s in samples]
        answers = [s.answer_text for s in samples]
        em_scores = [s.em_score for s in samples]
        for i, s in enumerate(samples):
            if args.pair_select_mode == "nearest":
                try:
                    j, used_diff_answer, used_near_miss = _choose_partner_nearest_answer_aware(
                        outputs, answers, em_scores, i, rng
                    )
                    if used_diff_answer:
                        diff_answer_selections += 1
                    if used_near_miss:
                        near_miss_selections += 1
                except Exception:
                    nearest_fallbacks += 1
                    j = _choose_partner_random(len(samples), i, rng)
            else:
                j = _choose_partner_random(len(samples), i, rng)

            partner = samples[j]
            row = _pair_to_row(
                idx=len(rows),
                data_source=data_source,
                system_message=system_message,
                question=s.question,
                t1=s,
                t2=partner,
                rng=rng,
            )
            rows.append(row)

    if not rows:
        raise ValueError("No pairs generated (all groups had <2 valid rollouts).")

    # Extract dataset_name to a top-level helper column for grouping
    full_dataset = pd.DataFrame(rows)
    full_dataset["dataset_name"] = full_dataset["extra_info"].apply(lambda x: x.get("dataset_name"))

    # Determine validation indices: per dataset_name, take the first 10% by original order
    val_indices = []
    for _, grp_idx in full_dataset.groupby("dataset_name").groups.items():
        grp_df = full_dataset.loc[list(grp_idx)]
        # sort by the original enumeration order stored in extra_info['index'] to ensure stability
        order = grp_df["extra_info"].apply(lambda x: x.get("index", 0))
        grp_df = grp_df.loc[order.sort_values().index]
        n = len(grp_df)
        k = int(n * 0.1)
        if k > 0:
            val_indices.extend(list(grp_df.index[:k]))

    # Assign split column
    full_dataset["split"] = "train"
    if val_indices:
        full_dataset.loc[val_indices, "split"] = "validation"

    # Also mirror split into nested extra_info['split'] for downstream compatibility
    full_dataset["extra_info"] = full_dataset.apply(lambda r: {**r["extra_info"], "split": r["split"]}, axis=1)

    # Create split datasets
    train_dataset = full_dataset[full_dataset["split"] == "train"].drop(columns=["dataset_name", "split"])
    validation_dataset = full_dataset[full_dataset["split"] == "validation"].drop(columns=["dataset_name", "split"])

    # Save train/validation datasets
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    base = os.path.basename(args.metadata_path).rsplit(".", 1)[0]
    train_name = args.train_filename or f"train_{base}.parquet"
    val_name = args.val_filename or f"validation_{base}.parquet"
    train_path = os.path.join(local_dir, train_name)
    val_path = os.path.join(local_dir, val_name)
    train_dataset.to_parquet(train_path)
    validation_dataset.to_parquet(val_path)

    print("metadata_path:", args.metadata_path)
    print("groups_total:", len(grouped), "groups_skipped(<2):", skipped_groups, "expected_k:", int(args.k))
    if args.pair_select_mode == "nearest":
        print("nearest_fallbacks:", nearest_fallbacks)
        print("diff_answer_selections(answer!=):", diff_answer_selections, "/", len(full_dataset))
        print("near_miss_selections(answer!= & em!=):", near_miss_selections, "/", len(full_dataset))
    print("pairs_total:", len(full_dataset), "train:", len(train_dataset), "validation:", len(validation_dataset))
    print("train_path:", train_path)
    print("validation_path:", val_path)
