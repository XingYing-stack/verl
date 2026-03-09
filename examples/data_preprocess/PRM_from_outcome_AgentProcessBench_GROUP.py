"""
Learning a PRM from outcome reward labels
"""

import argparse
import os
import re
import pandas as pd
import datasets
import pickle
from copy import deepcopy
import json
import glob
from typing import Any, Iterable
import random
import math
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from tqdm import tqdm
random.seed(66)

JUDGE_RUBRIC = """You are a strict but fair trajectory annotator for tool-use agents.

You will be given one or more complete trajectories consisting of system, user, assistant,
and tool messages, together with the tool definitions.

Your task is to label EACH assistant message (each assistant message constitutes
one Step) using the following scheme:

+1: Correct and effective.
    The step is factually correct given the information available at that time
    and clearly moves the task closer to successful completion by:
    (i) correctly invoking a tool or interpreting tool outputs, or
    (ii) introducing valid constraints, decisions, or information that
         reduces the remaining uncertainty of the task.

 0: Neutral or exploratory.
    The step is reasonable but has limited or unclear impact on task progress.
    This includes exploratory reasoning, redundant restatements, partial planning,
    or cases where the correctness is debatable given the available evidence.
    Tool calls that fail due to external reasons (e.g., timeout, 404), when the
    attempt itself is reasonable, are typically labeled 0.

-1: Incorrect or harmful.
    The step contains factual errors, misinterprets tool outputs, violates
    constraints, repeats failed actions without a meaningful change in strategy,
    fabricates tool results or evidence, or otherwise pushes the trajectory away
    from successful completion.

Important rules:

- Only assistant messages are labeled. User and tool messages serve only as evidence.
- Avoid hindsight bias: judge each step strictly based on the information available
  up to that point in the trajectory.
- Any step labeled -1 triggers a cumulative penalty: all subsequent assistant steps
  in the same workflow should also be labeled -1, unless one of the following holds:
    (i) the assistant explicitly acknowledges and corrects the earlier mistake, or
    (ii) the assistant produces a subsequent step that no longer depends on the
         incorrect assumption and effectively resumes progress toward the task.
- Repeating the same failed action without a meaningful change in parameters or
  strategy typically transitions from 0 to -1.
- If an incorrect statement does not affect any subsequent reasoning or actions
  and is not relied upon later, it may be labeled 0; otherwise, it should be labeled -1.
- Any violation of the policies or requirements specified in the system prompt results in a score of −1, except for certain output-formatting norms. The following behaviors are considered acceptable and do not incur penalties: providing a text response simultaneously with a tool call, not conducting reasoning before a tool call, failing to encapsulate reasoning content within `<think>...</think>` tags, responding to the user while executing a function call, or executing multiple parallel tool calls.
- A score of +1 is assigned if the entire conversation is initiated by the assistant and its first message is a greeting; this exemption applies only to the first message.
- Upon user request, if the assistant executes specific instructions, a score of +1 shall be awarded, notwithstanding any deviation from the overarching objective.

After labeling all assistant steps, also assign a label to:

FINAL_RESULT:
+1: The overall task is successfully completed.
-1: The task fails due to incorrect reasoning, tool misuse, or unresolved errors.

Output format:
You MUST first provide your reasoning process, analyzing each assistant step one by one.
Then, at the very end, output a JSON object wrapped in ```json ... ``` markdown code block as your judgement results."""


def _build_judge_input(
    *,
    item: dict[str, Any],
    assistant_indices: list[int],
) -> list[dict[str, Any]]:
    messages = item.get("messages")
    assert isinstance(messages, list), "item['messages'] must be a list."
    assert all(isinstance(i, int) for i in assistant_indices), "assistant_indices must be integers."
    assert all(0 <= i < len(messages) for i in assistant_indices), "assistant_indices contain out-of-range values."
    assert all(
        isinstance(messages[i], dict) and messages[i].get("role") == "assistant"
        for i in assistant_indices
    ), "assistant_indices must point to assistant messages."

    payload: dict[str, Any] = {
        "question": item.get("question"),
        "task_description": item.get("task_description"),
        "tools": item.get("tools"),
        "messages": list(enumerate(messages)),
        "assistant_message_indices": assistant_indices,
        "notes": {
            "step_definition": "Each Step == one message with role=='assistant'. Use the given indices.",
            "output_requirements": "Return JSON with step_labels, final_label, explanations.",
        },
    }

    used_judge_rubric = JUDGE_RUBRIC
    user_instructions = """Label every index in assistant_message_indices.

First, analyze each assistant message step by step.
After your reasoning, output the final JSON result wrapped in ```json ... ``` markdown code block.

JSON schema:
{
  "step_labels": {"<assistant_index>": -1|0|1, ...},
  "final_label": -1|1,
  "explanations": {
    "steps": {"<assistant_index>": "short reason for humans", ...},
    "final": "short reason for humans"
  }
}

Rules:
- step_labels MUST contain ALL assistant indices (as strings).
- explanations.steps MUST contain ALL assistant indices (as strings).
- Keep each explanation concise (<= 2 sentences).
"""

    return [
        {"role": "system", "content": used_judge_rubric},
        {"role": "user", "content": user_instructions + "\n\nTRAJECTORY_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def _build_group_judge_input(
    *,
    items: list[dict[str, Any]],
    assistant_indices_list: list[list[int]],
) -> list[dict[str, Any]]:
    assert items, "items must be non-empty"
    assert len(items) == len(assistant_indices_list), "assistant_indices_list length mismatch"

    question = items[0].get("question")
    payload: dict[str, Any] = {
        "question": question,
        "trajectories": [],
        "notes": {
            "trajectory_definition": "Each trajectory is independent; Step == one message with role=='assistant'.",
            "output_requirements": "Return per-trajectory step_labels, final_label, explanations.",
        },
    }

    for traj_id, (item, assistant_indices) in enumerate(zip(items, assistant_indices_list, strict=True)):
        messages = item.get("messages")
        assert isinstance(messages, list), "item['messages'] must be a list."
        payload["trajectories"].append(
            {
                "trajectory_id": str(traj_id),
                "task_description": item.get("task_description"),
                "tools": item.get("tools"),
                "messages": list(enumerate(messages)),
                "assistant_message_indices": assistant_indices,
            }
        )

    user_instructions = """You will be given multiple trajectories for the SAME question.

For EACH trajectory, label every index in its assistant_message_indices.

First, analyze trajectory by trajectory, step by step.
After your reasoning, output the final JSON result wrapped in ```json ... ``` markdown code block.

JSON schema:
{
  "trajectories": {
    "<trajectory_id>": {
      "step_labels": {"<assistant_index>": -1|0|1, ...},
      "final_label": -1|1,
      "explanations": {
        "steps": {"<assistant_index>": "short reason for humans", ...},
        "final": "short reason for humans"
      }
    },
    ...
  }
}

Rules:
- trajectories MUST contain ALL trajectory_ids (as strings) present in TRAJECTORIES_JSON.
- For each trajectory, step_labels MUST contain ALL assistant indices (as strings).
- For each trajectory, explanations.steps MUST contain ALL assistant indices (as strings).
- Keep each explanation concise (<= 2 sentences).
"""

    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": user_instructions + "\n\nTRAJECTORIES_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def _assistant_message_indices(messages: Any) -> list[int]:
    if not isinstance(messages, list):
        return []
    indices: list[int] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            indices.append(i)
    return indices


def _normalize_tool_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        raw = arguments.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def _trajectory_signature(example: dict[str, Any]) -> str:
    """Build a lightweight string signature for trajectory similarity."""
    messages = example.get("messages")
    if not isinstance(messages, list):
        return ""

    tool_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "")
            args = _normalize_tool_arguments(fn.get("arguments"))
            tool_parts.append(f"{name}({args})")

    final_assistant = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            final_assistant = str(msg.get("content") or "").strip()
            break

    # keep it short for fast similarity
    final_assistant = final_assistant[:400]
    tool_sig = "\n".join(tool_parts)
    return f"{tool_sig}\n{final_assistant}".strip()


def _select_group_indices(
    examples: list[dict[str, Any]],
    *,
    traj_num: int,
    strategy: str,
    k_groups: int,
) -> list[tuple[int, ...]]:
    """Return index tuples for selected trajectory groups within one question."""
    n = len(examples)
    if n == 0:
        return []
    if traj_num <= 0:
        raise ValueError(f"traj_num must be > 0, got {traj_num}")
    if k_groups <= 0:
        return []
    if n < traj_num:
        return []

    total = math.comb(n, traj_num)
    target_groups = min(k_groups, total)
    if target_groups * traj_num < n:
        raise ValueError(
            f"coverage-first requires K*traj_num >= num_trajectories, got K={k_groups}, traj_num={traj_num}, n={n}"
        )

    # Step 1: coverage-first groups (guarantee every trajectory appears at least once)
    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()

    if strategy == "random":
        order = list(range(n))
        random.shuffle(order)
        for start in range(0, n, traj_num):
            chunk = order[start : start + traj_num]
            if len(chunk) < traj_num:
                candidates = [i for i in range(n) if i not in chunk]
                chunk = chunk + random.sample(candidates, traj_num - len(chunk))
            group = tuple(sorted(chunk))
            if group not in selected_set:
                selected.append(group)
                selected_set.add(group)

    elif strategy == "near":
        signatures = [_trajectory_signature(ex) for ex in examples]
        pair_sims: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_sims[(i, j)] = SequenceMatcher(None, signatures[i], signatures[j]).ratio()

        def sim(i: int, j: int) -> float:
            return pair_sims[(i, j)] if i < j else pair_sims[(j, i)]

        uncovered = set(range(n))
        while uncovered:
            if len(uncovered) == 1:
                anchor = next(iter(uncovered))
            else:
                anchor = max(
                    uncovered,
                    key=lambda i: max(sim(i, j) for j in range(n) if j != i),
                )
            uncovered.remove(anchor)

            candidates = [j for j in range(n) if j != anchor]
            candidates.sort(key=lambda j: sim(anchor, j), reverse=True)

            picked: list[int] = []
            for j in candidates:
                if j in uncovered and len(picked) < traj_num - 1:
                    picked.append(j)
            for j in candidates:
                if j not in picked and len(picked) < traj_num - 1:
                    picked.append(j)

            group = tuple(sorted([anchor, *picked]))
            selected.append(group)
            selected_set.add(group)
            for idx in group:
                uncovered.discard(idx)

    else:
        raise ValueError(f"Unexpected strategy: {strategy}")

    # Step 2: fill remaining groups using the chosen strategy preference
    if strategy == "random":
        if k_groups >= total:
            return list(combinations(range(n), traj_num))

        max_attempts = max(200, target_groups * 50)
        attempts = 0
        while len(selected) < target_groups and attempts < max_attempts:
            group = tuple(sorted(random.sample(range(n), traj_num)))
            if group not in selected_set:
                selected.append(group)
                selected_set.add(group)
            attempts += 1
        if len(selected) < target_groups:
            remaining = [g for g in combinations(range(n), traj_num) if g not in selected_set]
            random.shuffle(remaining)
            for g in remaining[: (target_groups - len(selected))]:
                selected.append(g)
                selected_set.add(g)

    else:  # near
        # If combination count is huge, keep it simple to avoid O(C(n,t)) explosion.
        # AgentProcessBench has n=5, so this is always cheap.
        candidates = list(combinations(range(n), traj_num))

        def group_score(group: tuple[int, ...]) -> float:
            if len(group) <= 1:
                return 0.0
            denom = math.comb(len(group), 2)
            return sum(sim(i, j) for i, j in combinations(group, 2)) / denom

        candidates.sort(key=group_score, reverse=True)
        for g in candidates:
            if len(selected) >= target_groups:
                break
            if g not in selected_set:
                selected.append(g)
                selected_set.add(g)

    covered: set[int] = set()
    for g in selected:
        covered.update(g)
    assert covered == set(range(n)), f"coverage failed: covered={sorted(covered)} expected={list(range(n))}"
    return selected[:target_groups]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/verl/input_data/PRM_from_outcome")
    parser.add_argument("--metadata_path",
                        default="/workspace/AgentProcessBench_KDD-main/data")

    parser.add_argument("--label_form", default="outcome", choices=["process", "outcome"])
    parser.add_argument("--traj_num", type=int, default=2, help="一次输入的轨迹个数")
    parser.add_argument("--strategy", default='near', help="构建reward model输入的方法", choices=['near', 'random'])
    parser.add_argument("--K", type=int, default=5, help="每个question选多少个轨迹group")


    args = parser.parse_args()

    data_source = "AgentProcessBench_GROUP"

    base_path = f"{args.metadata_path}/AgentProcessBench"
    assert os.path.isdir(base_path), f"base_path does not exist: {base_path}"

    all_data = []
    jsonl_files = sorted(glob.glob(os.path.join(base_path, "**", "*.jsonl"), recursive=True))
    assert jsonl_files, f"No .jsonl files found under: {base_path}"

    # 递归读取所有 jsonl 文件
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    parsed = json.loads(line)
                    assert isinstance(parsed, dict), f"Each jsonl row must be an object. file={file_path}"
                    all_data.append(parsed)

    # df = pd.DataFrame(all_data)

    print(f"Loaded {len(all_data)} rows from jsonl files.")
    assert all_data, f"No valid rows loaded from jsonl files under: {base_path}"


    def process_group_fn(group: list[dict[str, Any]], split: str, idx: int, label_form: str) -> dict[str, Any]:
        assert split in {"train", "val"}, f"Unexpected split: {split}"
        assert group, "group must be non-empty"

        required_keys = {"messages", "final_label", "data_source", "total_index", "step_labels", "question"}
        assistant_indices_list: list[list[int]] = []
        for example in group:
            assert isinstance(example, dict), "Each example must be a dict."
            missing_keys = required_keys - set(example.keys())
            assert not missing_keys, f"Missing required keys in example: {sorted(missing_keys)}"
            assert isinstance(example["messages"], list), "example['messages'] must be a list."
            assert isinstance(example["step_labels"], (list, dict)), "example['step_labels'] must be list or dict."

            assistant_indices = _assistant_message_indices(example.get("messages"))
            assert assistant_indices, "Each example must contain at least one assistant message."
            assistant_indices_list.append(assistant_indices)

        messages = _build_group_judge_input(items=group, assistant_indices_list=assistant_indices_list)
        assert isinstance(messages, list) and len(messages) == 2, "_build_group_judge_input must return 2-message prompt."

        if label_form == "process":
            ground_truth = {str(i): ex["step_labels"] for i, ex in enumerate(group)}
        elif label_form == "outcome":
            ground_truth = {str(i): ex["final_label"] for i, ex in enumerate(group)}
        else:
            raise ValueError(f"Unexpected label_form: {label_form}")

        trajectories_info: list[dict[str, Any]] = []
        for traj_id, (example, assistant_indices) in enumerate(zip(group, assistant_indices_list, strict=True)):
            trajectories_info.append(
                {
                    "trajectory_id": str(traj_id),
                    "assistant_indices": assistant_indices,
                    "ori_data_source": example["data_source"],
                    "index": example["total_index"],
                    "final_label": example["final_label"],
                    "step_labels": example["step_labels"],
                }
            )

        data = {
            "data_source": data_source,
            "prompt": messages,
            "ability": f"{label_form}_reward_model",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "reward_type": label_form,
                "question": group[0]["question"],
                "strategy": args.strategy,
                "traj_num": args.traj_num,
                'final_label': {str(i): ex["final_label"] for i, ex in enumerate(group)},
                'step_labels': {str(i): ex["step_labels"] for i, ex in enumerate(group)},
                "k_groups_per_question": args.K,
                "trajectories": trajectories_info,
                "group_index": idx,
                "need_tools_kwargs": False,
            },
        }
        return data

    # group by question (prefer (data_source, query_index) if present), then build K groups per question
    question_to_examples: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in all_data:
        if "query_index" in row:
            key = (row.get("data_source"), row.get("query_index"))
        else:
            key = str(row.get("question") or "").strip()
        if key:
            question_to_examples[key].append(row)

    questions = list(question_to_examples.keys())
    random.shuffle(questions)
    split_q = int(len(questions) * 0.8)

    train_questions: list[Any] = questions[:split_q]
    val_questions: list[Any] = questions[split_q:]
    assert set(train_questions).isdisjoint(val_questions), "Train/val question split overlap detected."

    def build_groups_for_questions(question_list: list[Any]) -> tuple[list[list[dict[str, Any]]], int]:
        groups: list[list[dict[str, Any]]] = []
        skipped = 0
        for q in tqdm(question_list):
            examples = question_to_examples[q]
            group_indices = _select_group_indices(
                examples,
                traj_num=args.traj_num,
                strategy=args.strategy,
                k_groups=args.K,
            )
            if not group_indices:
                skipped += 1
                continue
            for g in group_indices:
                groups.append([examples[i] for i in g])
        return groups, skipped

    train_groups, skipped_train = build_groups_for_questions(train_questions)
    val_groups, skipped_val = build_groups_for_questions(val_questions)
    print(f"Built {len(train_groups)} train groups, skipped {skipped_train} train questions (not enough trajectories).")
    print(f"Built {len(val_groups)} val groups, skipped {skipped_val} val questions (not enough trajectories).")
    assert skipped_train == 0 and skipped_val == 0, (
        "coverage-first requires every question to form groups. "
        f"skipped_train={skipped_train}, skipped_val={skipped_val}"
    )

    def _collect_total_indices_from_questions(question_list: list[Any]) -> set[int]:
        return {int(ex["total_index"]) for q in question_list for ex in question_to_examples[q]}

    def _collect_total_indices_from_groups(groups: list[list[dict[str, Any]]]) -> set[int]:
        return {int(ex["total_index"]) for group in groups for ex in group}

    train_expected = _collect_total_indices_from_questions(train_questions)
    train_covered = _collect_total_indices_from_groups(train_groups)
    assert train_expected == train_covered, (
        "train coverage failed: "
        f"missing={sorted(train_expected - train_covered)[:20]} "
        f"extra={sorted(train_covered - train_expected)[:20]}"
    )

    val_expected = _collect_total_indices_from_questions(val_questions)
    val_covered = _collect_total_indices_from_groups(val_groups)
    assert val_expected == val_covered, (
        "val coverage failed: "
        f"missing={sorted(val_expected - val_covered)[:20]} "
        f"extra={sorted(val_covered - val_expected)[:20]}"
    )

    train_dataset = pd.DataFrame([process_group_fn(g, "train", idx, args.label_form) for idx, g in enumerate(train_groups)])
    val_dataset = pd.DataFrame([process_group_fn(g, "val", idx, args.label_form) for idx, g in enumerate(val_groups)])




    # ===================================
    # train_miss_number = len([group for group in train_groups if group[0]['final_label'] != group[1]['final_label']])
    #
    # print('train miss ratio:', train_miss_number / len(train_groups))
    #
    # val_miss_number = len([group for group in val_groups if group[0]['final_label'] != group[1]['final_label']])
    #
    # print('train miss ratio:', val_miss_number / len(val_groups))

    #
    from collections import Counter
    # 提取所有 ori_data_source
    sources = [
        ex["data_source"]
        for group in val_groups
        for ex in group
    ]

    # 统计频数
    counter = Counter(sources)

    total = len(sources)

    # 打印比例
    print("=== val_data ori_data_source Distribution ===")
    for k, v in counter.items():
        print(f"{k}: {v} ({v / total:.2%})")
    # ===================================

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    assert os.path.isdir(local_dir), f"Failed to create output directory: {local_dir}"
    name = f"{args.label_form}_train_group_{args.strategy}_K{args.K}_T{args.traj_num}.parquet"
    train_dataset.to_parquet(os.path.join(local_dir, name))
    print('path:', os.path.join(local_dir, name))
    print('length:', len(train_dataset))


    name = f"{args.label_form}_dev_group_{args.strategy}_K{args.K}_T{args.traj_num}.parquet"
    val_dataset.to_parquet(os.path.join(local_dir, name))
    print('path:', os.path.join(local_dir, name))
    print('length:', len(val_dataset))
