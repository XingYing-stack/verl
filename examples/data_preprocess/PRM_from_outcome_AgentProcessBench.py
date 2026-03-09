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
random.seed(108)

import json

def split_trajectory(ori_traj: str):
    """
    Split a trajectory string into structured steps (including tool calls and final answer).
    Each step will include a step_id and step_type, making it easier to count steps later.
    """

    # Split by assistant's responses — you can adjust delimiter based on your data format
    traj_list = [chunk.strip() for chunk in ori_traj.split('\nassistant\n') if chunk.strip()]

    steps = []
    for idx, content in enumerate(traj_list):
        steps.append({
            "step_id": idx,
            "content": content
        })

    # 输出结构化 JSON（list 而不是 dict，方便模型数 step）
    return json.dumps(steps, indent=2, ensure_ascii=False)


JUDGE_RUBRIC = """You are a strict but fair trajectory annotator for tool-use agents.

You will be given one complete trajectory consisting of system, user, assistant,
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


def _assistant_message_indices(messages: Any) -> list[int]:
    if not isinstance(messages, list):
        return []
    indices: list[int] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            indices.append(i)
    return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/verl/input_data/PRM_from_outcome")
    parser.add_argument("--metadata_path",
                        default="/workspace/AgentProcessBench_KDD-main/data")

    parser.add_argument("--label_form", default="process", choices=["process", "outcome"])

    args = parser.parse_args()

    data_source = "AgentProcessBench"

    base_path = f"{args.metadata_path}/AgentProcessBench"
    assert os.path.isdir(base_path), f"base_path does not exist: {base_path}"

    all_data = []
    jsonl_files = glob.glob(os.path.join(base_path, "**", "*.jsonl"), recursive=True)
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


    # add a row to each data item that represents a unique id

    def process_fn(example, split ,idx, label_form):
        assert split in {"train", "val"}, f"Unexpected split: {split}"
        assert isinstance(example, dict), "Each example must be a dict."
        required_keys = {"messages", "final_label", "data_source", "total_index", "step_labels", "question"}
        missing_keys = required_keys - set(example.keys())
        assert not missing_keys, f"Missing required keys in example: {sorted(missing_keys)}"
        assert isinstance(example["messages"], list), "example['messages'] must be a list."
        assert isinstance(example["step_labels"], (list, dict)), "example['step_labels'] must be list or dict."

        assistant_indices = _assistant_message_indices(example.get("messages"))
        assert assistant_indices, "Each example must contain at least one assistant message."
        messages = _build_judge_input(
            item=example,
            assistant_indices=assistant_indices,
        )
        assert isinstance(messages, list) and len(messages) == 2, "_build_judge_input must return 2-message prompt."

        if label_form == "process":
            ground_truth = example['step_labels']
        elif label_form == "outcome":
            ground_truth = example['final_label']
        else:
            raise ValueError(f"Unexpected label_form: {label_form}")


        # NOTE： 我们可以约定如果ground_truth为数字，就用ORM，否则用PRM
        data = {
            "data_source": data_source,
            "prompt": messages,
            "ability": f"{label_form}_reward_model",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                'split': split,
                'assistant_indices': assistant_indices,
                'reward_type': label_form,
                'ori_data_source': example['data_source'],
                'index': example['total_index'],
                'final_label': example['final_label'],
                'step_labels': example['step_labels'],
                'question': example['question'],
                'data_source': example['data_source'],
                "need_tools_kwargs": False,
            }
        }
        return data



    n = len(all_data)
    indices = list(range(n))
    random.shuffle(indices)

    split = int(n * 0.8)

    train_idx = indices[:split]
    val_idx = indices[split:]
    assert len(train_idx) + len(val_idx) == n, "Train/val split size mismatch."
    assert set(train_idx).isdisjoint(val_idx), "Train/val split overlap detected."

    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    train_dataset = pd.DataFrame([
        process_fn(row, 'train', idx, args.label_form) for idx, row in enumerate(train_data)
    ])

    val_dataset = pd.DataFrame([
        process_fn(row, 'val', idx, args.label_form) for idx, row in enumerate(val_data)
    ])




    # ===================================
    from collections import Counter
    # 提取所有 ori_data_source
    sources = [
        item["data_source"]
        for item in val_data
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
    name = f"{args.label_form}_train_0228.parquet"
    train_dataset.to_parquet(os.path.join(local_dir, name))
    print('path:', os.path.join(local_dir, name))
    print('length:', len(train_dataset))


    name = f"{args.label_form}_dev_0228.parquet"
    val_dataset.to_parquet(os.path.join(local_dir, name))
    print('path:', os.path.join(local_dir, name))
    print('length:', len(val_dataset))