"""
Preprocess the GAIA dev dataset to parquet format
"""

import argparse
import os
import re
import pandas as pd
import datasets
import pickle
from copy import deepcopy
import json
import random


import json

def split_trajectory(ori_traj: str, special_token):
    """
    Split a trajectory string into structured steps (including tool calls and final answer).
    Each step will include a step_id and step_type, making it easier to count steps later.
    """

    # Split by assistant's responses — you can adjust delimiter based on your data format
    traj_list = [chunk.strip() for chunk in ori_traj.split('<|im_start|>assistant\n') if chunk.strip()]


    # 去掉system prompt和特殊字符
    traj_list = [chunk.replace('<|im_start|>','').replace('<|im_end|>','') for chunk in traj_list if '<|im_start|>system' not in chunk]
    steps = []
    for idx, content in enumerate(traj_list):
        steps.append({
            "step_id": idx,
            "content": content
        })

    # 输出结构化 JSON（list 而不是 dict，方便模型数 step）
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/fanshengda/verl/input_data/near_miss_prm")
    parser.add_argument("--metadata_path",
                        default="/workspace/fanshengda/verl/input_data/near_miss_pairs_1108.pkl")
    args = parser.parse_args()

    data_source = "searchR1_near_miss_prm"

    with open(args.metadata_path, "rb") as f:
        metadata = pickle.load(f)


    # add a row to each data item that represents a unique id

    def process_fn(example, idx):
        question = example[0]
        dataset_name = example[1][1]['data_source']
        if random.random() < 0.5:
            tag1, traj1 = example[1]
            tag2, traj2 = example[2]
        else:
            tag1, traj1 = example[2]
            tag2, traj2 = example[1]

        data = {
            "data_source": data_source,
            "prompt": traj1['messages'],
            "ability": "prm",
            "reward_model": {
                "style": "rule",
                "ground_truth": tag1
            },
            "extra_info": {
                # placeholder; will be assigned after grouping by dataset_name
                'split': 'train',
                'index': idx,
                'question': question,
                'trajectory': traj1,
                'anchor': traj2,
                "need_tools_kwargs": False,
                'dataset_name': dataset_name
                # Ensure non-empty per-tool kwargs to avoid PyArrow struct<> write error
                # Ref: ArrowNotImplementedError: Cannot write struct type with no child field
            }
        }
        return data

    # First pass: build the full dataframe
    full_dataset = pd.DataFrame([process_fn(row, idx) for idx, row in enumerate(metadata)])

    # Extract dataset_name to a top-level helper column for grouping
    full_dataset['dataset_name'] = full_dataset['extra_info'].apply(lambda x: x.get('dataset_name'))

    # Determine validation indices: per dataset_name, take the first 10% by original order
    val_indices = []
    for dname, grp_idx in full_dataset.groupby('dataset_name').groups.items():
        grp_df = full_dataset.loc[list(grp_idx)]
        # sort by the original enumeration order stored in extra_info['index'] to ensure stability
        order = grp_df['extra_info'].apply(lambda x: x.get('index', 0))
        grp_df = grp_df.loc[order.sort_values().index]
        n = len(grp_df)
        k = int(n * 0.1)
        if k > 0:
            val_indices.extend(list(grp_df.index[:k]))

    # Assign split column
    full_dataset['split'] = 'train'
    if val_indices:
        full_dataset.loc[val_indices, 'split'] = 'validation'

    # Also mirror split into nested extra_info['split'] for downstream compatibility
    full_dataset['extra_info'] = full_dataset.apply(
        lambda r: {**r['extra_info'], 'split': r['split']}, axis=1
    )

    # Create split datasets
    train_dataset = full_dataset[full_dataset['split'] == 'train'].drop(columns=['dataset_name', 'split'])
    validation_dataset = full_dataset[full_dataset['split'] == 'validation'].drop(columns=['dataset_name', 'split'])

    # Save train/validation datasets
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, 'anchor_train_1112.parquet')
    val_path = os.path.join(local_dir, 'anchor_validation_1112.parquet')
    train_dataset.to_parquet(train_path)
    validation_dataset.to_parquet(val_path)


    print('train_path:', train_path)
    print('train_length:', len(train_dataset))
    print('validation_path:', val_path)
    print('validation_length:', len(validation_dataset))
