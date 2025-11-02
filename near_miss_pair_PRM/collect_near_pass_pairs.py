import json
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
path = "/workspace/fanshengda/verl/rollout_data/qwen3-4b_PureTool-n8-agentcpm_SFT_ASearcher1010-train"


data = []


for i in tqdm(range(1, 60)):
    file_path = f"{path}/{i}.jsonl"

    temp = pd.read_json(file_path, lines=True)
    data  += temp.to_dict(orient="records")


# 第一步筛选，去掉不含<answer>的
data = [sample for sample in data if '<answer>' in sample['output'] and '</answer>' in sample['output']]



gt2sample = defaultdict(list)

print(len(data))