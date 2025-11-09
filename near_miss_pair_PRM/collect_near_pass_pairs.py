import json
import pickle
import re
from collections import defaultdict
from itertools import combinations

import pandas as pd
from tqdm import tqdm

from utils import *
def remove_think_tags(input_string):
    if not input_string.startswith("<think>") and '<think>' in input_string and '</think>' in input_string:
        input_string = '<think>\n' + input_string
    # 使用正则表达式去除 <think> 和 </think> 标签之间的内容
    result = re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    result = result.strip()
    return result


path = "/workspace/fanshengda/verl/rollout_data/qwen3-4b-2507_1010SFT_PureTool-n8-agentcpm-train"


data = []


for i in tqdm(range(75, 89)):
    file_path = f"{path}/{i}.jsonl"

    temp = pd.read_json(file_path, lines=True)
    data  += temp.to_dict(orient="records")


# 第一步筛选，去掉不含<answer>的
data = [sample for sample in data if '<answer>' in sample['output'] and '</answer>' in sample['output']]

# 第二步筛选，去掉<think>
for sample in tqdm(data):
    sample['output'] = remove_think_tags(sample['output'])
    sample['query'] =sample['input'].split('question:')[-1].split('assistant')[0].strip()

query2sample = defaultdict(lambda: defaultdict(list))

for sample in tqdm(data):
    query2sample[sample['query']]['gold_label'] = sample['gts']
    if sample['score'] == 1:
        query2sample[sample['query']]['positive'].append(sample)
    else:
        query2sample[sample['query']]['negative'].append(sample)

all_near_miss_pairs = []


N = 3
for query, sample_bucket in query2sample.items():
    positives = sample_bucket.get('positive', [])
    negatives = sample_bucket.get('negative', [])

    pos_neg_pairs = []
    for positive_sample in positives:
        for negative_sample in negatives:
            if positive_sample['output'] == negative_sample['output']:
                continue
            similarity = tool_call_similarity(positive_sample['output'], negative_sample['output'])
            pos_neg_pairs.append((query, ('positive', positive_sample['output']), ('negative', negative_sample['output']), similarity))

    pos_pos_pairs = []
    # for sample_a, sample_b in combinations(positives, 2):
    #     if tool_call_similarity(sample_a['output'], sample_b['output']) > 1.0:
    #         continue
    #     similarity = tool_call_similarity(sample_a['output'], sample_b['output'])
    #     pos_pos_pairs.append((query, ('positive', sample_a['output']), ('positive', sample_b['output']), similarity))
    #
    neg_neg_pairs = []
    # for sample_a, sample_b in combinations(negatives, 2):
    #     if tool_call_similarity(sample_a['output'], sample_b['output']) == 1.0:
    #         continue
    #     similarity = tool_call_similarity(sample_a['output'], sample_b['output'])
    #     neg_neg_pairs.append((query, ('negative', sample_a['output']), ('negative', sample_b['output']), similarity))

    query_pairs = []
    for pairs in (pos_neg_pairs, pos_pos_pairs, neg_neg_pairs):
        if pairs:
            query_pairs.extend(sorted(pairs, key=lambda x: x[3], reverse=True)[:N])

    all_near_miss_pairs.extend(query_pairs)



with open('./input_data/near_miss_pairs_1016.pkl', 'wb') as f:
    pickle.dump(all_near_miss_pairs, f)

print('length:',len(all_near_miss_pairs))
