import json
import pickle
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import *
from verl.utils.reward_score.search_r1_like_qa_em import compute_score


import os
import time
import re
import string
import re
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
from omegaconf import OmegaConf





# ==============CONFIG=================


reward_type = "llm"
llm_judger_model = "Qwen2.5-32B-Instruct"


# ==============CONFIG=================
def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        with _llm_client_lock:
            if _llm_client is None:
                _llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _llm_client


def _llm_scorer_function(model_answer: str, ground_truth: str, question: str, model: str) -> bool:
    if not model:
        raise Exception("No LLM model specified")

    LLM_JUDGE_PROMPT_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

    Question: {question}

    Labeled Answer: {labeled_answer}

    Predicted Answer: {pred_answer}

    Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
    """.strip()

    # Build the concrete prompt by injecting strings
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=str(question or ""),
        labeled_answer=str(ground_truth or ""),
        pred_answer=str(model_answer or ""),
    )

    t_ba_start = time.time()
    try:
        client = _get_llm_client()
    except Exception as e:
        raise Exception("Initialize browser_agent client failed: %s", e)

    attempts = 5
    for i in range(attempts):
        t1 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1.0,
                n=1,
                max_tokens=16,
                timeout=30,
            )
            content = ""
            if getattr(resp, "choices", None):
                content = resp.choices[0].message.content or ""
            norm = (content or "").strip().lower()
            latency_ms = int((time.time() - t1) * 1000)
            if "incorrect" in norm:
                return False
            if "correct" in norm:
                return True

        except Exception as e:
            print("llm_scorer attempt %d failed: %s", i + 1, e)

    raise Exception("llm_scorer failed")



def llm_scorer(prediction, golden_answers, question, model):
    prediction = extract_solution(prediction)
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        # if golden_answer == normalized_prediction:
        if _llm_scorer_function(model_answer=normalized_prediction, ground_truth=golden_answer, question=question, model=model):
            score = 1
            break
    return score



tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen/Qwen2.5-1.5B-Instruct")

MAX_WORKERS = max(1, int(os.environ.get("NEAR_PASS_WORKERS", 32)))
tokenizer_lock = threading.Lock()

OPENAI_BASE_URL = os.environ.get("NEAR_PASS_OPENAI_BASE", "http://localhost:8888/v1")
OPENAI_API_KEY = os.environ.get("NEAR_PASS_OPENAI_KEY", "")
_llm_client = None
_llm_client_lock = threading.Lock()


# with open('/workspace/fanshengda/verl/input_data/near_miss_pairs_1106.pkl', 'rb') as fp:
#     pickle.load(fp)


path_list = [
    "/workspace/fanshengda/verl/rollout_data/tree_sampling/tree_sampling_train_hotpotqa_0_20000_20251104_062539.jsonl",
    "/workspace/fanshengda/verl/rollout_data/tree_sampling/tree_sampling_train_nq_0_20000_20251104_175629.jsonl",
]


data = []

# 我们这里不需要去掉不带answer的，都有
for file_path in path_list:
    temp = pd.read_json(file_path, lines=True)
    data  += temp.to_dict(orient="records")


query2sample = defaultdict(lambda: defaultdict(list))



def _score_and_render_sample(sample):
    """Evaluate a single sample and attach score/output."""
    processed = sample.copy()
    ground_truth = processed['ground_truth']
    solution_str = processed['leaf_output']

    if reward_type == "rule":
        processed['score'] = compute_score(solution_str=solution_str, ground_truth=ground_truth)
    elif reward_type == "llm":
        processed['score'] = llm_scorer(
            prediction=solution_str,
            golden_answers=ground_truth["target"],
            question=processed['question'],
            model=llm_judger_model,
        )

    # Tokenizer is not documented as thread-safe; guard the call.
    with tokenizer_lock:
        processed['output'] = tokenizer.apply_chat_template(processed['messages'][2:], tokenize=False)

    return processed['question'], processed


def _iter_samples_with_workers(samples, max_workers):
    if max_workers <= 1:
        for sample in samples:
            yield _score_and_render_sample(sample)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_score_and_render_sample, samples):
            yield result



# data = data[:10000]
for question, sample in tqdm(_iter_samples_with_workers(data, MAX_WORKERS), total=len(data)):
    query2sample[question]['gold_label'] = sample['ground_truth']
    if sample['score'] == 1:
        query2sample[question]['positive'].append(sample)
    else:
        query2sample[question]['negative'].append(sample)


contrastive_query2sample = {key: value for key, value in query2sample.items() if 'positive' in value.keys() and 'negative' in value.keys()}

all_near_miss_pairs = []


N = 3
for query, sample_bucket in tqdm(contrastive_query2sample.items()):
    positives = sample_bucket.get('positive', [])
    negatives = sample_bucket.get('negative', [])

    pos_neg_pairs = []
    for positive_sample in positives:
        for negative_sample in negatives:
            if positive_sample['output'] == negative_sample['output']:
                continue
            similarity = tool_call_similarity(positive_sample['output'], negative_sample['output'])
            pos_neg_pairs.append((query, ('positive', positive_sample), ('negative', negative_sample), similarity))

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


# 1108 ： LLM judge
with open('./input_data/near_miss_pairs_1108.pkl', 'wb') as f:
    pickle.dump(all_near_miss_pairs, f)

print('length:',len(all_near_miss_pairs))
