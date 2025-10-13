import os
import re
import json
import time
import pickle
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
)
from trl import SFTConfig
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
warnings.filterwarnings("ignore")
import swanlab



if os.getenv('PYCHARM_HOSTED') != '1':
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=6))
    # Initialize the Accelerator
accelerator = Accelerator(mixed_precision='bf16')

# 只在主进程做一次副作用操作（登录/初始化/创建目录/写配置等）
if accelerator.is_main_process:
    os.environ["SWANLAB_API_KEY"] = "WoZrF9qolYJjzYBCfArih"     # <<< 填你的 key
    os.environ["SWANLAB_WORKSPACE"] = "AgentCPM_MCP"             # <<< 你的 workspace

# ==========================
# 工具：长度分位点 + 两种预处理（truncate/drop）
# ==========================
def preprocess_data_truncate(data, tokenizer, max_seq_length):
    input_ids_list = []
    input_lengths = []
    for input_text in tqdm(data, desc="tokenizing(trunc=True)"):
        tokenized = tokenizer(input_text, return_tensors="pt",
                              truncation=True, max_length=max_seq_length)
        ids = tokenized["input_ids"][0].tolist()
        input_ids_list.append(ids)
        input_lengths.append(len(ids))

    quantiles = [20, 40, 60, 80, 90, 95]
    percentiles = np.percentile(input_lengths, quantiles)
    print("\n📊 Input Length Percentiles:")
    for q, p in zip(quantiles, percentiles):
        print(f"{q}th percentile: {int(p)} tokens")

    df = pd.DataFrame({"input_ids": input_ids_list})
    return Dataset.from_pandas(df)


def preprocess_data_drop(data, tokenizer, max_seq_length):
    input_ids_list = []
    input_lengths = []
    for input_text in tqdm(data, desc="tokenizing(trunc=False)"):
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=False)
        ids = tokenized["input_ids"][0].tolist()
        input_ids_list.append(ids)
        input_lengths.append(len(ids))

    quantiles = [20, 40, 60, 80, 90, 95]
    percentiles = np.percentile(input_lengths, quantiles)
    print("\n📊 Input Length Percentiles:")
    for q, p in zip(quantiles, percentiles):
        print(f"{q}th percentile: {int(p)} tokens")

    # 直接丢弃超过 max_len 的样本，提升监督密度
    input_ids_list = [s for s in input_ids_list if len(s) < max_seq_length]
    print("len(dataset) after drop:", len(input_ids_list))

    df = pd.DataFrame({"input_ids": input_ids_list})
    return Dataset.from_pandas(df)



# ==========================
# 关键1：子序列匹配（支持多次出现）
# ==========================
def find_all_subseq_indices(seq: torch.Tensor, sub: torch.Tensor):
    """返回所有 sub 在 seq 中的起始索引列表"""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return []
    hits = []
    # 为了效率，可用 rolling hash；这里直扫够用了
    for i in range(n - m + 1):
        if torch.equal(seq[i:i + m], sub):
            hits.append(i)
    return hits



# ==========================
# 关键2：Collator —— 直接产出 labels，非监督位=-100
# ==========================
# ========== Collator：labels 用 ignore_index，并“包含 end 标记” ==========
# ========== Collator：labels 用 ignore_index，并“包含 end 标记” ==========
class CollatorWithIgnoreIndex:
    def __init__(self, tokenizer, loss_start_token, loss_end_token,
                 include_end_token=True, report_supervised_ratio=True,
                 supervise_mode: str = "all"):
        """
        supervise_mode: "all" 或 "last"
          - "all": 监督所有 start/end 之间的内容（保持你原来的行为）
          - "last": 只监督最后一个 start 及其后最近的 end 之间的内容
        """
        self.tok = tokenizer
        self.start_ids = torch.tensor(
            self.tok.encode(loss_start_token, add_special_tokens=False),
            dtype=torch.long,
        )
        self.end_ids = torch.tensor(
            self.tok.encode(loss_end_token, add_special_tokens=False),
            dtype=torch.long,
        )
        self.include_end_token = include_end_token
        self.report = report_supervised_ratio
        self.supervise_mode = supervise_mode  # <<<< 新增
        self._reported = False

        print('pad token id:', self.tok.pad_token_id)
        print('start_ids:', self.start_ids)
        print('end_ids:', self.end_ids)
        print('supervise_mode:', supervise_mode)

    def __call__(self, features):
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            self.tok.pad_token = self.tok.eos_token
            pad_id = self.tok.pad_token_id

        item_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        padded = pad_sequence(item_ids, batch_first=True, padding_value=pad_id)
        attention_mask = (padded != pad_id).long()

        labels = torch.full_like(padded, fill_value=-100)

        total_supervised = 0
        total_tokens = 0

        for b in range(padded.size(0)):
            ids = padded[b]
            valid_len = attention_mask[b].sum().item()
            seq = ids[:valid_len]

            start_hits = find_all_subseq_indices(seq, self.start_ids)
            end_hits = find_all_subseq_indices(seq, self.end_ids)
            end_hits_sorted = sorted(end_hits)

            # 一个小函数：给定 start 索引，找 “> start” 的最近 end
            def nearest_end_after(start_idx: int) -> int | None:
                for eh in end_hits_sorted:
                    if eh > start_idx:
                        return eh
                return None

            if self.supervise_mode == "last":
                # 只取最后一个 start
                if start_hits:
                    s = start_hits[-1]
                    content_start = s + len(self.start_ids)
                    e = nearest_end_after(s)

                    if e is None:
                        content_end_exclusive = valid_len
                    else:
                        content_end_exclusive = min(e + len(self.end_ids), valid_len) if self.include_end_token else e

                    if content_end_exclusive > content_start:
                        labels[b, content_start:content_end_exclusive] = ids[content_start:content_end_exclusive]
                        total_supervised += (content_end_exclusive - content_start)

            else:
                # "all"：沿用你原先的逻辑，监督所有 start/end 片段
                for s in start_hits:
                    content_start = s + len(self.start_ids)
                    e = nearest_end_after(s)

                    if e is None:
                        content_end_exclusive = valid_len
                    else:
                        content_end_exclusive = min(e + len(self.end_ids), valid_len) if self.include_end_token else e

                    if content_end_exclusive > content_start:
                        labels[b, content_start:content_end_exclusive] = ids[content_start:content_end_exclusive]
                        total_supervised += (content_end_exclusive - content_start)

            total_tokens += valid_len

        if self.report:
            ratio = (total_supervised / max(total_tokens, 1)) if total_tokens > 0 else 0.0
            print(f"[Collator] Supervised token ratio in this batch: {ratio:.3f} "
                  f"({total_supervised}/{total_tokens})")
            if total_supervised == 0:
                print("[Collator][WARN] No supervised tokens found! "
                      "确认样本里包含 `<|im_start|>assistant` 与 `<|im_end|>`。")
            self._reported = True

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ========== Trainer：ignore_index ==========
class IgnoreIndexTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,   # 兼容高版本
        **kwargs,                                # 兜底将来新增参数
    ):
        global tokenizer
        # pop 出 labels（Trainer 会在 forward 前把所有张量搬到 device）
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits.float()


        # --------debug----------------


        if labels is not None and "input_ids" in inputs:
            sup = (labels != -100)             # [B,T] 监督mask
            bsz = labels.size(0)

            def mask_to_spans(mask_1d: torch.Tensor):
                idx = torch.nonzero(mask_1d, as_tuple=False).flatten()
                if idx.numel() == 0:
                    return []
                gaps = torch.where((idx[1:] - idx[:-1]) > 1)[0]
                starts = torch.cat([idx[[0]], idx[gaps + 1]])
                ends   = torch.cat([idx[gaps] + 1, idx[[-1]] + 1])  # 右开
                return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]

            # 只检查第一个有监督的样本，避免刷屏
            for b in range(bsz):
                spans = mask_to_spans(sup[b])
                if not spans:
                    continue
                for (st, ed) in spans:
                    # 注意 ed 是右开；最后一个受监督 token 在 ed-1
                    first_id = inputs["input_ids"][b, st].item()
                    last_id  = inputs["input_ids"][b, ed-1].item()

                    first_tok = tokenizer.convert_ids_to_tokens([first_id])[0]
                    last_tok  = tokenizer.convert_ids_to_tokens([last_id])[0]

                    print(f"[DEBUG] sample#{b} span=[{st},{ed}) "
                        f"first_id={first_id}({first_tok}) "
                        f"last_id={last_id}({last_tok})")

                    # （可选）看一下片段解码（截断防刷屏）
                    seg_ids = inputs["input_ids"][b, st:ed].tolist()
                    seg_txt = tokenizer.decode(seg_ids, skip_special_tokens=False)
                    if len(seg_txt) > 200:
                        seg_txt = seg_txt[:100] + " ...[trunc]..." + seg_txt[-100:]
                    print("       >>>", seg_txt.replace("\n", "\\n"))

                    # print('logits.shape:', logits.shape)
                    # print('labels.shape', labels.shape)
                    break

        # ============================

        if labels is None:
            # 没有 labels 时，回退到模型自带 loss（eval 或特殊场景）
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
            if loss is None:
                # 自己算一个（极少见）
                return (torch.tensor(0.0, device=logits.device), outputs) if return_outputs else torch.tensor(0.0, device=logits.device)
            return (loss, outputs) if return_outputs else loss

        # 自回归 shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # debug
        print('shift_logits.shape:', shift_logits.shape)
        print('shift_labels.shape:', shift_labels.shape)

        # 1) sum 而不是 mean
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 2) 用 Trainer 传入的 num_items_in_batch 来做正规一化（否则退回到本 batch 的有效 token 数）
        num_items = num_items_in_batch

        # 放到同一 device，避免分布式/模型并行下的设备不一致
        if hasattr(num_items, "device") and num_items.device != shift_logits.device:
            num_items = num_items.to(shift_logits.device)

        # 防御 0
        num_items = torch.clamp(num_items, min=1)
        loss = loss / num_items
        loss = loss * self.accelerator.num_processes


        print('loss:', loss.item())
        print('num_items_in_batch:', num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

def main():
    # ===== 基本参数 =====
    max_seq_length = 26000
    # 如果设置为all，则使用默认的全部监督逻辑；若设置为last，则使用最后一步监督逻辑
    supervise_mode='all'
    model_path = "/home/test/test12/models/Qwen/Qwen3-4B"
    # ==================================================
    experiment_name = "Qwen3-4B-ASearcher_1010"
    output_path = f"./mcp_agent_ckpts/{experiment_name}"
    sft_data_path_list = [
        "./input/ASearcher_0926.pkl",
        "./input/ASearcher_0930.pkl",
        "./input/ASearcher_1003.pkl",
        "./input/ASearcher_1004.pkl",
        "./input/ASearcher_1006.pkl",
    ]

    # sft_data_path_list = [
    #     "./input/sft_data_ASearcher_AllThink.pkl",
    #     "./input/sft_data_scholarsearch_ToolHop_webshaper_R1AllThink.pkl",
    # ]

    learning_rate = 2e-5
    num_train_epochs = 3
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4
    warmup_ratio = 0.03

    loss_start_token = "<|im_start|>assistant"
    loss_end_token = "<|im_end|>"

    print(f"Experiment: {experiment_name}")
    # if int(os.environ.get("RANK", "0")) == 0:
    #     swanlab.init(api_key='WoZrF9qolYJjzYBCfArih',  run_name=experiment_name)

    # ===== 数据 =====
    raw_list = []
    for sft_data_path in sft_data_path_list:
        with open(sft_data_path, "rb") as fp:
            temp = pickle.load(fp)  # 每条是带 chat 模板的完整文本
        raw_list += temp

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = preprocess_data_drop(raw_list, tokenizer, max_seq_length)

    # ===== 模型 =====
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    # ===== 优化器 & 调度器 =====
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(dataset) // (per_device_train_batch_size * world_size)
    total_training_steps = (steps_per_epoch * num_train_epochs) // gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * warmup_ratio),
        num_training_steps=total_training_steps,
    )

    # ===== Trainer 参数 =====
    training_args = SFTConfig(
        output_dir=output_path,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        # lr_scheduler_type="cosine",
        # warmup_ratio=warmup_ratio,
        report_to="swanlab",
        run_name=experiment_name
        )

    # ===== Collator（闭区间，含 end 标记） =====
    data_collator = CollatorWithIgnoreIndex(
        tokenizer=tokenizer,
        loss_start_token=loss_start_token,
        loss_end_token=loss_end_token,
        include_end_token=True,                 # 关键：包含 <|im_end|>
        report_supervised_ratio=True,
        supervise_mode=supervise_mode
    )

    trainer = IgnoreIndexTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()


if __name__ == "__main__":
    main()