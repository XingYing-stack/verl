import os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Subset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from verl.utils.dataset.marker_anchored_multiturn_sft_dataset import MarkerAnchoredMultiTurnSFTDataset
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='bf16')

# 只在主进程做一次副作用操作（登录/初始化/创建目录/写配置等）
if accelerator.is_main_process:
    os.environ["SWANLAB_API_KEY"] = "WoZrF9qolYJjzYBCfArih"     # <<< 填你的 key



# ============ 只在最后一个非 PAD 位置打 label 的 collator ============
class DataCollatorForLastTokenClassification:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 明确只取这几个 key，避免各种奇怪类型
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        traj_label_list = [f["trajectory_labels"] for f in features]

        # input_ids / attention_mask 一般已经是 tensor，直接 stack
        input_ids = torch.stack(input_ids_list)           # (B, L)
        attention_mask = torch.stack(attention_mask_list) # (B, L)

        # trajectory_labels 可能是 int 也可能是 tensor，统一变成 (B,)
        if isinstance(traj_label_list[0], torch.Tensor):
            traj_labels = torch.stack(traj_label_list).view(-1).long()
        else:
            traj_labels = torch.tensor(traj_label_list, dtype=torch.long)

        B, L = input_ids.shape
        labels = torch.full((B, L), fill_value=-100, dtype=torch.long)

        # 最后一个非 PAD 的 index：attention_mask.sum(dim=-1) - 1
        last_indices = attention_mask.sum(dim=-1) - 1
        last_indices = last_indices.clamp(min=0)

        for i in range(B):
            idx = last_indices[i].item()
            labels[i, idx] = traj_labels[i].item()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==================== 评估指标 ====================
def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (B, L, C)  token 分类 logits
    eval_pred.label_ids:  (B, L)      只有一个位置是 0/1，其余为 -100
    """
    logits = eval_pred.predictions          # np.ndarray, (B, L, C)
    labels = eval_pred.label_ids            # np.ndarray, (B, L)

    pred_ids = logits.argmax(-1)            # (B, L)

    mask = labels != -100                   # (B, L)
    if mask.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = labels[mask].astype(np.int64)  # (N,)
    y_pred = pred_ids[mask].astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def safe_div(n, d):
        return float(n) / float(d) if d != 0 else 0.0

    accuracy  = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    prefix = '/home/test/test12'

    model_name = prefix + "/models/Qwen/Qwen2.5-7B-Instruct"
    output_dir = prefix + "/fanshengda/verl/prm_ckpts_last_token_ce"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    parquet_paths = [
        prefix + "/fanshengda/verl/input_data/near_miss_prm/anchor_train_1116.parquet"
    ]

    marker_cfg = {
        "marker_anchor": {
            "enable": True,
            "token": "<extra_0>",
            "loss_mask_value": 1,   # 让 <extra_0> 的 loss_mask 为 1，方便我们当作 marker_mask
            "roles": ["tool"], # 或 ["tool"]，按你上面的实现来
        },
        # MultiTurnSFTDataset 其他配置（max_length, truncation 等）也可以放这
        "max_length": 8192* 2,
        "multiturn": {"messages_key": "prompt", "tools_key": "tools"},
        "truncation": "error",
    }


    full_dataset = MarkerAnchoredMultiTurnSFTDataset(parquet_files=parquet_paths, tokenizer=tokenizer, config=marker_cfg)


    sample = full_dataset[0]
    print('dada')
    assert "trajectory_labels" in sample, "dataset 需要提供 trajectory_labels (0/1)!"

    # ============ 95% 训练 / 5% 验证 ============
    N = len(full_dataset)
    rng = np.random.default_rng(seed=42)
    indices = np.arange(N)
    rng.shuffle(indices)

    split = max(1, int(N * 0.95))
    train_indices = indices[:split].tolist()
    eval_indices  = indices[split:].tolist()

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset  = Subset(full_dataset, eval_indices)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        eval_steps=200,
        load_best_model_at_end=True,
        eval_strategy='steps',
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42,
        report_to='swanlab',   # 你要接 SwanLab/W&B 再改
        # ✅ 关键：lr 降到 1e-6 / 3e-6 这个级别
        learning_rate=3e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )

    collator = DataCollatorForLastTokenClassification()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
