# accelerate launch --config_file /home/test/test12/.cache/huggingface/accelerate/fsd_trl_sft.yaml near_miss_pair_PRM/train_marker_prm_with_trl.py
import os
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset
import torch.distributed as dist
from datetime import timedelta

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from verl.utils.dataset.marker_anchored_multiturn_sft_dataset import MarkerAnchoredMultiTurnSFTDataset
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

accelerator = Accelerator(mixed_precision='bf16')

# 只在主进程做一次副作用操作（登录/初始化/创建目录/写配置等）
if accelerator.is_main_process:
    os.environ["SWANLAB_API_KEY"] = "WoZrF9qolYJjzYBCfArih"     # <<< 填你的 key


# ============ 只在最后一个非 PAD 位置打 label 的 collator ============
class DataCollatorForTokenClassification:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 明确只取这几个 key，避免各种奇怪类型
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        traj_label_list = [f["trajectory_labels"] for f in features]
        loss_mask_list = [f["loss_mask"] for f in features]   # ✅ 新增

        # input_ids / attention_mask 一般已经是 tensor，直接 stack
        input_ids = torch.stack(input_ids_list)           # (B, L)
        attention_mask = torch.stack(attention_mask_list) # (B, L)

        # trajectory_labels 可能是 int 也可能是 tensor，统一变成 (B,)
        if isinstance(traj_label_list[0], torch.Tensor):
            traj_labels = torch.stack(traj_label_list).view(-1).long()
        else:
            traj_labels = torch.tensor(traj_label_list, dtype=torch.long)

        # loss_mask 可能是 list[int] / list[float] / tensor，统一成 (B, L) float
        if isinstance(loss_mask_list[0], torch.Tensor):
            loss_mask = torch.stack(loss_mask_list).float()
        else:
            loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)  # (B, L)

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
            "labels": labels,  # HF 默认用这个当 label_ids
            "loss_mask": loss_mask,  # ✅ 我们自定义 loss 用
            "trajectory_labels": traj_labels,  # ✅ 每个样本一个 label
        }



class MarkerPRMTrainer(Trainer):
    # def _aggregate_logits_and_labels(
    #         self,
    #         outputs,
    #         loss_mask: torch.Tensor,
    #         traj_labels: torch.Tensor,
    # ):
    #     """
    #     outputs.logits: (B, L, C)
    #     loss_mask:     (B, L)   —— marker 位置为 1，其余为 0
    #     traj_labels:   (B,)
    #     -> agg_logits: (B, C), traj_labels: (B,)
    #     """
    #     logits = outputs.logits.float()              # (B, L, C)
    #     loss_mask = loss_mask.to(logits.device)      # (B, L)
    #
    #     B, L, C = logits.shape
    #
    #     # 1. 找出每个样本最后一个 loss_mask == 1 的位置
    #     mask_bool = loss_mask > 0                   # (B, L) bool
    #     # 如果你确信每个样本至少一个 marker，可以直接 assert
    #     assert mask_bool.any(dim=1).all(), "some samples have no marker (loss_mask 全 0)"
    #
    #     # 位置索引 [0, 1, 2, ..., L-1]
    #     positions = torch.arange(L, device=logits.device).unsqueeze(0).expand(B, L)
    #     # 非 marker 位置设为 -1，这样 max 就会选到最后一个 True 的那个 index
    #     positions = positions.masked_fill(~mask_bool, -1)
    #     last_idx = positions.max(dim=1).values      # (B,)
    #
    #     # 2. 取出每个样本最后一个 marker 对应的 logits: (B, C)
    #     batch_idx = torch.arange(B, device=logits.device)
    #     agg_logits = logits[batch_idx, last_idx]    # (B, C)
    #
    #     # 3. labels 搬到同一设备
    #     traj_labels = traj_labels.to(logits.device).long()  # (B,)
    #
    #     return agg_logits, traj_labels


    # 能不能也学一个重要性呢？直接加权和，学一个重要重要程度？
    def _aggregate_logits_and_labels(
            self,
            outputs,
            loss_mask: torch.Tensor,
            traj_labels: torch.Tensor,
    ):
        """
        outputs.logits: (B, L, C)
        loss_mask:     (B, L)
        traj_labels:   (B,)
        -> agg_logits: (B, C), traj_labels: (B,)
        """
        logits = outputs.logits.float()  # (B, L, C)
        loss_mask = loss_mask.to(logits.device)  # (B, L)
        loss_mask = loss_mask.unsqueeze(-1)  # (B, L, 1)

        agg_logits = (logits * loss_mask).sum(dim=1)  # (B, C)
        traj_labels = traj_labels.to(logits.device).long()  # (B,)

        return agg_logits, traj_labels

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
    ):
        # 取出我们自己的监督信号
        traj_labels = inputs.pop("trajectory_labels")  # (B,)
        loss_mask = inputs.pop("loss_mask")  # (B, L)
        # 不需要默认的 token-level loss，去掉 labels，防止模型多算一次
        inputs.pop("labels", None)

        outputs = model(**inputs)
        agg_logits, traj_labels = self._aggregate_logits_and_labels(
            outputs, loss_mask, traj_labels
        )

        loss = F.cross_entropy(agg_logits, traj_labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        """
        让 eval/predict 阶段返回：
        - loss: aggregated PRM loss
        - logits: agg_logits (B, 2)
        - labels: traj_labels (B,)
        这样 compute_metrics 就是纯 PRM 语义了。
        """
        # 拿出我们自己的 label & mask
        traj_labels = inputs.pop("trajectory_labels")  # (B,)
        loss_mask = inputs.pop("loss_mask")  # (B, L)
        # 同样去掉 token-level labels，避免多算
        inputs.pop("labels", None)

        # HF 的标准预处理（放到正确 device，上 fp16/bf16 等）
        inputs = self._prepare_inputs(inputs)

        has_labels = traj_labels is not None

        with torch.no_grad():
            outputs = model(**inputs)
            agg_logits, traj_labels = self._aggregate_logits_and_labels(
                outputs, loss_mask, traj_labels
            )

            loss = None
            if has_labels:
                loss = F.cross_entropy(agg_logits, traj_labels)

        if prediction_loss_only:
            return (loss, None, None)

        # 注意这里返回的是：
        # logits = (B, 2)，labels = (B,)
        return (loss, agg_logits, traj_labels)


# ==================== 评估指标 ====================
def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (N, 2)  —— aggregated logits
    eval_pred.label_ids:  (N,)     —— trajectory_labels (0/1)
    """
    logits = eval_pred.predictions      # (N, 2)
    labels = eval_pred.label_ids        # (N,)

    # 有些版本会给 (N, 1)，保险起见 squeeze 一下
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    preds = logits.argmax(-1).astype(np.int64).reshape(-1)

    assert preds.shape == labels.shape

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

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
    output_dir = prefix + "/fanshengda/verl/prm_ckpts/last_token_ce"

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

    # ✅ 1. 增加新 token
    new_tokens = ["<extra_0>"]
    added = tokenizer.add_tokens(new_tokens)
    print(f"Added {added} new tokens")

    # ✅ 2. 同步模型 embedding
    model.resize_token_embeddings(len(tokenizer))

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
    assert "trajectory_labels" in sample, "dataset 需要提供 trajectory_labels (0/1)!"
    assert 'loss_mask' in sample, "dataset 需要提供 loss_mask (0/1)!"
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
        warmup_ratio=0.03,
    )

    collator = DataCollatorForTokenClassification()

    trainer = MarkerPRMTrainer(
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
