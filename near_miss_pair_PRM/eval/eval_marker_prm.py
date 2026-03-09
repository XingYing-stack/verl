# eval_marker_prm.py
import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from verl.utils.dataset.marker_anchored_multiturn_sft_dataset import (
    MarkerAnchoredMultiTurnSFTDataset,
)


# ===== collator：基本沿用你训练里的版本 =====
class DataCollatorForTokenClassification:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        traj_label_list = [f["trajectory_labels"] for f in features]
        loss_mask_list = [f["loss_mask"] for f in features]

        input_ids = torch.stack(input_ids_list)          # (B, L)
        attention_mask = torch.stack(attention_mask_list)  # (B, L)

        # trajectory_labels -> (B,)
        if isinstance(traj_label_list[0], torch.Tensor):
            traj_labels = torch.stack(traj_label_list).view(-1).long()
        else:
            traj_labels = torch.tensor(traj_label_list, dtype=torch.long)

        # loss_mask -> (B, L)
        if isinstance(loss_mask_list[0], torch.Tensor):
            loss_mask = torch.stack(loss_mask_list).float()
        else:
            loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)

        B, L = input_ids.shape
        labels = torch.full((B, L), fill_value=-100, dtype=torch.long)

        last_indices = attention_mask.sum(dim=-1) - 1
        last_indices = last_indices.clamp(min=0)

        for i in range(B):
            idx = last_indices[i].item()
            labels[i, idx] = traj_labels[i].item()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "trajectory_labels": traj_labels,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval Marker PRM: 对给定 parquet 文件中所有样本的 marker 位置打分并保存为 jsonl"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/nfsdata/fanshengda/verl/prm_ckpts/mean_mse_entropy_1209/checkpoint-2600",
        help="训练好 PRM 的模型目录（trainer.save_model 的 output_dir）",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="/nfsdata/fanshengda/verl/input_data/near_miss_prm/anchor_validation_1116.parquet",
        help="要评估的 parquet 文件路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/nfsdata/fanshengda/verl/near_miss_pair_PRM/output/prm_anchor_validation_1209_ckpt2600.jsonl',
        help="输出 json/jsonl 路径（建议 .jsonl）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="eval batch size（别太大，受限于 16k/32k 序列长度）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192 * 2,
        help="和训练时一样的 max_length",
    )
    parser.add_argument(
        "--marker_token",
        type=str,
        default="<extra_0>",
        help="训练时用的 marker token",
    )
    parser.add_argument(
        "--marker_roles",
        type=str,
        nargs="+",
        default=["tool"],
        help="训练时 marker_anchor.roles 的配置",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 1. 加载 tokenizer & model =====
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()

    # ===== 2. 构造和训练时一致的 dataset config =====
    marker_cfg = {
        "marker_anchor": {
            "enable": True,
            "token": args.marker_token,
            "loss_mask_value": 1,
            "roles": args.marker_roles,
        },
        "max_length": args.max_length,
        "multiturn": {"messages_key": "prompt", "tools_key": "tools"},
        "truncation": "error",
    }

    parquet_paths = [args.parquet_path]
    dataset = MarkerAnchoredMultiTurnSFTDataset(
        parquet_files=parquet_paths,
        tokenizer=tokenizer,
        config=marker_cfg,
    )

    print(f"Loaded dataset with {len(dataset)} samples from {args.parquet_path}")

    collator = DataCollatorForTokenClassification()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # ===== 3. 遍历所有样本，取出每个 marker 位置的打分 =====
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fout = open(args.output_path, "w", encoding="utf-8")

    global_index = 0  # 样本全局 idx
    # 聚合分数与标签（样本级）
    all_scores: list[float] = []
    all_labels: list[int] = []
    skipped_no_marker = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)         # (B, L)
            attention_mask = batch["attention_mask"].to(device)  # (B, L)
            loss_mask = batch["loss_mask"].to(device).float()  # (B, L)
            traj_labels = batch["trajectory_labels"].to(device).long()  # (B,)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, L, 1) —— 单 logit 回归到 0/1

            # ===== 样本级聚合：对 loss_mask>0 的位置做均值 =====
            mask = loss_mask.unsqueeze(-1)  # (B, L, 1)
            steps = mask.sum(dim=1).squeeze(-1)  # (B,)
            agg_sum = (logits.float() * mask).sum(dim=1).squeeze(-1)  # (B,)
            # 有些样本可能没有 marker（steps==0），做安全处理并统计跳过数量
            valid_mask = steps > 0
            if valid_mask.any():
                agg_mean = agg_sum[valid_mask] / steps[valid_mask]
                all_scores.extend(agg_mean.detach().cpu().tolist())
                all_labels.extend(traj_labels[valid_mask].detach().cpu().tolist())
            skipped_no_marker += int((~valid_mask).sum().item())

            B, L = input_ids.shape
            for b in range(B):
                seq_input_ids = input_ids[b]            # (L,)
                seq_attn = attention_mask[b]            # (L,)
                seq_loss_mask = loss_mask[b]            # (L,)
                seq_logits = logits[b]                  # (L, 1)

                # 有效长度（避免 decode 一堆 PAD）
                valid_len = int(seq_attn.sum().item())
                valid_ids = seq_input_ids[:valid_len].tolist()

                # 整个序列文本（可选）
                text = tokenizer.decode(valid_ids, skip_special_tokens=False)

                # marker 位置：loss_mask > 0 的地方
                marker_positions = (seq_loss_mask[:valid_len] > 0).nonzero(as_tuple=False)
                marker_positions = marker_positions.view(-1).tolist()

                markers = []
                for m_idx, pos in enumerate(marker_positions):
                    pos = int(pos)
                    token_id = int(seq_input_ids[pos].item())
                    # token 字面（注意 BPE，可能是片段）
                    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

                    score = float(seq_logits[pos, 0].item())  # 单 logit 分数（不做 sigmoid）

                    markers.append({
                        "marker_id": m_idx,      # 第几个 marker
                        "token_index": pos,      # 在序列中的 index
                        "token_id": token_id,
                        "token": token_str,
                        "logit": score,
                        "score": score,         # 保留同义 key，方便下游兼容
                    })

                record = {
                    "global_index": global_index,
                    "source_file": args.parquet_path,
                    "text": text,
                    "marker_count": len(markers),
                    "markers": markers,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_index += 1

    fout.close()
    print(f"Done. Wrote {global_index} samples to {args.output_path}")

    # ===== 4. 计算样本级聚合指标（与训练 compute_metrics 对齐）=====
    if len(all_scores) == 0:
        print("No valid samples for aggregation (no markers found). Metrics unavailable.")
        return

    scores = np.asarray(all_scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(all_labels, dtype=np.int64).reshape(-1)

    preds = (scores >= 0.5).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def safe_div(n: int, d: int) -> float:
        return float(n) / float(d) if d != 0 else 0.0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    # ROC AUC（基于秩，处理并列）
    def compute_auc(scores_arr: np.ndarray, labels_arr: np.ndarray) -> float | None:
        labels_arr = labels_arr.astype(np.int64)
        n_pos = int(labels_arr.sum())
        n_neg = int(labels_arr.shape[0] - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        order = np.argsort(scores_arr)
        ranks = np.empty_like(order, dtype=np.float64)
        n = scores_arr.shape[0]
        i = 0
        while i < n:
            j = i
            s_i = scores_arr[order[i]]
            while j + 1 < n and scores_arr[order[j + 1]] == s_i:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based average rank
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1
        sum_ranks_pos = ranks[labels_arr == 1].sum()
        auc_val = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc_val)

    auc = compute_auc(scores, labels)

    if auc is None:
        auc_str = "N/A (no positive or negative samples)"
    else:
        auc_str = f"{auc:.6f}"

    print(
        "Sample-level aggregated metrics (mean over marker logits):\n"
        f"  count: {len(scores)} (skipped_no_marker={skipped_no_marker})\n"
        f"  accuracy:  {accuracy:.6f}\n"
        f"  precision: {precision:.6f}\n"
        f"  recall:    {recall:.6f}\n"
        f"  f1:        {f1:.6f}\n"
        f"  auc:       {auc_str}"
    )


if __name__ == "__main__":
    main()
