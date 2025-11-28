# eval_marker_prm.py
import os
import json
import argparse
from typing import Dict, Any, List

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
        default="/nfsdata/fanshengda/verl/prm_ckpts/sum_ce_0.1entropy/checkpoint-3000",
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
        default='/nfsdata/fanshengda/verl/near_miss_pair_PRM/output/prm_anchor_validation_1128_ckpt3500.jsonl',
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

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)         # (B, L)
            attention_mask = batch["attention_mask"].to(device)  # (B, L)
            loss_mask = batch["loss_mask"].to(device)         # (B, L)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, L, num_labels=2)
            probs = torch.softmax(logits, dim=-1)  # (B, L, 2)

            B, L = input_ids.shape
            for b in range(B):
                seq_input_ids = input_ids[b]            # (L,)
                seq_attn = attention_mask[b]            # (L,)
                seq_loss_mask = loss_mask[b]            # (L,)
                seq_logits = logits[b]                  # (L, 2)
                seq_probs = probs[b]                    # (L, 2)

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

                    logit_0 = float(seq_logits[pos, 0].item())
                    logit_1 = float(seq_logits[pos, 1].item())
                    prob_0 = float(seq_probs[pos, 0].item())
                    prob_1 = float(seq_probs[pos, 1].item())

                    markers.append(
                        {
                            "marker_id": m_idx,      # 第几个 marker
                            "token_index": pos,      # 在序列中的 index
                            "token_id": token_id,
                            "token": token_str,
                            "logits": [logit_0, logit_1],
                            "probs": [prob_0, prob_1],
                            "score_pos": prob_1,     # 通常我们关心 label=1 的概率
                        }
                    )

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


if __name__ == "__main__":
    main()
