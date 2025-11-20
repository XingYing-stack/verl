# conda activate /home/test/test03/miniconda3/envs/fsd

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    default_data_collator,
)
from trl import SFTTrainer, SFTConfig, RewardTrainer
from datasets import Dataset

from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
import warnings

warnings.filterwarnings("ignore")
import swanlab
from datetime import timedelta


if os.getenv('PYCHARM_HOSTED') != '1':
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=6))
    # Initialize the Accelerator
accelerator = Accelerator(mixed_precision='bf16')


# 只在主进程做一次副作用操作（登录/初始化/创建目录/写配置等）
if accelerator.is_main_process:
    os.environ["SWANLAB_API_KEY"] = "WoZrF9qolYJjzYBCfArih"     # <<< 填你的 key

# 你自己的文件：Qwen2ForProcessRewardModel 定义和权重加载逻辑
from near_miss_pair_PRM.modeling_qwen2_rm import Qwen2ForProcessRewardModel
# 你上面贴的 Dataset 基类
from verl.utils.dataset.marker_anchored_multiturn_sft_dataset import MarkerAnchoredMultiTurnSFTDataset

from transformers import AutoModelForTokenClassification

# ============================================================
# 1. 实现 Marker-Anchored Step Scoring 的核心函数
# ============================================================

def marker_anchored_sum_of_logits_bce(
    step_logits: torch.Tensor,
    marker_mask: torch.Tensor,
    traj_labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    对应伪代码中的：

      z_{i,t}: step-level logit at marker
      s_i    = sum_t z_{i,t}
      K_i    = number of markers
      Z_i    = alpha * s_i - gamma * log(K_i) + beta
      L      = BCEWithLogits(Z_i, y_i)

    参数:
      step_logits: (B, L)，比如取 logits[..., 1] 后得到的正类 logit
      marker_mask: (B, L) bool/0-1，1 表示该位置是 step-end marker
      traj_labels: (B,) 0/1 轨迹标签
    """
    # 保证类型
    marker_mask = marker_mask.to(torch.bool)
    step_logits = step_logits.float()
    traj_labels = traj_labels.float()

    # 只在 marker 位置上保留 logit，其余位置为 0
    masked_logits = step_logits * marker_mask  # (B, L)

    # s_i = sum_{t in T_i} z_{i,t}
    s = masked_logits.sum(dim=-1)  # (B,)

    # K_i = |T_i|，避免 log(0)，用 clamp(min=1)
    K = marker_mask.sum(dim=-1).clamp(min=1).to(step_logits.dtype)  # (B,)

    s = s / K

    # Z_i = alpha * s_i - gamma * log(K_i) + beta
    Z = alpha * s - gamma * torch.log(K) + beta  # (B,)

    Z = torch.clamp(Z, -30.0, 30.0)

    # 直接用 BCE with logits，等价于 sigmoid + BCE 但数值更稳定
    loss = F.binary_cross_entropy_with_logits(Z, traj_labels)

    return loss, Z


# ============================================================
# 3. 自定义 Trainer：在 compute_loss 里实现伪代码
# ============================================================

@dataclass
class MarkerPRMConfig(SFTConfig):
    """
    在 SFTConfig 上加上三个超参数：
      alpha, beta, gamma
    对应伪代码里的 (α, β, γ)

    如果你暂时不想学这些参数，而是把它们固定为常数，
    就用这里的值即可。
    """
    alpha: float = field(default=1.0)
    beta: float = field(default=0.0)
    gamma: float = field(default=0.0)


class MarkerAnchoredPRMTrainer(SFTTrainer):
    """
    继承 TRL 的 RewardTrainer，自定义 compute_loss：

    - model: Qwen2ForProcessRewardModel（输出 token-level logits [B, L, 2]）
    - inputs:
        input_ids, attention_mask, position_ids, loss_mask, trajectory_labels
    - 不把 labels 传给 model，避免触发内部的 token-level CE
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None, **kwargs):
        # 取出我们自定义的东西，注意要从 inputs 里 pop 掉，不然会传进 model.forward
        traj_labels: torch.Tensor = inputs.pop("trajectory_labels")  # (B,)
        loss_mask: torch.Tensor = inputs.pop("loss_mask")  # (B, L)，<extra_0> 位置为 1
        attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask", None)

        # 调用模型，只需要 logits，labels 不传，避免内部 CE
        outputs = model(**inputs, labels=None, return_dict=True)
        logits = outputs.logits  # (B, L, num_labels=2)，来自 Qwen2ForProcessRewardModel.score
        print('logits.shape:', logits.shape)
        # step-level scalar logit z_{i,t}
        # 这里用正类的 logit（index=1），也可以用 logit1-logit0 形成 log-odds
        step_logits = logits[..., 1]  # (B, L)

        # marker mask：只在 <extra_0> 且不为 padding 的位置上为 1
        marker_mask = (loss_mask > 0)

        print('marker_mask.sum(-1):', marker_mask.sum(-1))
        if attention_mask is not None:
            marker_mask = marker_mask & attention_mask.bool()

        # 按照伪代码做 sum-of-logits + length normalization + BCE
        loss, Z = marker_anchored_sum_of_logits_bce(
            step_logits=step_logits,
            marker_mask=marker_mask,
            traj_labels=traj_labels,
            alpha=self.args.alpha,
            beta=self.args.beta,
            gamma=self.args.gamma,
        )
        # ==========================
        # print('num_items_in_batch:', num_items_in_batch)
        # num_items = num_items_in_batch
        #
        # # 放到同一 device，避免分布式/模型并行下的设备不一致
        # if hasattr(num_items, "device") and num_items.device != logits.device:
        #     num_items = num_items.to(logits.device)
        # # 防御 0
        # num_items = torch.clamp(num_items, min=1)
        # loss = loss / num_items
        # loss = loss * self.accelerator.num_processes
        # ==========================
        if return_outputs:
            # 顺便把 trajectory-level logits 也塞回去，方便 eval 记录
            outputs["trajectory_logits"] = Z
            return loss, outputs

        return loss



from datasets import Dataset




# ============================================================
# 4. 训练脚本入口
# ============================================================

def main():
    # ------------ 基本配置 ------------
    # model_name = "/home/test/test12/models/Qwen/Qwen2.5-7B-Instruct"  # 或者你自己的 PRM base
    # output_dir = "/home/test/test12/fanshengda/verl/prm_ckpts"



    model_name = "/workspace/models/Qwen/Qwen2.5-1.5B-Instruct"  # 或者你自己的 PRM base
    output_dir = "/workspace/fanshengda/verl/prm_ckpts"

    # ------------ Tokenizer & Model ------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen 通常用 eos_token 作为 pad_token
        tokenizer.pad_token = tokenizer.eos_token

    # 加载官方的 Qwen2ForProcessRewardModel（来自 modeling_qwen2_rm.py）
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # ✅ 1. 增加新 token
    new_tokens = ["<extra_0>"]
    added = tokenizer.add_tokens(new_tokens)
    print(f"Added {added} new tokens")

    # ✅ 2. 同步模型 embedding
    model.resize_token_embeddings(len(tokenizer))



    # ------------ Dataset ------------
    parquet_paths = [
        "/home/test/test12/fanshengda/verl/input_data/near_miss_prm/anchor_train_1116.parquet"
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


    train_dataset = MarkerAnchoredMultiTurnSFTDataset(parquet_files=parquet_paths, tokenizer=tokenizer, config=marker_cfg)
    # ===== 优化器 & 调度器 =====
    num_train_epochs = 3
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4
    warmup_ratio = 0.03
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * world_size)
    total_training_steps = (steps_per_epoch * num_train_epochs) // gradient_accumulation_steps

    # 区分 backbone 和新 PRM 头
    base_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "score" in name:  # Qwen2ForProcessRewardModel.score 的 MLP 头
            head_params.append(param)
        else:
            base_params.append(param)

    optimizer = AdamW(
        [
            {"params": base_params, "lr": 5e-6, "weight_decay": 0.01},  # backbone 小一点
            {"params": head_params, "lr": 1e-4, "weight_decay": 0.01},  # 新头大一点
        ]
    )

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * warmup_ratio),
        num_training_steps=total_training_steps,
    )


    # ------------ TRL SFTConfig ------------
    training_args = MarkerPRMConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        bf16=True,  # 你的 GPU 支持的话
        max_grad_norm=1.0,
        # 很关键：保留我们自定义的字段（loss_mask, trajectory_labels）
        # 关闭 packing，因为我们自己已经在 Dataset 里 control 了长度
        packing=False,
        # 伪代码里的 (α, β, γ)
        alpha=1.0,
        beta=0.0,
        gamma=0.0,  # 设置为 0 等价于关闭 length norm
        remove_unused_columns=False,  # 保留 loss_mask / trajectory_labels 等自定义键
        max_length=None,  # 你已预处理，就别再让 SFTTrainer truncate
        dataset_kwargs={"skip_prepare_dataset": True}  # 跳过内部 prepare_dataset / map 流程
    )

    # ------------ Trainer ------------
    trainer = MarkerAnchoredPRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler),
        data_collator=default_data_collator,  # 数据已经是 tensor + pad 好的，直接用默认 collator
    )

    # ------------ Train ------------
    trainer.train()

    # 保存最终模型（包括更新后的 PRM 头参数）
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
