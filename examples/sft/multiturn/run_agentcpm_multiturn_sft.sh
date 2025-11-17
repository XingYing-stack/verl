#!/bin/bash
set -x

export LOGLEVEL=DEBUG
ulimit -n 65535

export VERL_ASSERT_ROLLOUT_METRICS=1
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export SWANLAB_WORKSPACE="AgentCPM_MCP"
export VERL_LOGGING_LEVEL=DEBUG

nproc_per_node=8
experiment_name=Qwen3-4B-2507-deepseek-1116-filter_empty
save_path=/workspace/fanshengda/verl/mcp_agent_ckpts/${experiment_name}
train_json=$(python - <<'PY'
import json



deepseek_paths = [
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_0926.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_0930.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1003.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1004.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1006.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1015.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1020.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1021.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_1022.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/deepdive_1031.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/ASearcher_force_answer_1031.json",
]


tongyi_ASearcher_paths = [
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1106.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1107.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1108.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1109.json"
]


yishan_all_paths = [
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-mirovoyage_hard_10k_all_messages.json',
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-deepdive_qa_sft_all_messages.json',
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-infoseek_hard_13k_all_messages.json',
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-deepdive_qa_rl_all_messages.json',
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-mirovoyage_filted_top4000_all_messages.json',
    '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-webwalker_silver_13k_all_messages.json'
]


yishan_correct_paths = [
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-mirovoyage_filted_top4000_all_messages.json',
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-deepdive_qa_sft_all_messages.json',
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-infoseek_hard_13k_all_messages.json',
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-mirovoyage_hard_10k_all_messages.json',
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-deepdive_qa_rl_all_messages.json',
  '/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1114-correct-webwalker_silver_13k_all_messages.json'
]
paths = deepseek_paths

print(json.dumps(paths))
PY
)

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.custom_cls.path=pkg://verl.utils.dataset.agentcpm_multiturn_sft_dataset \
    data.custom_cls.name=AgentCPMMultiTurnSFTDataset \
    data.train_files="$train_json" \
    data.val_files="[/workspace/fanshengda/AgentCPM-MCP/sft_data/tongyi-ds-1109.json]" \
    data.multiturn.enable=true \
    data.max_length=128000 \
    data.train_batch_size=32 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr=2e-5 \
    data.truncation=error \
    +data.enable_thinking=true \
    +data.multiturn.filter_empty=true \
    data.micro_batch_size_per_gpu=1 \
    model.fsdp_config.model_dtype=bfloat16 \
    model.partial_pretrain=/workspace/models/Qwen/Qwen3-4B-Thinking-2507-keep-empty-think \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=agentcpm-sft \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=3 \
    trainer.logger='["console","swanlab"]' \
    use_remove_padding=true \
    trainer.test_freq=100 \
    trainer.save_freq=300 \
    +trainer.val_before_train=False \
    ulysses_sequence_parallel_size=8