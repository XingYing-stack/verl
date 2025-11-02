#!/bin/bash
set -x

nproc_per_node=8
experiment_name=Qwen3-4B-2507-ALL_ASearcher_DeepDive_1102
save_path=/workspace/fanshengda/verl/mcp_agent_ckpts/${experiment_name}
train_json=$(python - <<'PY'
import json

paths = [
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
]

print(json.dumps(paths))
PY
)

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.custom_cls.path=pkg://verl.utils.dataset.agentcpm_multiturn_sft_dataset \
    data.custom_cls.name=AgentCPMMultiTurnSFTDataset \
    data.train_files="$train_json" \
    data.val_files='[]' \
    data.multiturn.enable=true \
    data.max_length=64000 \
    optim.lr=2e-5 \
    data.truncation=error \
    data.enable_thinking=True \
    data.micro_batch_size_per_gpu=1 \
    model.fsdp_config.model_dtype=bfloat16 \
    model.partial_pretrain=/workspace/models/Qwen/Qwen3-4B-Thinking-2507-keep-empty-think \
    model.trust_remote_code=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=agentcpm-sft \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=3 \
    trainer.logger='[console,swanlab]' \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=2
