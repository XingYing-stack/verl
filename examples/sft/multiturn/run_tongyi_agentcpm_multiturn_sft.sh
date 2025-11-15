#!/bin/bash
set -x

export LOGLEVEL=DEBUG
ulimit -n 65535

export VERL_ASSERT_ROLLOUT_METRICS=1
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export SWANLAB_WORKSPACE="AgentCPM_MCP"
export VERL_LOGGING_LEVEL=DEBUG

nproc_per_node=8
experiment_name=Qwen3-4B-2507-tydpr_deepdive-1111-CPT
save_path=/workspace/fanshengda/verl/mcp_agent_ckpts/${experiment_name}
train_json=$(python - <<'PY'
import json

paths = [
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/deepdive_qa_rl_all_valid_messages.json",
    "/workspace/fanshengda/AgentCPM-MCP/sft_data/deepdive_qa_sft_all_valid_messages.json",
]

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
    data.max_length=65536 \
    data.train_batch_size=32 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr=1.5e-5 \
    data.truncation=error \
    +data.enable_thinking=true \
    data.micro_batch_size_per_gpu=1 \
    model.fsdp_config.model_dtype=bfloat16 \
    model.partial_pretrain=/workspace/fanshengda/verl/mcp_agent_ckpts/Qwen3-4B-2507-all_asearcher_MiroVerse_tydpr_deepdive-1106/hf_global_step_3000 \
    model.trust_remote_code=true \
    model.strategy=fsdp2 \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=agentcpm-sft \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=4 \
    trainer.logger='["console","swanlab"]' \
    use_remove_padding=true \
    trainer.test_freq=100 \
    trainer.save_freq=300 \
    +trainer.val_before_train=False \
    ulysses_sequence_parallel_size=4