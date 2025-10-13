# run on 8xH100
# make sure your current working directory is the root of the project

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535

export VERL_ASSERT_ROLLOUT_METRICS=1
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export SWANLAB_WORKSPACE="AgentCPM_MCP"
export VERL_LOGGING_LEVEL=DEBUG
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Algorithm
temperature=0.6
top_p=0.95
top_k=20 # 0 for HF rollout, -1 for vLLM rollout
#model_path="/workspace/fanshengda/trl_sft/mcp_agent_ckpts/Qwen3-4B-ASearcher_0926/checkpoint-108"
model_path="/workspace/fanshengda/trl_sft/mcp_agent_ckpts/Qwen3-4B-ASearcher_1010/checkpoint-414"
# model_path="/workspace/models/Qwen/Qwen3-4B"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
experiment_name="qwen3-4b_PureTool-n8-agentcpm_SFT_ASearcher1010"
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gaia_dev_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=6000 \
    data.max_response_length=26000 \
    +data.max_model_len=32000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.nccl_timeout=8000 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='gaia_dev_async_rl' \
    trainer.experiment_name=$experiment_name \
    trainer.validation_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-validation" \
    trainer.rollout_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-train" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    +trainer.rollout_metrics.aggregate_only=False \
    data.train_files=/workspace/fanshengda/verl/input_data/asearcher/asearcher_1010.parquet \
    data.val_files=/workspace/fanshengda/verl/input_data/gaia_dev/dev_0929.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml" \
    trainer.total_epochs=15 $@