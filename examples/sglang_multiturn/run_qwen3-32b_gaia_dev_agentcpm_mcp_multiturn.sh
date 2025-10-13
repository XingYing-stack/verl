# run on 8xH100
# make sure your current working directory is the root of the project

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535

export VERL_ASSERT_ROLLOUT_METRICS=1
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export SWANLAB_WORKSPACE="AgentCPM_MCP"

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
experiment_name="qwen3-32b_function_rm-gaia_dev-sgl-multi-w-tool-verify-n4-agentcpm"
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gaia_dev_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.model.path="/workspace/models/Qwen/Qwen3-32B" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    data.train_files=/workspace/fanshengda/verl/input_data/gaia_dev/dev_0928.parquet \
    data.val_files=/workspace/fanshengda/verl/input_data/gaia_dev/dev_0928.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml" \
    trainer.total_epochs=15 $@