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
model_path="/workspace/fanshengda/trl_sft/mcp_agent_ckpts/Qwen3-4B-2507-ASearcher_context_1023/checkpoint-263"
#model_path="/workspace/models/Qwen/Qwen3-4B-Thinking-2507"
PROJECT_DIR="$(pwd)"
loss_agg_mode="token-mean"

rollout_num=8
max_turns=30
ppo_epochs=1
MAX_PROMPT_LENGTH=6000
MAX_RESPONSE_LENGTH=34000

CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
experiment_name="qwen3-4b-async-DAPO_ppo_epochs${ppo_epochs}_1023SFT_ckpt263_PureTool-n${rollout_num}-gaia_dev-turn${max_turns}"
python3 -m recipe.dapo.main_dapo \
    --config-path="$CONFIG_PATH" \
    --config-name='gaia_dev_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.gen_batch_size=2 \
    data.val_batch_size=8 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    +data.max_model_len=40000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.nccl_timeout=8000 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=dapo \
    +reward_model.overlong_buffer.enable=False \
    +reward_model.overlong_buffer.len=4096 \
    +reward_model.overlong_buffer.penalty_factor=1.0 \
    +algorithm.filter_groups.enable=true \
    +algorithm.filter_groups.metric=seq_reward \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='gaia_dev_async_rl' \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir="/workspace/fanshengda/verl/mcp_agent_ckpts/$experiment_name" \
    trainer.validation_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-validation" \
    trainer.rollout_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-train" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    +trainer.rollout_metrics.aggregate_only=False \
    data.train_files=/workspace/fanshengda/verl/input_data/ARPO-RL-DeepSearch-1K/arpo_1029.parquet \
    data.val_files=/workspace/fanshengda/verl/input_data/gaia_dev/dev_1029.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml" \
    trainer.total_epochs=15 $@
