#!/usr/bin/env bash
# Fully async variant of examples/sglang_multiturn/run_LLM_sync_DAPO_qwen3-4b_gaia_dev_agentcpm_mcp_multiturn.sh
# Run from repo root: bash recipe/fully_async_policy/shell/grpo_qwen3_4b_async_mcp.sh

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535
export TOKENIZERS_PARALLELISM=False
export VERL_ASSERT_ROLLOUT_METRICS=1
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export SWANLAB_WORKSPACE="AgentCPM_MCP"
export VERL_LOGGING_LEVEL=DEBUG
export OPENAI_API_KEY="sk-6y8kz2o3U0hG77KSQEto0s0GFWGprChx2tzO8DmL1TSfJlQ1"
export OPENAI_BASE_URL="https://api.moonshot.cn/v1"

PROJECT_DIR="$(pwd)"

# Algorithm / model args
temperature=0.6
top_p=0.95
top_k=20
model_path="/workspace/fanshengda/verl/mcp_agent_ckpts/Qwen3-4B-2507-correct-1114-128K/hf_global_step_4800"
loss_agg_mode="token-mean"
rollout_num=8
max_turns=50
context_warning_ratio=0.8
ppo_epochs=1
MAX_PROMPT_LENGTH=4000
MAX_RESPONSE_LENGTH=70000
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export VLLM_USE_V1=1
# Cluster layout (customize as needed)
TRAINER_NNODES=${TRAINER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
TRAINER_NGPUS_PER_NODE=${TRAINER_NGPUS_PER_NODE:-4}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-4}

train_batch_size=0
gen_prompt_bsz=1
ppo_mini_batch_size=8
staleness_threshold=${STALENESS_THRESHOLD:-1.0}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
require_batches=${REQUIRE_BATCHES:-1}
partial_rollout=${PARTIAL_ROLLOUT:-true}
data_train="/workspace/fanshengda/verl/input_data/webshaper/ziqin_LLMJudge_docker25_tongyi.parquet"
data_val="/workspace/fanshengda/verl/input_data/gaia_dev/dev_TextOnly_LLMJudge_docker25_tongyi.parquet"
tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml"
project_name='gaia_dev_async_rl'
experiment_name="qwen3-4b-fully-async-GRPO_1114SFT_ckpt4800-n${rollout_num}-webshaper-turn${max_turns}-ppo_epochs${ppo_epochs}"
default_local_dir="/workspace/fanshengda/verl/mcp_agent_ckpts/$experiment_name"
validation_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-validation"
rollout_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-train"
test_freq=300
total_epochs=15



# ==============================================================================
# Rollout Correction Configuration
# ==============================================================================

# Importance Sampling (IS) weights configuration
rollout_is="token"                        # "token", "sequence", or null to disable
rollout_is_threshold=2.0                  # Upper threshold for IS weights

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # "token", "sequence", "geometric", or null to disable
rollout_rs_threshold="null"               # RS upper threshold
rollout_rs_threshold_lower="null"         # RS lower threshold

# Veto mechanism (optional, independent of IS/RS)
rollout_token_veto_threshold="null"       # Per-token veto threshold (null to disable)


python3 -m recipe.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${data_train} \
    data.val_files=${data_val} \
    data.return_raw_chat=True \
    data.train_batch_size=${train_batch_size} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.nccl_timeout=64000 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=20000 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    +actor_rollout_ref.rollout.multi_turn.context_warning_ratio=${context_warning_ratio} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.over_sample_rate=0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    +algorithm.rollout_correction.rollout_is=${rollout_is} \
    +algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    +algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    +algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    +algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    +algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=${default_local_dir} \
    trainer.validation_data_dir=${validation_dir} \
    trainer.rollout_data_dir=${rollout_dir} \
    trainer.n_gpus_per_node=${TRAINER_NGPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.critic_warmup=0 \
    trainer.total_epochs=${total_epochs} \
    +trainer.rollout_metrics.aggregate_only=False \
    rollout.nnodes=${ROLLOUT_NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE} \
    rollout.total_rollout_steps=80000 \
    rollout.test_freq=${test_freq} \
    rollout.total_epochs=${total_epochs} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_rollout_log_probs=True \
    async_training.compute_prox_log_prob=True \
    +async_training.final_answer_tag="'</answer>'" \
    "$@"
