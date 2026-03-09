
# nohup bash /nfsdata/fanshengda/verl/examples/near_miss_prm/run_qwen3-4b-group.sh > ./near_miss_prm_logs/run_qwen3_4B_GROUP_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535

export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export VERL_LOGGING_LEVEL=DEBUG
model_path="/nfsdata/fanshengda/models/Qwen/Qwen3-4B-Thinking-2507"
temperature=0.6
top_p=0.95
top_k=20 # 0 for HF rollout, -1 for vLLM rollout

MODE="outcome"
GROUP_SIZE="2"

experiment_name="qwen3-4B-2507-PRM_from_${MODE}_GROUP${GROUP_SIZE}"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nfsdata/fanshengda/verl/input_data/PRM_from_outcome/${MODE}_train_group_near_K5_T${GROUP_SIZE}.parquet \
    data.val_files=/nfsdata/fanshengda/verl/input_data/PRM_from_outcome/${MODE}_dev_group_near_K5_T${GROUP_SIZE}.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=48000 \
    data.max_response_length=8000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.nccl_timeout=8000 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='NeurIPS_2026_PRM' \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir="/nfsdata/fanshengda/reward_model_ckpts/$experiment_name" \
    trainer.validation_data_dir="/nfsdata/fanshengda/verl/rollout_data/$experiment_name-validation" \
    trainer.rollout_data_dir="/nfsdata/fanshengda/verl/rollout_data/$experiment_name-train" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=50 $@