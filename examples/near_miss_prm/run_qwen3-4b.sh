
# n/examples/near_miss_prm/run_qwen2.5-7b.sh > ./near_miss_prm_logs/run_qwen25_7B_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535

export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export VERL_LOGGING_LEVEL=DEBUG
model_path="/workspace/models/Qwen/Qwen3-4B-Thinking-2507"
temperature=0.6
top_p=0.95
top_k=20 # 0 for HF rollout, -1 for vLLM rollout

experiment_name="qwen3-4B-2507-SearchR1-1108-DrGRPO"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/fanshengda/verl/input_data/near_miss_prm/train_1108.parquet \
    data.val_files=/workspace/fanshengda/verl/input_data/near_miss_prm/validation_1108.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=10000 \
    data.max_response_length=16000 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
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
    trainer.project_name='near_miss_prm' \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir="/workspace/fanshengda/verl/reward_model_ckpts/$experiment_name" \
    trainer.validation_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-validation" \
    trainer.rollout_data_dir="/workspace/fanshengda/verl/rollout_data/$experiment_name-train" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 $@