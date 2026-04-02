# (base) fanshengda@node1040:~/verl$ nohup bash ./PRM_from_ORM/run_math_process_judge_dapo.sh > "./PRM_from_ORM/logs/run_math_process_judge_dapo_$(date +"%Y%m%d_%H%M%S").log" 2>&1 < /dev/null &
#
set -x
ulimit -n 65535

export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"
export LOGLEVEL=DEBUG
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400


MODE="ORM"
#MODEL_PATH=/nfsdata/fanshengda/models/Qwen/Qwen3-4B-Instruct-2507
MODEL_PATH=/nfsdata/models/Qwen2.5-7B-Instruct
DATA_DIR=/nfsdata/fanshengda/verl/PRM_from_ORM/processed_math_process_judge_ICL
EXPERIMENT_NAME="Qwen2.5-7B-Instruct-PRM_from_${MODE}_MATH_DAPO_ICL"
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/scan_pro_train.parquet}
#VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/scan_pro_dev.parquet','${DATA_DIR}/processbench_eval.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/processbench_eval.parquet']"}



python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples=90000 \
    data.train_batch_size=32 \
    data.max_prompt_length=12000 \
    data.max_response_length=18192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.nccl_timeout=64000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward_model.reward_manager=naive \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=seq_reward \
    algorithm.filter_groups.max_num_gen_batches=-1 \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='NeurIPS_2026_PRM' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="/nfsdata/fanshengda/reward_model_ckpts/$EXPERIMENT_NAME" \
    trainer.validation_data_dir="/nfsdata/fanshengda/verl/rollout_data/$EXPERIMENT_NAME-validation" \
    trainer.rollout_data_dir="/nfsdata/fanshengda/verl/rollout_data/$EXPERIMENT_NAME-train" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 $@
