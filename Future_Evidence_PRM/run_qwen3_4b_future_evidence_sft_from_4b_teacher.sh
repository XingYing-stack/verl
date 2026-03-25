#!/usr/bin/env bash

set -x
export LOGLEVEL=DEBUG
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VERL_LOGGING_LEVEL=DEBUG
export SWANLAB_API_KEY="WoZrF9qolYJjzYBCfArih"

MODEL_PATH="/nfsdata/fanshengda/models/Qwen/Qwen3-4B-Instruct-2507"
TRAIN_JSONL="/nfsdata/fanshengda/verl/input_data/future_evidence_prm/Qwen3-4B-Instruct-2507_hotpotqa_future_evidence_sft_train_v2.jsonl"
EVAL_JSONL="/nfsdata/fanshengda/verl/input_data/future_evidence_prm/Qwen3-4B-Instruct-2507_hotpotqa_future_evidence_sft_validation_v2.jsonl"
OUTPUT_DIR="/nfsdata/fanshengda/verl/prm_ckpts/qwen3_4b_teacher_future_evidence_sft"

accelerate launch --num_processes 4 --main_process_port 29510 near_miss_pair_PRM/train_future_evidence_sft.py \
    --train_jsonl "${TRAIN_JSONL}" \
    --eval_jsonl "${EVAL_JSONL}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --max_length 24000 \
    --logging_steps 1 \
    --save_steps 200 \
    --eval_steps 10 \
    "$@"
