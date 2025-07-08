#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=Qwen/Qwen3-1.7B  # replace it with your local file path
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_1.7b_math_grpo \
    trainer.project_name=${PROJECT_NAME} \
    worker.actor.global_batch_size=64 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=40 \
    data.rollout_batch_size=128 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0.0003 \
    algorithm.dual_lr=0.0002 \
    algorithm.lambda_floor=0.0001 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=375 \
    worker.rollout.n=5 
