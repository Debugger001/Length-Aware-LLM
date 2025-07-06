#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-1.7B  # replace it with your local file path
PROJECT_NAME=Length-LLM

CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_1.7b_math_grpo \
    trainer.project_name=${PROJECT_NAME} \
    worker.actor.global_batch_size=32 \
    trainer.save_freq=40 \
    data.rollout_batch_size=128
