#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-0.6B  # replace it with your local file path
PROJECT_NAME=Length-LLM

export CUDA_VISIBLE_DEVICES=4,5,6,7
ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_0.6b_math_grpo \
    trainer.project_name=${PROJECT_NAME} \
    worker.actor.global_batch_size=64 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=40 \
    data.rollout_batch_size=128
