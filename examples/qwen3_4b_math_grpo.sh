#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path
PROJECT_NAME=Length-LLM

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_math_grpo \
    trainer.project_name=${PROJECT_NAME}
