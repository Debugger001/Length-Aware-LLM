#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-32B  # replace it with your local file path
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=agentica-org/DeepScaleR-Preview-Dataset \
    data.val_files=HuggingFaceH4/MATH-500@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=128 \
    data.max_response_length=4096 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=qwen3_32b_deepscaler_2000_0804 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=25 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0 \
    algorithm.dual_lr=0.002 \
    algorithm.lambda_floor=0.00001 \
    algorithm.lambda_ceil=0.1 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=2000 \
    data.seed=20250521 \
    worker.rollout.n=4
