#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B-Instruct  # replace it with your local file path
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=agentica-org/DeepScaleR-Preview-Dataset \
    data.val_files=HuggingFaceH4/MATH-500@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=128 \
    data.max_response_length=2048 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=qwen2_5_1.5b_deepscaler_500_0816 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=25 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0 \
    algorithm.dual_lr=0.002 \
    algorithm.lambda_floor=0.00001 \
    algorithm.lambda_ceil=0.1 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=500 \
    data.seed=20250521 \
    worker.rollout.n=4
