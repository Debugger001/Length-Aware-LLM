#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=test_qwen2_5_7b_math_grpo \
    trainer.n_gpus_per_node=4 \
    algorithm.hit_cap_coef=50 \
    algorithm.penalty_cap=0.1 \
    algorithm.lambda_len_init=0.003 \
    algorithm.dual_lr=0.004 \
    algorithm.hit_cap_coef=100 \
    algorithm.threshold=380
