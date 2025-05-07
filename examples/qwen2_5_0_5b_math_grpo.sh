#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=test_qwen2_5_0_5b_math_grpo \
    trainer.n_gpus_per_node=2 \
    worker.rollout.n=3 \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    data.rollout_batch_size=128 \
    algorithm.penalty_cap=0.03 \
    algorithm.lambda_len_init=0.003 \
    algorithm.dual_lr=0.004 \
    algorithm.hit_cap_coef=100 \
    algorithm.threshold=380
