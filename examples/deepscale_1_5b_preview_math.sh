#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=agentica-org/DeepScaleR-1.5B-Preview  # replace it with your local file path
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=agentica-org/DeepScaleR-Preview-Dataset \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=64 \
    trainer.experiment_name=deepscale_1.5b_math \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=40 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0.0003 \
    algorithm.dual_lr=0.0002 \
    algorithm.lambda_floor=0.0001 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=325 \
    data.seed=20250521 \
    worker.rollout.n=5 