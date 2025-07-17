#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=agentica-org/DeepScaleR-1.5B-Preview  # replace it with your local file path
# CKPT_PATH=/home/cliu/Length-Aware-LLM/checkpoints/Length-LLM/deepscale_1.5b_math/global_step_160/
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=agentica-org/DeepScaleR-Preview-Dataset \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.rollout_batch_size=128 \
    data.max_response_length=2048 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=deepscale_1.5b_deepscaler \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=25 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0 \
    algorithm.dual_lr=0.0002 \
    algorithm.lambda_floor=0.0001 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=500 \
    data.seed=20250521 \
    worker.rollout.n=4
