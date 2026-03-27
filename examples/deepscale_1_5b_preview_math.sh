#!/bin/bash

set -x

source /home/lliu/miniconda3/etc/profile.d/conda.sh
conda activate LACONIC

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=agentica-org/DeepScaleR-1.5B-Preview  # replace it with your local file path
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=HuggingFaceH4/MATH-500@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.load_checkpoint_path=${CKPT_PATH} \
    data.rollout_batch_size=128 \
    data.max_response_length=4096 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=deepscale_1.5b_math_500_0727 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=25 \
    trainer.save_limit=4 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0 \
    algorithm.dual_lr=0.0001 \
    algorithm.lambda_floor=0.0001 \
    algorithm.lambda_ceil=0.02 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=500 \
    data.seed=20250521 \
    worker.rollout.n=4 
