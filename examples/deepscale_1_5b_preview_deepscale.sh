#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=agentica-org/DeepScaleR-1.5B-Preview  # replace it with your local file path
CKPT_PATH=/data/cliu/Length-Aware-LLM/checkpoints/Length-LLM/deepscaler_1.5b_deepscaler_2000_nf_0901/global_step_150
PROJECT_NAME=Length-LLM

ray stop --force
ray start --head

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=agentica-org/DeepScaleR-Preview-Dataset \
    data.val_files=HuggingFaceH4/MATH-500@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.load_checkpoint_path=${CKPT_PATH} \
    data.rollout_batch_size=128 \
    data.max_response_length=4096 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=deepscaler_1.5b_deepscaler_1000_nf_0902 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=25 \
    trainer.save_limit=4 \
    algorithm.penalty_cap=0.4 \
    algorithm.lambda_len_init=0.00000 \
    algorithm.dual_lr=0.002 \
    algorithm.lambda_floor=0.00000 \
    algorithm.lambda_ceil=0.1 \
    algorithm.hit_cap=0.1 \
    algorithm.threshold=1000 \
    data.seed=20250521 \
    worker.rollout.n=4
