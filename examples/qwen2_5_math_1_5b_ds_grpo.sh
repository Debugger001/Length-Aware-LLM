#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B-Instruct  # replace it with your local file path
CKPT_PATH=/data/cliu/Length-Aware-LLM/checkpoints/Length-LLM/qwen2_5_math_1.5b_ds_grpo_0815/global_step_100/
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
    data.max_response_length=2048 \
    worker.actor.global_batch_size=32 \
    trainer.experiment_name=qwen2_5_math_1.5b_ds_grpo_0920 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=25 \
    trainer.save_limit=1 \
    data.seed=20250521 \
    worker.rollout.n=4
