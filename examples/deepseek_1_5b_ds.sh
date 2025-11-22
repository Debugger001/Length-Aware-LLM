#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # replace it with your local file path
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
    trainer.experiment_name=deepseek_1.5b_deepscaler_grpo_base_1122 \
    trainer.project_name=Length-LLM \
    trainer.n_gpus_per_node=3 \
    trainer.save_freq=25 \
    data.seed=20250521 \
    worker.rollout.n=2
