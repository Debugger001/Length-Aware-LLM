set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_simple.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=test_qwen2_5_0_5b_math_grpo \
    trainer.n_gpus_per_node=2
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
