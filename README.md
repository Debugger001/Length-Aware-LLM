# LACONIC: Length-Aware Constrained Reinforcement Learning for LLMs

[![Paper](https://img.shields.io/badge/arXiv-2602.14468-b31b1b.svg)](https://arxiv.org/abs/2602.14468)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/Debugger001/Length-Aware-LLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

Official implementation of **LACONIC**, a length-aware reinforcement learning method for large language models. LACONIC enforces a target token budget during RL training by combining task reward with an adaptive length cost, yielding shorter responses without requiring any inference-time modification.

This repository contains the code used for the paper and builds on top of EasyR1 / veRL. It includes training scripts, evaluation utilities, and checkpoint export tools for releasing LACONIC models.

## Quick Links

- [Paper](https://arxiv.org/abs/2602.14468)
- [Code](https://github.com/Debugger001/Length-Aware-LLM)
- Model checkpoints: coming soon
- Project page: coming soon

## Why LACONIC

- Enforces a target token budget directly during RL training.
- Preserves or improves task performance while reducing response length.
- Integrates into standard RL fine-tuning pipelines with minimal code changes.
- Requires no decoding tricks or post-processing at inference time.
- Supports reasoning, code, and function-calling style evaluation workflows.

## Key Claims From The Paper

According to the paper abstract, LACONIC:

- preserves or improves `pass@1` on mathematical reasoning benchmarks while reducing output length by over 50%
- maintains out-of-domain performance on general knowledge and multilingual benchmarks with 44% fewer tokens
- integrates into standard RL tuning with no inference changes and minimal deployment overhead

## Paper

**LACONIC: Length-Aware Constrained Reinforcement Learning for LLM**  
Chang Liu, Yiran Zhao, Lawrence Liu, Yaoqi Ye, Csaba Szepesvári, Lin F. Yang  
[arXiv:2602.14468](https://arxiv.org/abs/2602.14468)

## Overview

Reinforcement learning often improves reasoning quality at the cost of substantially longer outputs, which increases latency and serving cost. LACONIC addresses this by introducing a constrained RL objective that penalizes excessive response length relative to a target token budget. The penalty scale is adjusted adaptively during training, which makes the method more robust than fixed heuristic reward shaping.

## What Is In This Repo

The core implementation lives in the standard RL training path:

- [`verl/trainer/ray_trainer.py`](./verl/trainer/ray_trainer.py): applies the LACONIC length penalty and dual update during training.
- [`verl/trainer/config.py`](./verl/trainer/config.py): defines length-control hyperparameters such as `threshold`, `dual_lr`, `penalty_cap`, and `hit_cap`.
- [`examples/config.yaml`](./examples/config.yaml): base experiment config.
- [`examples/`](./examples): launch scripts for different model families and target budgets.
- [`evaluation_r1/eval_llm.py`](./evaluation_r1/eval_llm.py): reasoning benchmark evaluation.
- [`evaluation_r1/eval_code.py`](./evaluation_r1/eval_code.py): code benchmark evaluation.
- [`evaluation_r1/eval_bfcl.py`](./evaluation_r1/eval_bfcl.py): BFCL evaluation helper.
- [`scripts/model_merger.py`](./scripts/model_merger.py): merges FSDP checkpoints and optionally uploads to the Hugging Face Hub.

## Method Overview

LACONIC adds a learnable length penalty to RL fine-tuning. At a high level, training proceeds as follows:

1. Generate rollouts.
2. Compute task reward.
3. Measure how much each response exceeds a target token budget.
4. Penalize over-budget responses with a dual variable `lambda`.
5. Update `lambda` online so the model stays close to the desired average output length.

The main public hyperparameters are:

- `algorithm.threshold`: target response budget.
- `algorithm.lambda_len_init`: initial dual variable.
- `algorithm.dual_lr`: dual update step size.
- `algorithm.penalty_cap`: upper bound on the per-sample length penalty.
- `algorithm.hit_cap`: extra penalty for responses that hit `max_response_length`.
- `algorithm.lambda_floor`, `algorithm.lambda_ceil`: clamp range for the dual variable.

## Release Snapshot

This repository is being prepared for a cleaner public release. The `LACONIC` branch is the main public branch for the project. The codebase still inherits some naming and package structure from the underlying EasyR1 / veRL framework, but the active LACONIC implementation is in the top-level training and evaluation code paths listed above.

For reproducibility, use the top-level repository code rather than the nested `evaluation_r1/EasyR1/` snapshot.

## Installation

### Recommended

```bash
git clone https://github.com/Debugger001/Length-Aware-LLM.git
cd Length-Aware-LLM
git checkout LACONIC

conda create -n laconic python=3.10 -y
conda activate laconic

pip install --upgrade pip
pip install -e .
```

### Dependencies and Notes

- Python `>=3.9`
- `transformers>=4.51.0,<4.53.0`
- `vllm>=0.8.0`
- `flash-attn>=2.4.3`
- `ray[default]`

The exact CUDA / PyTorch / FlashAttention / vLLM combination matters. If `pip install -e .` is not sufficient on your machine, install PyTorch, FlashAttention, and vLLM first using versions compatible with your driver and CUDA runtime, then re-run:

```bash
pip install -e .
```

If you plan to export or upload checkpoints to Hugging Face, also install:

```bash
pip install huggingface_hub
```

## Getting Started

### 1. Launch Training

Example: DeepScaleR-1.5B-Preview with a target budget of 1500 tokens.

```bash
bash examples/deepscale_1_5b_preview_deepscale.sh
```

Example: DeepSeek-R1-Distill-Qwen-1.5B.

```bash
bash examples/deepseek_1_5b_ds.sh
```

Example: Qwen2.5-1.5B-Instruct on math.

```bash
bash examples/qwen2_5_1_5b_math_grpo.sh
```

These launchers override the base config in [`examples/config.yaml`](./examples/config.yaml) and set model-specific values such as:

- `worker.actor.model.model_path`
- `algorithm.threshold`
- `algorithm.dual_lr`
- `algorithm.lambda_ceil`
- `trainer.n_gpus_per_node`
- `worker.rollout.n`

### 2. Merge The Checkpoint Into Hugging Face Format

After training, merge an FSDP actor checkpoint:

```bash
python scripts/model_merger.py \
  --local_dir checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor
```

This creates:

```text
checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface/
```

### 3. Evaluate The Exported Model

Math / reasoning evaluation:

```bash
python evaluation_r1/eval_llm.py \
  --model_name checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface \
  --tasks '["aime","amc","math","minerva","olympiad_bench"]' \
  --template training \
  --tensor_parallel_size 4 \
  --greedy True
```

Code evaluation:

```bash
python evaluation_r1/eval_code.py \
  --model_name checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface \
  --tasks '["humaneval_plus","livecodebench","codeforces"]' \
  --tensor_parallel_size 4 \
  --greedy True
```

BFCL evaluation:

```bash
python evaluation_r1/eval_bfcl.py \
  --model_paths '["checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface"]' \
  --model_names '["<release_name>"]' \
  --test_categories '["all"]' \
  --backend vllm \
  --num_gpus 4
```

## Reproducing The Main Pipeline

The minimal publication workflow is:

```bash
git checkout LACONIC
pip install -e .

# train
bash examples/deepscale_1_5b_preview_deepscale.sh

# merge actor shards
python scripts/model_merger.py \
  --local_dir checkpoints/Length-LLM/<experiment_name>/global_step_<best_step>/actor

# evaluate
python evaluation_r1/eval_llm.py \
  --model_name checkpoints/Length-LLM/<experiment_name>/global_step_<best_step>/actor/huggingface \
  --tasks '["aime","amc","math","minerva","olympiad_bench"]' \
  --template training \
  --tensor_parallel_size 4 \
  --greedy True
```

### Notes On The Current Codebase

- The Python package name is still `verl`.
- Some scripts and directory names still reference `EasyR1` or earlier experiment names.
- There is a nested `evaluation_r1/EasyR1/` snapshot that appears to be a baseline copy rather than the active training code.

## Data Format

The default config expects fields such as:

- `problem`
- `answer`
- `images`
- `videos`

See [`examples/config.yaml`](./examples/config.yaml) for the active keys and prompt formatting options.

Reward functions are defined in:

- [`examples/reward_function/math.py`](./examples/reward_function/math.py)
- [`examples/reward_function/r1v.py`](./examples/reward_function/r1v.py)
- [`examples/reward_function/dapo.py`](./examples/reward_function/dapo.py)

Prompt templates are defined in:

- [`examples/format_prompt/math.jinja`](./examples/format_prompt/math.jinja)
- [`examples/format_prompt/r1v.jinja`](./examples/format_prompt/r1v.jinja)
- [`examples/format_prompt/dapo.jinja`](./examples/format_prompt/dapo.jinja)

## Exporting And Uploading A Model To Hugging Face

There are two ways to upload a trained checkpoint.

### Option A: Merge And Upload In One Step

```bash
python scripts/model_merger.py \
  --local_dir checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor \
  --hf_upload_path <hf_user_or_org>/<repo_name>
```

`model_merger.py` uses `huggingface_hub.HfApi.create_repo()` and `upload_folder()` internally.

### Option B: Upload An Existing `huggingface/` Folder

If the merged folder already exists:

```bash
hf auth login
hf upload-large-folder <hf_user_or_org>/<repo_name> \
  checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface \
  --repo-type model
```

For smaller uploads, this also works:

```bash
hf upload <hf_user_or_org>/<repo_name> \
  checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface \
  . \
  --repo-type model
```

### Suggested Model Card Fields

When you publish checkpoints, include:

- base model
- training dataset
- target token budget
- training script
- best checkpoint step
- evaluation command
- evaluation results
- license and intended use
- limitations and failure modes

## Planned Model Releases

The first public checkpoints will likely include variants such as:

| Model | Base Model | Budget | Status |
| --- | --- | --- | --- |
| `LACONIC-DeepScaleR-1.5B-1500` | `agentica-org/DeepScaleR-1.5B-Preview` | 1500 | Planned |
| `LACONIC-DeepSeek-R1-Distill-Qwen-1.5B-1500` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1500 | Planned |
| `LACONIC-Qwen2.5-1.5B-550` | `Qwen/Qwen2.5-1.5B-Instruct` | 550 | Planned |

## Results

Final benchmark tables will be added here as the public release is finalized. A good public-facing version of this section should include both performance and efficiency so readers can immediately see the tradeoff LACONIC achieves.

Suggested subsections for the final table set:

- math reasoning
- out-of-domain general reasoning
- multilingual evaluation
- code and function-calling evaluation
- length reduction statistics

## Figures

Figures will be added here in the polished release version of the repository.

Recommended assets:

- method overview
- reward + length tradeoff figure
- response length comparison plot

## Citation

```bibtex
@misc{liu2026laconic,
  title        = {LACONIC: Length-Aware Constrained Reinforcement Learning for LLM},
  author       = {Chang Liu and Yiran Zhao and Lawrence Liu and Yaoqi Ye and Csaba Szepesv{\'a}ri and Lin F. Yang},
  year         = {2026},
  eprint       = {2602.14468},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2602.14468}
}
```

If this repository remains a derivative of EasyR1 / veRL in the public release, it is also appropriate to acknowledge the upstream framework.

## Acknowledgments

This project builds on EasyR1 and veRL. We thank the upstream authors for releasing the training framework that made this work possible.
