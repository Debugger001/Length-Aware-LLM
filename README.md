# LACONIC: Length-Aware Constrained Reinforcement Learning for LLMs

[![Paper](https://img.shields.io/badge/arXiv-2602.14468-b31b1b.svg)](https://arxiv.org/abs/2602.14468)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/Debugger001/Length-Aware-LLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

Official implementation of **LACONIC**, a reinforcement learning method that teaches LLMs to respect a target token budget during training. Instead of only rewarding task success, LACONIC also charges a length-based cost when generations become unnecessarily long, and adjusts that cost automatically over time.

This lets models become **shorter, cheaper, and faster at inference** without changing the decoding pipeline at deployment time.

This codebase is **adapted from [EasyR1](https://github.com/hiyouga/EasyR1)**, an RL training framework based on veRL. LACONIC should be viewed as a lightweight length-aware extension built on top of that training stack, rather than a brand new framework from scratch.

Just as importantly, LACONIC is **extremely easy to implement and deploy**:

- **implementation:** add one extra cost term to the reward and update one scalar dual variable
- **training:** no architecture change, no extra model, no new data format
- **deployment:** use the trained model exactly as usual, with no inference-time control logic

## Why This Is Interesting

Reinforcement learning can improve reasoning performance, but it often makes models much more verbose. That extra verbosity increases latency and serving cost, and it is hard to control reliably with fixed heuristic penalties.

LACONIC addresses that problem directly during RL training:

- the model is rewarded for solving the task
- the model is penalized when it exceeds a target token budget
- the penalty strength is adjusted adaptively instead of being fixed by hand

In one sentence: **LACONIC makes unnecessary output tokens expensive during RL training.**

## Headline Results

According to the paper abstract, LACONIC:

| Setting | Main Outcome |
| --- | --- |
| Mathematical reasoning | Preserves or improves `pass@1` while reducing output length by **over 50%** |
| General knowledge + multilingual | Maintains out-of-domain performance with **44% fewer tokens** |
| Deployment | Requires **no inference-time changes** and adds minimal serving overhead |

Paper link: [arXiv:2602.14468](https://arxiv.org/abs/2602.14468)

## How LACONIC Works

The idea can be summarized in three lines:

$$
\text{objective} = r_{\text{task}} - \lambda \, c_{\text{len}}
$$

$$
c_{\text{len}} = \max\left(0, \frac{L-B}{B}\right)
$$

$$
\bar L > B \Rightarrow \lambda \text{ increases}, \qquad
\bar L < B \Rightarrow \lambda \text{ decreases}
$$

That is the whole method:

1. compute the usual task reward
2. subtract an extra cost only when the output is too long
3. adjust one scalar, $\lambda$, so the average response length stays near the target budget

where:

- $r_{\text{task}}$ is the usual task reward
- $L$ is the response length
- $B$ is the target token budget
- $c_{\text{len}}$ is the over-length cost
- $\lambda$ controls how expensive extra length is

The interpretation is simple:

- if a response stays within budget, the length cost is zero
- if a response goes over budget, it pays an extra cost
- if the model keeps being too long on average, LACONIC raises $\lambda$
- if the model is already short enough, LACONIC relaxes $\lambda$

So LACONIC is just **standard RL plus an adaptive cost on overlong outputs**.

## Why It Is Easy To Implement And Deploy

LACONIC is intentionally lightweight.

- **No model changes:** the policy architecture stays the same.
- **No inference changes:** once training is done, decoding is unchanged.
- **No auxiliary model:** there is no extra predictor or controller at serving time.
- **Minimal training change:** conceptually, it is one extra reward term and one scalar update rule.
- **Small configuration surface:** the main knobs are just the target budget `B` and a few scalar hyperparameters such as `dual_lr`, `penalty_cap`, and `hit_cap`.

If you already have an RL fine-tuning pipeline, LACONIC is closer to a **small reward modification** than to a new training stack.

<p align="center">
  <img src="./assets/laconic_overview.png" alt="LACONIC overview" width="88%">
</p>
<p align="center"><em>Policy updates optimize task reward minus a length-aware cost, while the dual update adjusts the strength of that cost to keep generations near the desired token budget.</em></p>

## Why It Matters

- Shorter outputs reduce inference latency.
- Shorter outputs reduce serving cost.
- Training-time length control is easier to deploy than brittle decoding-time heuristics.
- The method integrates naturally into standard RL fine-tuning workflows.

## Quick Start

### Installation

```bash
git clone https://github.com/Debugger001/Length-Aware-LLM.git
cd Length-Aware-LLM
git checkout LACONIC

conda create -n laconic python=3.10 -y
conda activate laconic

pip install --upgrade pip
pip install -e .
```

Main dependencies:

- Python `>=3.9`
- `transformers>=4.51.0,<4.53.0`
- `vllm>=0.8.0`
- `flash-attn>=2.4.3`
- `ray[default]`

If you plan to merge or upload checkpoints to Hugging Face, also install:

```bash
pip install huggingface_hub
```

### Run Training

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

These launchers override the base configuration in [`examples/config.yaml`](./examples/config.yaml), including:

- `worker.actor.model.model_path`
- `algorithm.threshold`
- `algorithm.dual_lr`
- `algorithm.lambda_ceil`
- `trainer.n_gpus_per_node`
- `worker.rollout.n`

### Merge A Checkpoint To Hugging Face Format

```bash
python scripts/model_merger.py \
  --local_dir checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor
```

This creates:

```text
checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface/
```

### Evaluate A Merged Model

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

## Repository Guide

The main LACONIC implementation lives in:

- [`verl/trainer/ray_trainer.py`](./verl/trainer/ray_trainer.py): LACONIC reward adjustment and dual update in the RL loop.
- [`verl/trainer/config.py`](./verl/trainer/config.py): length-control hyperparameters.
- [`examples/config.yaml`](./examples/config.yaml): base training config.
- [`examples/`](./examples): experiment launch scripts.
- [`evaluation_r1/eval_llm.py`](./evaluation_r1/eval_llm.py): reasoning benchmarks.
- [`evaluation_r1/eval_code.py`](./evaluation_r1/eval_code.py): code benchmarks.
- [`evaluation_r1/eval_bfcl.py`](./evaluation_r1/eval_bfcl.py): BFCL helper.
- [`scripts/model_merger.py`](./scripts/model_merger.py): FSDP checkpoint merger and optional HF upload.

Upstream base framework:

- [EasyR1](https://github.com/hiyouga/EasyR1): the training framework this repository is adapted from

For reproducibility, use the **top-level repository code**. The nested `evaluation_r1/EasyR1/` directory is an inherited snapshot and is not the active implementation path.

## Data And Prompting

The default config expects dataset fields such as:

- `problem`
- `answer`
- `images`
- `videos`

Prompt templates:

- [`examples/format_prompt/math.jinja`](./examples/format_prompt/math.jinja)
- [`examples/format_prompt/r1v.jinja`](./examples/format_prompt/r1v.jinja)
- [`examples/format_prompt/dapo.jinja`](./examples/format_prompt/dapo.jinja)

Reward functions:

- [`examples/reward_function/math.py`](./examples/reward_function/math.py)
- [`examples/reward_function/r1v.py`](./examples/reward_function/r1v.py)
- [`examples/reward_function/dapo.py`](./examples/reward_function/dapo.py)

## Planned Model Releases

The first public checkpoints will likely include variants such as:

| Model | Base Model | Budget | Status |
| --- | --- | --- | --- |
| `LACONIC-DeepScaleR-1.5B-1500` | `agentica-org/DeepScaleR-1.5B-Preview` | 1500 | Planned |
| `LACONIC-DeepSeek-R1-Distill-Qwen-1.5B-1500` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1500 | Planned |
| `LACONIC-Qwen2.5-1.5B-550` | `Qwen/Qwen2.5-1.5B-Instruct` | 550 | Planned |

Model checkpoints and model cards will be added here as the public release is finalized.

## Uploading Models To Hugging Face

Two supported paths:

### Merge And Upload In One Step

```bash
python scripts/model_merger.py \
  --local_dir checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor \
  --hf_upload_path <hf_user_or_org>/<repo_name>
```

### Upload An Existing `huggingface/` Folder

```bash
hf auth login
hf upload-large-folder <hf_user_or_org>/<repo_name> \
  checkpoints/Length-LLM/<experiment_name>/global_step_<step>/actor/huggingface \
  --repo-type model
```

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

## Acknowledgments

This project is adapted from [EasyR1](https://github.com/hiyouga/EasyR1), which itself builds on veRL. We thank the EasyR1 and veRL authors for releasing the training framework that made this work possible.
