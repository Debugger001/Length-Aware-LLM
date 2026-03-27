# LACONIC Codebase Guide

**Paper**: [arxiv.org/abs/2602.14468](https://arxiv.org/abs/2602.14468)
**Method**: Lagrangian-based length regularization for RL-trained reasoning LLMs.
**Base framework**: EasyR1 / veRL (v0.3.2.dev0) — multi-modal RL training with Ray + vLLM + FSDP.

---

## Environment Setup

`conda activate LACONIC` alone won't work if conda isn't initialized in the shell.
Use this two-step command:

```bash
source /home/lliu/miniconda3/etc/profile.d/conda.sh && conda activate LACONIC
```

This activates Python 3.12.13 with `verl`, `vllm`, and `ray` pre-installed.
Run this before any training, evaluation, or script execution.

This has already been added to the identified main entry point scripts:
- `examples/deepscale_1_5b_preview_deepscale.sh`
- `examples/deepscale_1_5b_preview_math.sh`
- `examples/qwen3_32b_deepscaler.sh`

**Any other shell scripts or new entry points you create will need these two lines added manually** near the top (after `set -x`, before any `export` or `ray` calls).

---

## Quick Orientation

### Entry Points

| Step | Command | What it does |
|------|---------|-------------|
| **Train** | `bash examples/deepscale_1_5b_preview_deepscale.sh` | Launches `python3 -m verl.trainer.main config=examples/config.yaml` with overrides |
| **Merge** | `python scripts/model_merger.py --local_dir <ckpt>` | Merges FSDP shards into HuggingFace format |
| **Eval** | `python evaluation_r1/eval_llm.py --model_name <path> ...` | vLLM-based pass@1 evaluation on math/reasoning benchmarks |

### Directory Layout

```
verl/                              # The RL framework (installed as `verl` package)
  trainer/
    main.py                        # Training entry point
    ray_trainer.py                 # RayPPOTrainer — main loop + LACONIC penalty (lines 654-717)
    core_algos.py                  # GRPO, REINFORCE++, GAE, RLOO, ReMax, PPO loss
    config.py                      # All hyperparameters (AlgorithmConfig lines 105-112 = LACONIC params)
  workers/
    actor/dp_actor.py              # Policy gradient updates
    critic/dp_critic.py            # Value function (only used with GAE)
    rollout/vllm_rollout_spmd.py   # vLLM-based generation
    reward/function.py             # Reward manager (batch or sequential mode)
  utils/
    dataset.py                     # RLHFDataset, Jinja template application
    torch_functional.py            # log_probs, masking, padding utilities
  protocol.py                      # DataProto — central data container

examples/
  config.yaml                      # Default training config (overridden by shell scripts)
  *.sh                             # Training launch scripts for various model/dataset combos
  reward_function/
    math.py                        # boxed{} grading: 10% format + 90% accuracy
    r1v.py                         # <answer> tag grading: 50% format + 50% accuracy
    dapo.py                        # +/-1.0 accuracy + soft overlong punishment
  format_prompt/
    math.jinja                     # <think>...</think> + \boxed{} prompt template
    r1v.jinja                      # <think> + <answer> template
    dapo.jinja                     # "Answer: $Answer" template

evaluation_r1/
  eval_llm.py                      # Eval script (AIME, AMC, MATH, Minerva, Olympiad, GPQA, MMLU, LSAT)
  utils/math_grader.py             # boxed_reward_fn used at eval time
  EasyR1/                          # Full copy of original EasyR1 (baseline reference)

scripts/
  model_merger.py                  # FSDP checkpoint -> HuggingFace merger
```

---

## The LACONIC Algorithm

The paper's core contribution is ~60 lines in `verl/trainer/ray_trainer.py` (lines 654-717).

### What it does

After computing task reward but **before** advantage estimation, it:

1. **Computes relative excess length**: `rel_excess = max(0, (L - threshold) / threshold)`
2. **Scales by Lagrangian multiplier**: `penalty = lambda * rel_excess`
3. **Caps the penalty**: `penalty = min(penalty, penalty_cap)`
4. **Adds hit-cap penalty**: `penalty += hit_cap` if response hit `max_response_length`
5. **Applies at last valid token only**: `reward[last_token] -= penalty`
6. **Updates dual variable**: `lambda <- clip(lambda + dual_lr * (mean(L/threshold) - 1), floor, ceil)`

### Hyperparameters (in `AlgorithmConfig`, `config.py:105-112`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_len_init` | 0.0 | Initial Lagrangian multiplier |
| `dual_lr` | 3.5e-3 | Learning rate for dual (lambda) updates |
| `threshold` | 380 | Target token budget |
| `penalty_cap` | 4.0e-2 | Maximum penalty per sample |
| `hit_cap` | 1.0e-2 | Extra penalty for responses that hit the max length |
| `lambda_floor` | 3.0e-4 | Minimum lambda |
| `lambda_ceil` | 2.0e-3 | Maximum lambda |

Shell scripts override these heavily. Larger models use higher thresholds and ceilings.

### Tracked Metrics (wandb)

- `len/lambda_len` — current Lagrangian multiplier
- `len/length_ratio` — mean(actual_length / threshold), target is 1.0
- `len/avg_len`, `len/avg_len_ema` — response length statistics
- `len/max_penalty` — largest penalty in the batch

---

## Training Pipeline (inside `RayPPOTrainer.fit()`)

```
1. Generate rollouts (vLLM)
2. Compute rewards (via reward_function/*.py)
3. Apply LACONIC length penalty + dual lambda update     <-- paper's contribution
4. Apply KL penalty to rewards (optional, depends on disable_kl)
5. Compute advantages (GRPO / REINFORCE++ / GAE / RLOO / ReMax)
6. Update critic (if using GAE)
7. Update actor (PPO loss with dual-clip)
8. Log to wandb, save checkpoints
```

---

## Supported Algorithms

Set via `algorithm.adv_estimator`:

| Algorithm | Needs Critic | Key Property |
|-----------|-------------|--------------|
| `grpo` | No | Group-normalized advantages, needs rollout.n > 1 |
| `reinforce_plus_plus` | No | Discounted return-to-go |
| `gae` | Yes | Generalized Advantage Estimation |
| `rloo` | No | Leave-one-out baseline |
| `remax` | No | Simple reward-minus-baseline |

All algorithms can be combined with the LACONIC length penalty.

---

## Evaluation

`evaluation_r1/eval_llm.py` uses `fire.Fire(main)` with these key args:

```python
main(
    model_name="path/to/model",
    tasks=["aime", "amc", "math", "minerva", "olympiad_bench", "gpqa", "mmlu", "lsat"],
    template="training",          # or "r1", "qwen_math", "l1"
    dataset_name="./datasets/evaluation_suite",
    temperature=0.6, top_p=0.95,
    max_tokens=8000,
    max_model_len=32768,
    n_samples=16,
    tensor_parallel_size=4,
    greedy=True,                  # overrides: n=1, temp=0, top_p=1
)
```

**Important**: The `template="training"` option in eval constructs the prompt differently
from the Jinja templates used during training (`math.jinja`). The training template in
eval uses `apply_training_template()` which appends the suffix inline. Compare carefully
if results seem off.

Outputs are saved to `./logs_llm/` as JSON with per-sample rewards, lengths, and clip ratio.

---

## Git Branches (algorithm exploration history)

| Branch | What |
|--------|------|
| `hit-cap-penalty` | Main/default branch |
| `LACONIC` | The LACONIC paper implementation |
| `LACONIC-code` | **Current branch** — extends LACONIC from math to code-based evaluations |
| `PID-Lagrangian` | PID controller for lambda |
| `aug-Lag` | Augmented Lagrangian variant |
| `clipped-penalty` | Clipped penalty |
| `linear-penalty` | Linear penalty |
| `grpo-base` | Baseline (no length penalty) |

---

## Known Gotchas

1. **`evaluation_r1/EasyR1/`** is a full nested copy of the original EasyR1 repo. The actual
   training code lives in the top-level `verl/`, NOT in this nested copy.
2. **Template mismatch risk**: Training uses `examples/format_prompt/math.jinja`, but eval
   has its own `apply_training_template()` function. They should produce the same prompt
   format but verify if eval results seem wrong.
3. **`model_merger.py`** expects checkpoint dir to NOT end with "huggingface" — it creates
   a `huggingface/` subdirectory inside the checkpoint dir.
4. **No documentation** existed before this file — the README describes the base EasyR1
   framework, not the LACONIC modifications.
