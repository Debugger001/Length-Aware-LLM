# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code benchmark evaluation script.

Evaluates LLMs on HumanEval+, LiveCodeBench, and Codeforces using vLLM
inference and execution-based grading. Mirrors eval_llm.py structure.

Usage:
    python evaluation_r1/eval_code.py \\
        --model_name path/to/model \\
        --tasks humaneval_plus livecodebench codeforces \\
        --greedy True \\
        --tensor_parallel_size 4
"""

import base64
import json
import os
import pickle
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import numpy as np
import vllm
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from utils.code_grader import (
    grade_codeforces,
    grade_humaneval,
    grade_lcb_functional,
    grade_lcb_stdin,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def apply_code_training_template(question: str) -> str:
    """Training-aligned template for code generation tasks.

    Mirrors math.jinja: <think> reasoning + code block answer.
    """
    q = question.strip()
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{q}\n\n"
        "You FIRST think about the reasoning process as an internal monologue enclosed in <think></think>, "
        "and THEN provide your solution as a single Python code block enclosed in ```python and ```."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_r1_template(question: str) -> str:
    """R1-style template for code generation tasks."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST be a single Python code block enclosed in ```python and ```.\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_humaneval_plus() -> list[dict]:
    """Load HumanEval+ problems from HuggingFace."""
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    problems = []
    for row in dataset:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
        })
    return problems


def _decompress_lcb_tests(compressed: str) -> list[dict]:
    """Decompress LiveCodeBench private test cases."""
    try:
        raw = base64.b64decode(compressed)
        raw = zlib.decompress(raw)
        obj = pickle.loads(raw)
        if isinstance(obj, str):
            obj = json.loads(obj)
        return obj
    except Exception:
        return []


def load_livecodebench(version: str = "all") -> list[dict]:
    """Load LiveCodeBench problems from HuggingFace.

    Each HF file contains only the *incremental* problems added in that version's
    time window (not cumulative). "all" concatenates all files (~1055 problems total).

    NOTE: LCB's official "release_vN" configs are cumulative (v1..vN), but here
    each version key selects only that file's incremental slice. Use "all" to
    replicate release_v6 (full benchmark).

    Version date ranges and problem counts (incremental per file):
        v1  test.jsonl   May 2023 – Mar 2024   ~400 problems
        v2  test2.jsonl  Mar 2024 – May 2024   ~111 problems
        v3  test3.jsonl  May 2024 – Jul 2024   ~101 problems
        v4  test4.jsonl  Jul 2024 – Sep 2024   ~101 problems
        v5  test5.jsonl  Sep 2024 – Jan 2025   ~167 problems
        v6  test6.jsonl  Jan 2025 – Apr 2025   ~175 problems

    Contamination note: for models trained through date X, use only versions
    whose window starts after X to avoid benchmark contamination.

    Args:
        version: "all" for all versions (recommended), "v1".."v6" for a single
                 incremental slice.
    """
    version_map = {
        "v1": "test.jsonl",
        "v2": "test2.jsonl",
        "v3": "test3.jsonl",
        "v4": "test4.jsonl",
        "v5": "test5.jsonl",
        "v6": "test6.jsonl",
    }

    if version == "all":
        files = list(version_map.values())
    elif version in version_map:
        files = [version_map[version]]
    else:
        raise ValueError(f"Unknown LCB version: {version!r}. Choose 'all' or 'v1'..'v6'.")

    problems = []
    for filename in files:
        try:
            local_path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=filename,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"Warning: could not download {filename}: {e}")
            continue

        with open(local_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                # Parse public test cases
                pub_tests = []
                if row.get("public_test_cases"):
                    try:
                        pub_tests = json.loads(row["public_test_cases"])
                    except Exception:
                        pass

                # Decompress private test cases
                priv_tests = []
                if row.get("private_test_cases"):
                    priv_tests = _decompress_lcb_tests(row["private_test_cases"])

                all_tests = pub_tests + priv_tests

                fn_name = None
                if row.get("metadata"):
                    try:
                        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
                        fn_name = meta.get("fn_name")
                    except Exception:
                        pass

                problems.append({
                    "question_id": row.get("question_id", ""),
                    "question_content": row.get("question_content", ""),
                    "starter_code": row.get("starter_code", ""),
                    "fn_name": fn_name,
                    "test_cases": all_tests,
                    "is_functional": bool(fn_name),
                    "platform": row.get("platform", ""),
                    "difficulty": row.get("difficulty", ""),
                })

    return problems


def load_codeforces(split: str = "test") -> list[dict]:
    """Load Codeforces problems from HuggingFace (open-r1/codeforces).

    Args:
        split: "test" (468 recent problems) or "train" (full ~8,760 problems).
    """
    dataset = load_dataset("open-r1/codeforces", name="verifiable", split=split)
    problems = []
    for row in dataset:
        problems.append({
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "input_format": row.get("input_format", ""),
            "output_format": row.get("output_format", ""),
            "examples": row.get("examples", []),
            "official_tests": row.get("official_tests", []),
            "generated_checker": row.get("generated_checker", None),
            "time_limit": row.get("time_limit", 2.0),
            "memory_limit": row.get("memory_limit", 256),
            "rating": row.get("rating", 0),
        })
    return problems


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_humaneval_prompt(problem: dict, apply_template) -> str:
    """HumanEval+ uses function completion: prompt IS the problem."""
    return apply_template(problem["prompt"])


def build_lcb_prompt(problem: dict, apply_template) -> str:
    """Build LiveCodeBench prompt."""
    question = problem["question_content"]
    if problem["is_functional"] and problem["starter_code"]:
        sc = problem["starter_code"].strip()
        question = question + f"\n\nComplete the following code:\n```python\n{sc}\n```"
    return apply_template(question)


def build_codeforces_prompt(problem: dict, apply_template) -> str:
    """Build Codeforces prompt from structured fields."""
    parts = [problem["title"], "", problem["description"]]
    if problem["input_format"]:
        parts.extend(["", "Input:", problem["input_format"]])
    if problem["output_format"]:
        parts.extend(["", "Output:", problem["output_format"]])
    if problem["examples"]:
        parts.append("\nExamples:")
        for i, ex in enumerate(problem["examples"]):
            parts.append(f"\nExample {i + 1}:")
            parts.append(f"Input:\n{ex['input']}")
            parts.append(f"Output:\n{ex['output']}")
    return apply_template("\n".join(parts))


# ---------------------------------------------------------------------------
# Grading wrappers (for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _grade_humaneval_task(args):
    model_output, prompt, test_code, entry_point, timeout, max_memory_mb = args
    return grade_humaneval(model_output, prompt, test_code, entry_point, timeout, max_memory_mb)


def _grade_lcb_task(args):
    model_output, problem, timeout, max_memory_mb = args
    test_cases = problem["test_cases"]
    if not test_cases:
        return 0.0
    if problem["is_functional"]:
        return grade_lcb_functional(
            model_output,
            problem["starter_code"],
            problem["fn_name"],
            test_cases,
            timeout,
            max_memory_mb,
        )
    else:
        return grade_lcb_stdin(model_output, test_cases, timeout, max_memory_mb)


def _grade_codeforces_task(args):
    model_output, problem, timeout, max_memory_mb = args
    return grade_codeforces(
        model_output,
        problem["official_tests"],
        problem.get("generated_checker"),
        timeout,
        max_memory_mb,
    )


def grade_parallel(grade_fn, args_list: list, max_workers: int = 16) -> list[float]:
    """Grade a list of tasks in parallel using ProcessPoolExecutor."""
    results = [0.0] * len(args_list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(grade_fn, args): i for i, args in enumerate(args_list)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Warning: grading failed for index {idx}: {e}")
                results[idx] = 0.0
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
    tasks: list = ["humaneval_plus", "livecodebench", "codeforces"],
    template: str = "training",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8000,
    max_model_len: int = 32768,
    n_samples: int = 16,
    max_test: int = 99999999,
    save: bool = True,
    tensor_parallel_size: int = 4,
    greedy: bool = True,
    exec_timeout: float = 10.0,
    exec_memory_mb: int = 512,
    lcb_version: str = "v6",
    grade_workers: int = 16,
):
    if greedy:
        n_samples = 1
        temperature = 0
        top_p = 1
        print("Using greedy samples to compute pass@1")
    else:
        print("Using multiple samples to compute pass@1")

    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=2,
        seed=int(time.time_ns()),
    )

    model = vllm.LLM(
        model_name,
        swap_space=32,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        tensor_parallel_size=tensor_parallel_size,
    )

    print("Using template:", template)
    if template == "training":
        apply_template = apply_code_training_template
    elif template == "r1":
        apply_template = apply_code_r1_template
    else:
        raise ValueError(f"Unknown template: {template!r}. Choose 'training' or 'r1'.")

    results = {}
    avg_lens = {}
    max_lens = {}
    to_be_saved = []
    num_clipped = 0
    total_outputs = 0

    for task_name in tasks:
        print(f"\n=== Loading dataset: {task_name} ===")

        # Load dataset and build prompts
        if task_name == "humaneval_plus":
            problems = load_humaneval_plus()[:max_test]
            prompts = [build_humaneval_prompt(p, apply_template) for p in problems]

        elif task_name == "livecodebench":
            problems = load_livecodebench(version=lcb_version)[:max_test]
            prompts = [build_lcb_prompt(p, apply_template) for p in problems]

        elif task_name in ("codeforces", "codeforces_full"):
            split = "test" if task_name == "codeforces" else "train"
            problems = load_codeforces(split=split)[:max_test]
            prompts = [build_codeforces_prompt(p, apply_template) for p in problems]

        else:
            print(f"Unknown task: {task_name}, skipping.")
            continue

        print(f"Loaded {len(problems)} problems. Running inference...")
        outputs = model.generate(prompts, sampling_params)

        # Build grading args (one entry per problem, contains all n_samples outputs)
        batch_scores = []
        batch_lengths = []

        # Collect all (problem_idx, sample_idx, model_output) tuples for parallel grading
        grade_args = []
        output_index = []  # maps flat index -> (problem_idx, sample_idx)

        if task_name == "humaneval_plus":
            task_grade_fn = _grade_humaneval_task
        elif task_name == "livecodebench":
            task_grade_fn = _grade_lcb_task
        else:
            task_grade_fn = _grade_codeforces_task

        for k, output in enumerate(outputs):
            for s_idx, sample in enumerate(output.outputs):
                if task_name == "humaneval_plus":
                    args = (
                        sample.text,
                        problems[k]["prompt"],
                        problems[k]["test"],
                        problems[k]["entry_point"],
                        exec_timeout,
                        exec_memory_mb,
                    )
                else:
                    args = (sample.text, problems[k], exec_timeout, exec_memory_mb)

                grade_args.append(args)
                output_index.append((k, s_idx))

        print(f"Grading {len(grade_args)} outputs with {grade_workers} workers...")
        flat_rewards = grade_parallel(task_grade_fn, grade_args, max_workers=grade_workers)

        # Reassemble flat rewards back into per-problem structure
        rewards_by_problem = [[] for _ in range(len(problems))]
        lengths_by_problem = [[] for _ in range(len(problems))]

        for flat_idx, (prob_idx, _) in enumerate(output_index):
            rewards_by_problem[prob_idx].append(flat_rewards[flat_idx])

        for k, output in enumerate(outputs):
            for o in output.outputs:
                total_outputs += 1
                if getattr(o, "finish_reason", None) == "length":
                    num_clipped += 1
            lengths_by_problem[k] = [len(o.token_ids) for o in output.outputs]
            batch_lengths.append(lengths_by_problem[k])
            batch_scores.append(np.mean(rewards_by_problem[k]))

            to_be_saved.append({
                "task_name": task_name,
                "prompt": outputs[k].prompt,
                "gt": None,
                "model_output": [o.text for o in outputs[k].outputs],
                "reward": rewards_by_problem[k],
                "length": lengths_by_problem[k],
            })

        results[task_name] = float(np.mean(batch_scores))
        avg_lens[task_name] = float(np.mean(batch_lengths))
        max_lens[task_name] = int(np.max(batch_lengths))
        print(f"avg_rewards: {results[task_name]:.4f}")
        print(f"avg_lens: {avg_lens[task_name]:.1f}")

    print("\n=== Results ===")
    print(results)
    print("avg:", float(np.mean(list(results.values()))) if results else 0.0)
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)

    clip_ratio = num_clipped / total_outputs if total_outputs else 0.0
    print(f"clip_ratio: {num_clipped}/{total_outputs}  ({clip_ratio:.2%})")

    if save:
        os.makedirs("./logs_code", exist_ok=True)
        fn = (
            f"{model_name.replace('/', '_')}"
            f"_maxtkn_{max_tokens}"
            f"_template_{template}"
            f"_temp{temperature}"
            f"_topp{top_p}"
            f"_n{n_samples}.json"
        )
        save_path = os.path.join("./logs_code", fn)
        print(f"Saving outputs to {save_path}")

        summary_data = {
            "results": results,
            "avg": float(np.mean(list(results.values()))) if results else 0.0,
            "avg_lens": avg_lens,
            "max_lens": max_lens,
            "clip_ratio": clip_ratio,
            "detailed_results": to_be_saved,
        }

        with open(save_path, "w") as f:
            json.dump(summary_data, f, indent=4)


fire.Fire(main)
