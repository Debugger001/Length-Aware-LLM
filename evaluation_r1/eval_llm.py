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

import json
import time

import fire
import numpy as np
import vllm

from datasets import load_from_disk, load_dataset

# from mathruler.grader import extract_boxed_content, grade_answer
from utils.math_grader import boxed_reward_fn

import pandas as pd


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + "Please reason step by step, and put your final answer within \\boxed{}."
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"""
        + """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}. """
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

# def apply_training_template(question: str):
#     """
#     Training-aligned template: single user message.
#     Problem text (trimmed) + suffix appended exactly as in math.jinja.
#     """
#     return (
#         "<|im_start|>user\n"
#         + question
#         + "<|im_end|>\n<|im_start|>system\n You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
#         + "<|im_end|>\n<|im_start|>assistant\n"
#     )

def apply_training_template(question: str) -> str:
    """
    Training-aligned template (match validation): single user turn containing
    the problem text followed by the SAME suffix. System comes first.
    """
    q = question.strip()
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{q}\n\n"
        "You FIRST think about the reasoning process as an internal monologue enclosed in <think></think>, "
        "and THEN provide only the final answer. The final answer MUST be in \\boxed{}."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_l1_template(question: str):
    return (question + "\n\n Let's think step by step and output the final answer within \\box{{}}. Think for 1000 tokens.")


# def boxed_reward_fn(model_output: str, gt: str):
#     extracted_answer = extract_boxed_content(model_output)

#     if isinstance(gt, float) or isinstance(gt, int):
#         gt = str(gt)

#     if isinstance(gt, str):
#         return 1.0 if grade_answer(extracted_answer, gt) else 0.0
#     elif isinstance(gt, list):
#         is_correct = False
#         for gt_item in gt:
#             if isinstance(gt_item, float) or isinstance(gt_item, int):
#                 gt_item = str(gt_item)
#             is_correct |= grade_answer(extracted_answer, gt_item)
#         return 1.0 if is_correct else 0.0

#     return 0.0

hf_data_path = {
    "gpqa": "Idavidrein/gpqa",
    "gpqa_subset": "gpqa_diamond",
    "lsat": "dmayhem93/agieval-lsat-ar",
    "lsat_subset": None,
    "mmlu": "cais/mmlu",
    "mmlu_subset": "all"
}

def convert_dataset(task_name: str):
    """
    Convert a dataset from Hugging Face to the format expected by the evaluation script.
    """
    if task_name not in hf_data_path:
        raise ValueError(f"Task {task_name} not found in hf_data_path.")

    dataset = load_dataset(hf_data_path[task_name], name = hf_data_path[f"{task_name}_subset"])

    all_data = []
    if task_name == "gpqa":
        for i in range(len(dataset["train"])):
            correct_ans = dataset["train"][i]["Correct Answer"].strip()
            choices = [correct_ans]
            choices.append(dataset["train"][i]["Incorrect Answer 1"].strip())
            choices.append(dataset["train"][i]["Incorrect Answer 2"].strip())
            choices.append(dataset["train"][i]["Incorrect Answer 3"].strip())
            import random
            random.shuffle(choices)
            question = dataset['train'][i]['Question']
            question += f"\n\nOptions:\n"
            for i, choice in enumerate(choices):
                question += f"{chr(65 + i)}. {choice}\n"
            correct_choice = chr(65 + choices.index(correct_ans))
            all_data.append({
                "problem": question,
                "answer": correct_choice
            })
    elif task_name == "lsat":
        for i in range(len(dataset["test"])):
            question = dataset["test"][i]["query"]
            correct_choice = chr(65 + dataset["test"][i]["gold"][0])
            all_data.append({
                "problem": question,
                "answer": correct_choice
            })
    else: # task_name == "mmlu"
        for i in range(len(dataset['test'])):
            row = dataset['test'][i]
            options = row['choices']
            options_str  = ""
            for j in range(len(options)):
                options_str += f"{chr(65 + j)}. {options[j]}\n"
            # Add Options to the question
            question = row['question'] + "\n\nOptions:\n" + options_str
            correct_choice = chr(65 + row['answer'])
            all_data.append({
                "problem": question,
                "answer": correct_choice
            })

    return pd.DataFrame(all_data)

def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench", "gpqa", "mmlu", "lsat"],
    template: str = "training",
    dataset_name: str = "./datasets/evaluation_suite",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8000,
    max_model_len: int = 32768,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 16,
    max_test: int = 99999999,
    save: bool = True,
    tensor_parallel_size: int = 4,
    greedy: bool = True
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
        tensor_parallel_size=tensor_parallel_size,  # Added tensor parallelism config
    )

    print("Using template:", template)
    # math_reward_fn = boxed_reward_fn
    if template == "qwen_math":
        apply_template = apply_qwen_math_template

    elif template == "r1":
        apply_template = apply_r1_template

    elif template == "training":
        apply_template = apply_training_template

    elif template == "l1":
        apply_template = apply_l1_template

    else:
        raise ValueError

    results = {}
    avg_lens = {}
    max_lens = {}
    to_be_saved = []
        # --- counters for clip-ratio ---
    num_clipped = 0      # how many generations stopped by length
    total_outputs = 0    # total number of generations

    response_lengths = {t: [] for t in tasks}
    response_rewards = {t: [] for t in tasks}

    reward_fn = boxed_reward_fn

    local_datasets = load_from_disk(dataset_name)
    for task_name in tasks:
    # for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in local_datasets:
            if task_name not in hf_data_path:
                continue
            dataset = convert_dataset(task_name)
        else:
            dataset = local_datasets[task_name]

        prompts = dataset["problem"][:max_test]
        targets = dataset["answer"][:max_test]

        prompts = list(map(apply_template, prompts))
        print("inference for ", task_name)
        outputs = model.generate(prompts, sampling_params)
        batch_scores = []
        batch_lengths = []
        for k in range(len(outputs)):
            output = outputs[k]
            gt_repeated = [targets[k]] * sampling_params.n
            rewards = []
            for model_output, gt in zip([o.text for o in output.outputs], gt_repeated):
                r = reward_fn(model_output, gt)
                rewards.append(r)

            # --- record clip info precisely once per output ---
            for o in output.outputs:
                total_outputs += 1
                if getattr(o, "finish_reason", None) == "length":
                    num_clipped += 1
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": gt_repeated,
                    "model_output": [o.text for o in output.outputs],
                    "reward": [r for r in rewards],
                    "length":  [len(o.token_ids) for o in output.outputs],
                }
            )

        results[task_name] = np.mean(batch_scores)
        avg_lens[task_name] = np.mean(batch_lengths)
        max_lens[task_name] = np.max(batch_lengths)
        print("avg_rewards:",np.mean(batch_scores))
        print("avg_lens:", np.mean(batch_lengths))

    print(results)
    print("avg:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    clip_ratio = num_clipped / total_outputs if total_outputs else 0.0
    print(f"clip_ratio: {num_clipped}/{total_outputs}  ({clip_ratio:.2%})")


    if save:
        # Create directory if it doesn't exist
        import os
        os.makedirs("./logs_llm", exist_ok=True)

        # Create the filename
        fn = f"{model_name.replace('/', '_')}_maxtkn_{max_tokens}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        save_path = os.path.join("./logs_llm", fn)
        print(f"saving model outputs at {save_path}")

        # Convert NumPy types to Python native types
        results_dict = {k: float(v) for k, v in results.items()}
        avg_lens_dict = {k: float(v) for k, v in avg_lens.items()}
        max_lens_dict = {k: int(v) for k, v in max_lens.items()}

        # Create final data structure with Python native types
        summary_data = {
            "results": results_dict,
            "avg": float(np.mean(list(results.values()))),
            "avg_lens": avg_lens_dict,
            "max_lens": max_lens_dict,
            "clip_ratio": clip_ratio,
            "detailed_results": to_be_saved
        }

        # Save to file
        with open(save_path, "w") as f:
            json.dump(summary_data, f, indent=4)


fire.Fire(main)
