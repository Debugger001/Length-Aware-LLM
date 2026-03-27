import glob
import json
import os
import subprocess
from pathlib import Path

import fire
import numpy as np


DEFAULT_TEST_CATEGORIES = ["all"]


def _run_cmd(cmd, env=None):
    print("\n[Running]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _is_hf_model_id(model_path: str) -> bool:
    """
    Accept strings like:
      agentica-org/DeepScaleR-1.5B-Preview
    but not:
      /abs/path/to/model
      ./relative/path
      ../relative/path
    """
    if not isinstance(model_path, str) or len(model_path.strip()) == 0:
        return False

    if model_path.startswith("/") or model_path.startswith("./") or model_path.startswith("../"):
        return False

    parts = model_path.split("/")
    return len(parts) == 2 and all(len(p) > 0 for p in parts)


def _validate_model_path(model_path: str):
    is_local_dir = os.path.isdir(model_path)
    is_hf_id = _is_hf_model_id(model_path)

    if not is_local_dir and not is_hf_id:
        raise ValueError(
            f"model_path must be either a local directory or a Hugging Face repo id, got: {model_path}"
        )


def _find_score_files(score_root: Path, model_name: str, test_categories: list[str]):
    """
    BFCL usually writes files like:
      score/<model_name>/*_<category>_score.json
    """
    model_score_dir = score_root / model_name
    found = {}

    if not model_score_dir.exists():
        return found

    for cat in test_categories:
        pattern = str(model_score_dir / f"*_{cat}_score.json")
        matches = sorted(glob.glob(pattern))
        if matches:
            found[cat] = matches[-1]

    # Special case: when test_categories includes "all" or "all_scoring",
    # BFCL may emit many per-category files rather than one literal "*_all_score.json".
    if "all" in test_categories or "all_scoring" in test_categories:
        all_matches = sorted(glob.glob(str(model_score_dir / "*_score.json")))
        for path in all_matches:
            name = os.path.basename(path)
            found[name] = path

    return found


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _extract_scalar_scores(score_json):
    """
    Robust extraction of obvious scalar metrics from BFCL score jsons.
    """
    out = {}

    if not isinstance(score_json, dict):
        return out

    for k, v in score_json.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v

    for key in ["summary", "metrics", "result", "scores"]:
        if key in score_json and isinstance(score_json[key], dict):
            for k, v in score_json[key].items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    out[f"{key}.{k}"] = v

    return out


def main(
    model_paths: list[str],
    model_names: list[str] = None,
    test_categories: list[str] = DEFAULT_TEST_CATEGORIES,
    backend: str = "vllm",
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    project_root: str = "./evaluation_r1/bfcl_runs",
    result_json: str = "./evaluation_r1/bfcl_runs/summary.json",
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    extra_generate_args: list[str] = None,
):
    """
    Example:
    python evaluation_r1/eval_bfcl.py \
      --model_paths '["agentica-org/DeepScaleR-1.5B-Preview"]' \
      --model_names '["DeepScaleR-1.5B-Preview"]' \
      --test_categories '["all"]' \
      --backend vllm \
      --num_gpus 4 \
      --gpu_memory_utilization 0.9
    """

    if model_names is None:
        model_names = [Path(p).name if os.path.isdir(p) else p.replace("/", "__") for p in model_paths]

    assert len(model_paths) == len(model_names), "model_paths and model_names must match in length"

    os.makedirs(project_root, exist_ok=True)

    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = os.path.abspath(project_root)

    summary = {
        "project_root": env["BFCL_PROJECT_ROOT"],
        "backend": backend,
        "num_gpus": num_gpus,
        "gpu_memory_utilization": gpu_memory_utilization,
        "test_categories": test_categories,
        "models": {},
    }

    test_category_arg = ",".join(test_categories)

    for model_name, model_path in zip(model_names, model_paths):
        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name}")
        print(f"Model source: {model_path}")
        print("=" * 80)

        _validate_model_path(model_path)

        if not skip_generation:
            gen_cmd = [
                "bfcl",
                "generate",
                "--model",
                model_name,
                "--test-category",
                test_category_arg,
                "--backend",
                backend,
                "--num-gpus",
                str(num_gpus),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--local-model-path",
                model_path,
            ]
            if extra_generate_args:
                gen_cmd.extend(extra_generate_args)
            _run_cmd(gen_cmd, env=env)

        if not skip_evaluation:
            eval_cmd = [
                "bfcl",
                "evaluate",
                "--model",
                model_name,
                "--test-category",
                test_category_arg,
            ]
            _run_cmd(eval_cmd, env=env)

        score_root = Path(env["BFCL_PROJECT_ROOT"]) / "score"
        score_files = _find_score_files(score_root, model_name, test_categories)

        model_summary = {
            "model_path": model_path,
            "score_files": score_files,
            "per_category": {},
        }

        numeric_vals = []

        for cat, score_file in score_files.items():
            score_json = _read_json(score_file)
            extracted = _extract_scalar_scores(score_json)

            model_summary["per_category"][cat] = {
                "score_file": score_file,
                "extracted": extracted,
            }

            for k, v in extracted.items():
                if isinstance(v, (int, float)):
                    lk = k.lower()
                    if any(tok in lk for tok in ["acc", "accuracy", "score", "overall", "ast"]):
                        numeric_vals.append(float(v))

        model_summary["mean_extracted_metric"] = float(np.mean(numeric_vals)) if numeric_vals else None
        summary["models"][model_name] = model_summary

        print(f"\nFinished {model_name}")
        print("Found score files:")
        for cat, path in score_files.items():
            print(f"  {cat}: {path}")
        print("Mean extracted metric:", model_summary["mean_extracted_metric"])

    os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
    with open(result_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {result_json}")

    print("\n=== Compact Summary ===")
    for model_name, model_summary in summary["models"].items():
        print(f"{model_name}: mean_extracted_metric={model_summary['mean_extracted_metric']}")
        for cat, entry in model_summary["per_category"].items():
            print(f"  - {cat}: {entry['score_file']}")


if __name__ == "__main__":
    fire.Fire(main)