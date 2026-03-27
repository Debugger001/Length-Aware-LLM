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

"""Subprocess-based code execution and per-benchmark grading functions."""

import ast
import math
import os
import resource
import subprocess
import sys
import tempfile
from typing import Optional

from utils.code_extraction import extract_python_code


# ---------------------------------------------------------------------------
# Core execution engine
# ---------------------------------------------------------------------------

def _set_resource_limits(max_memory_mb: int, timeout: float) -> None:
    """Called in child process (preexec_fn) to set resource limits."""
    mem_bytes = max_memory_mb * 1024 * 1024
    # Address space limit
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    # CPU time limit (add 1s buffer over wall timeout)
    cpu_limit = int(timeout) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    # Max child processes (prevent fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def execute_code(
    code: str,
    stdin_input: Optional[str] = None,
    timeout: float = 10.0,
    max_memory_mb: int = 512,
) -> tuple[str, str, bool]:
    """Execute Python code in an isolated subprocess.

    Returns:
        (stdout, stderr, timed_out)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code)

        stdin_bytes = stdin_input.encode() if stdin_input is not None else None

        try:
            result = subprocess.run(
                [sys.executable, code_path],
                input=stdin_bytes,
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
                preexec_fn=lambda: _set_resource_limits(max_memory_mb, timeout),
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            return stdout, stderr, False
        except subprocess.TimeoutExpired:
            return "", "TimeoutExpired", True
        except Exception as e:
            return "", str(e), False


# ---------------------------------------------------------------------------
# HumanEval+ grading
# ---------------------------------------------------------------------------

def grade_humaneval(
    model_output: str,
    prompt: str,
    test_code: str,
    entry_point: str,
    timeout: float = 10.0,
    max_memory_mb: int = 512,
) -> float:
    """Grade a HumanEval+ problem.

    The prompt contains the function signature + docstring. The model is
    expected to complete the function. We concatenate prompt + extracted code
    + the test harness and execute.

    Returns 1.0 if all assertions pass, 0.0 otherwise.
    """
    extracted = extract_python_code(model_output)
    if not extracted:
        return 0.0

    # If the model repeated the function signature, deduplicate.
    # The prompt ends with the function def + docstring; the model's code
    # should start with the body. If extracted already starts with 'def ',
    # use it as-is (model wrote the full function). Otherwise prepend prompt.
    if extracted.lstrip().startswith("def ") or extracted.lstrip().startswith("class "):
        full_code = extracted
    else:
        # Model output is just the body — prepend prompt (signature)
        full_code = prompt + "\n" + extracted

    # Append the test harness
    full_code = full_code + "\n\n" + test_code + f"\ncheck({entry_point})\n"

    stdout, stderr, timed_out = execute_code(
        full_code, timeout=timeout, max_memory_mb=max_memory_mb
    )

    if timed_out:
        return 0.0
    # exit code 0 means all assertions passed
    # We infer success from absence of exception in stderr
    if stderr and ("Error" in stderr or "Traceback" in stderr):
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# LiveCodeBench grading
# ---------------------------------------------------------------------------

def _normalize_output(s: str) -> str:
    """Normalize whitespace in program output for comparison."""
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").rstrip().split("\n"))


def grade_lcb_stdin(
    model_output: str,
    test_cases: list[dict],
    timeout: float = 10.0,
    max_memory_mb: int = 512,
) -> float:
    """Grade a LiveCodeBench stdin/stdout problem.

    test_cases: list of {"input": str, "output": str}
    Returns 1.0 if all test cases pass, 0.0 otherwise.
    """
    extracted = extract_python_code(model_output)
    if not extracted:
        return 0.0

    for tc in test_cases:
        stdout, stderr, timed_out = execute_code(
            extracted,
            stdin_input=tc["input"],
            timeout=timeout,
            max_memory_mb=max_memory_mb,
        )
        if timed_out:
            return 0.0
        if _normalize_output(stdout) != _normalize_output(tc["output"]):
            return 0.0
    return 1.0


def grade_lcb_functional(
    model_output: str,
    starter_code: str,
    fn_name: str,
    test_cases: list[dict],
    timeout: float = 10.0,
    max_memory_mb: int = 512,
) -> float:
    """Grade a LiveCodeBench functional (LeetCode-style) problem.

    test_cases: list of {"input": str (Python literal), "output": str (Python literal)}
    Returns 1.0 if all test cases pass, 0.0 otherwise.
    """
    extracted = extract_python_code(model_output)
    if not extracted:
        return 0.0

    # Build the test harness
    test_lines = ["import sys", ""]

    # Include model code (which should contain the Solution class)
    test_lines.append(extracted)
    test_lines.append("")
    test_lines.append("_sol = Solution()")
    test_lines.append("_passed = True")

    for i, tc in enumerate(test_cases):
        try:
            inputs = ast.literal_eval(tc["input"])
            expected = ast.literal_eval(tc["output"])
        except Exception:
            # If we can't parse the test case, skip it
            continue

        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs_repr = repr(inputs)
        expected_repr = repr(expected)
        test_lines.append(
            f"_result_{i} = getattr(_sol, {repr(fn_name)})(*{inputs_repr})"
        )
        test_lines.append(
            f"assert _result_{i} == {expected_repr}, "
            f"f'Test {i} failed: {{_result_{i}}} != {expected_repr}'"
        )

    full_code = "\n".join(test_lines)

    stdout, stderr, timed_out = execute_code(
        full_code, timeout=timeout, max_memory_mb=max_memory_mb
    )

    if timed_out:
        return 0.0
    if stderr and ("Error" in stderr or "Traceback" in stderr):
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Codeforces grading
# ---------------------------------------------------------------------------

def grade_codeforces(
    model_output: str,
    official_tests: list[dict],
    generated_checker: Optional[str] = None,
    timeout: float = 10.0,
    max_memory_mb: int = 512,
) -> float:
    """Grade a Codeforces problem.

    official_tests: list of {"input": str, "output": str}
    generated_checker: optional Python checker script (string) for multi-answer problems
    Returns 1.0 if all test cases pass, 0.0 otherwise.
    """
    extracted = extract_python_code(model_output)
    if not extracted:
        return 0.0

    for tc in official_tests:
        stdin_input = tc["input"].replace("\r\n", "\n")
        expected = _normalize_output(tc["output"])

        stdout, stderr, timed_out = execute_code(
            extracted,
            stdin_input=stdin_input,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
        )

        if timed_out:
            return 0.0

        actual = _normalize_output(stdout)

        if generated_checker:
            # Use checker to validate (handles multi-answer problems)
            verdict = _run_checker(generated_checker, stdin_input, expected, actual, timeout)
            if not verdict:
                return 0.0
        else:
            if actual != expected:
                return 0.0

    return 1.0


def _run_checker(
    checker_code: str,
    problem_input: str,
    expected_output: str,
    actual_output: str,
    timeout: float,
) -> bool:
    """Run a Codeforces checker script.

    The checker receives input, expected, and actual output as command-line
    file paths. Returns True if checker accepts (prints 1 or 100).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checker_path = os.path.join(tmpdir, "checker.py")
        input_path = os.path.join(tmpdir, "input.txt")
        expected_path = os.path.join(tmpdir, "expected.txt")
        actual_path = os.path.join(tmpdir, "actual.txt")

        with open(checker_path, "w") as f:
            f.write(checker_code)
        with open(input_path, "w") as f:
            f.write(problem_input)
        with open(expected_path, "w") as f:
            f.write(expected_output)
        with open(actual_path, "w") as f:
            f.write(actual_output)

        try:
            result = subprocess.run(
                [sys.executable, checker_path, input_path, expected_path, actual_path],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            verdict = result.stdout.decode("utf-8", errors="replace").strip()
            return verdict in ("1", "100")
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Pass@k estimator
# ---------------------------------------------------------------------------

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
