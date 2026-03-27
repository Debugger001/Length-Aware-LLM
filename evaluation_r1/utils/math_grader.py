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

"""Math answer grading utilities for eval_llm.py."""

from mathruler.grader import extract_boxed_content, grade_answer


def boxed_reward_fn(model_output: str, gt) -> float:
    """Extract boxed answer and grade against ground truth.

    Args:
        model_output: Raw model output string.
        gt: Ground truth answer (str, int, float, or list of those).

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    extracted_answer = extract_boxed_content(model_output)

    if isinstance(gt, (float, int)):
        gt = str(gt)

    if isinstance(gt, str):
        return 1.0 if grade_answer(extracted_answer, gt) else 0.0
    elif isinstance(gt, list):
        is_correct = False
        for gt_item in gt:
            if isinstance(gt_item, (float, int)):
                gt_item = str(gt_item)
            is_correct |= grade_answer(extracted_answer, gt_item)
        return 1.0 if is_correct else 0.0

    return 0.0
