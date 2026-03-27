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

"""Utilities for extracting executable Python code from LLM outputs."""

import re


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    # Handle properly closed tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle unclosed <think> tag: strip from <think> to end of string
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text


def extract_code_block(text: str) -> str:
    """Extract the last Python code block from text.

    Looks for ```python ... ``` first, then ``` ... ```.
    If no fenced block found, returns the raw text (stripped).
    """
    # Try ```python ... ``` blocks first
    python_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()

    # Fall back to generic ``` ... ``` blocks
    generic_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    return text.strip()


def extract_python_code(model_output: str) -> str:
    """Main entry point: strip think tags, extract code block, strip whitespace."""
    text = strip_think_tags(model_output)
    return extract_code_block(text)
