"""
Dataset adapters for ENAMEL and other code evaluation datasets.

Provides a unified interface through the adapter pattern.
"""

from __future__ import annotations

import ast
import re
import textwrap
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

# Common imports injected at the top of every generated solution
IMPORT_PKG = """from typing import *
from bisect import *
from collections import *
from copy import *
from heapq import *
from math import *
from itertools import *
from functools import *
import string, re, math, random, itertools, functools
"""


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @property
    @abstractmethod
    def dataset_name(self) -> str: ...

    @abstractmethod
    def load_dataset(self, path: Optional[str] = None) -> List[Dict]: ...

    @abstractmethod
    def get_prompt_template(self) -> str: ...

    @abstractmethod
    def format_prompt(self, task: Dict, prompt_template: str) -> str: ...

    @abstractmethod
    def extract_solution(self, task: Dict, completion: str) -> str: ...


# ---------------------------------------------------------------------------
# ENAMEL adapter
# ---------------------------------------------------------------------------

_ENAMEL_PROMPT_TEMPLATE = "{prompt}"


class ENAMELAdapter(DatasetAdapter):
    """Adapter for the ENAMEL dataset (CSV format).

    The ENAMEL CSV has columns: task_id, prompt, entry_point, ...
    The 'prompt' column already contains the function signature + docstring.
    """

    dataset_name = "enamel"

    def __init__(self, csv_path: str = "dataset/enamel.csv") -> None:
        self.csv_path = csv_path

    def load_dataset(self, path: Optional[str] = None) -> List[Dict]:
        p = path or self.csv_path
        df = pd.read_csv(p)
        records = []
        for idx, row in df.iterrows():
            records.append(
                {
                    "problem_id": int(idx),
                    "task_id": str(row["task_id"]),
                    "entry_point": str(row["entry_point"]),
                    "prompt": str(row["prompt"]),
                }
            )
        return records

    def get_prompt_template(self) -> str:
        return _ENAMEL_PROMPT_TEMPLATE

    def format_prompt(self, task: Dict, prompt_template: str) -> str:
        return prompt_template.format(prompt=task["prompt"])

    @staticmethod
    def _extract_code_block(completion: str) -> str:
        """Extract the last fenced Python block, fallback to raw text."""
        matches = re.findall(r"```python\s*(.*?)```", completion, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"```\s*(.*?)```", completion, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return completion.strip()

    @staticmethod
    def _trim_after_target_function(code: str, entry_def: str) -> str:
        """Keep only the target function block and drop runaway continuations."""
        start = code.find(entry_def)
        if start < 0:
            return code
        lines = code[start:].splitlines()
        kept: list[str] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            is_top_level = bool(line) and (line[0] not in (" ", "\t"))
            if idx > 0 and is_top_level and (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("if __name__")
            ):
                break
            kept.append(line)
        return "\n".join(kept).strip()

    @staticmethod
    def _normalize_body(body_text: str) -> str:
        """Normalize body-only completion into a compilable function body."""
        raw_lines = body_text.replace("\t", "    ").splitlines()
        start_idx = 0
        code_prefixes = (
            "for ", "while ", "if ", "elif ", "else:", "try:", "except ",
            "with ", "return ", "raise ", "assert ", "from ", "import ",
            "def ", "class ", "@", "pass", "break", "continue", "#",
        )
        for i, line in enumerate(raw_lines):
            stripped = line.strip()
            if stripped and (stripped.startswith(code_prefixes) or "=" in stripped):
                start_idx = i
                break
        dedented = textwrap.dedent("\n".join(raw_lines[start_idx:])).strip("\n")
        if not dedented:
            return ""

        normalized_lines: list[str] = []
        indent_level = 1
        for line in dedented.splitlines():
            stripped = line.strip()
            if not stripped:
                normalized_lines.append("")
                continue
            if stripped.startswith(("elif ", "else:", "except ", "finally:")):
                indent_level = max(1, indent_level - 1)
            normalized_lines.append((" " * (indent_level * 4)) + stripped)
            if stripped.endswith(":") and not stripped.startswith("#"):
                indent_level += 1
            elif stripped.startswith(("return ", "break", "continue", "pass", "raise ")):
                indent_level = max(1, indent_level - 1)
        return "\n".join(normalized_lines).strip("\n")

    def extract_solution(self, task: Dict, completion: str) -> str:
        """Build executable code as: imports + target function body."""
        completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
        completion = self._extract_code_block(completion)
        completion = completion.strip()

        entry_def = f"def {task['entry_point']}"
        prompt = task["prompt"].rstrip("\n")
        if entry_def in completion:
            normalized = self._trim_after_target_function(completion, entry_def)
        else:
            body = self._normalize_body(completion)
            normalized = prompt if not body else (prompt + "\n" + body)

        full_code = IMPORT_PKG + "\n" + normalized.strip() + "\n"
        try:
            ast.parse(full_code)
        except SyntaxError:
            # Final fallback: keep only stripped body lines under the prompt.
            body_fallback = textwrap.dedent(completion).strip("\n")
            body_fallback = "\n".join(
                ("    " + line.strip()) if line.strip() else ""
                for line in body_fallback.splitlines()
            )
            normalized = prompt if not body_fallback else (prompt + "\n" + body_fallback)
            full_code = IMPORT_PKG + "\n" + normalized.strip() + "\n"
        return full_code


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Return the appropriate adapter for the given dataset name."""
    name = name.lower()
    if name == "enamel":
        return ENAMELAdapter(**kwargs)
    raise ValueError(
        f"Unknown dataset adapter {name!r}. "
        "Available adapters: ['enamel']"
    )
