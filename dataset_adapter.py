"""
Dataset adapters for ENAMEL and other code evaluation datasets.

Provides a unified interface through the adapter pattern.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
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

    def extract_solution(self, task: Dict, completion: str) -> str:
        """Build executable code as: IMPORT_PKG + function definition + body."""
        import re
        import textwrap

        # 1) Remove chain-of-thought / markdown wrappers.
        completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
        completion = completion.replace("```python", "").replace("```", "")
        completion = completion.strip()

        entry_def = f"def {task['entry_point']}"
        if entry_def in completion:
            # Completion already contains a full function; keep only from target def.
            start = completion.find(entry_def)
            normalized = completion[start:].rstrip()
            return IMPORT_PKG + "\n" + normalized

        # 2) Base-model case: completion is function body only.
        #    Explicitly insert task['prompt'] between imports and completion.
        body = textwrap.dedent(completion).strip("\n")
        indented_body = "\n".join(
            ("    " + line) if line.strip() else ""
            for line in body.splitlines()
        )
        prompt = task["prompt"].rstrip("\n")
        normalized = prompt if not indented_body else (prompt + "\n" + indented_body)
        return IMPORT_PKG + "\n" + normalized


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
