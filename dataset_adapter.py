"""
Dataset adapters for ENAMEL and other code evaluation datasets.

Provides a unified interface through the adapter pattern.

Notes on the ENAMEL extractor (this file)
-----------------------------------------
The completion → ``samples`` extractor used by :class:`ENAMELAdapter` mirrors
the logic we hardened in the L4E bridge
(``L4E/LLM4EFFI/tools/openllm_out_to_enamel_samples.py``). The recipe is:

1. Strip ``<think>...</think>`` reasoning blocks.
2. Extract the last fenced code block (``python`` or unlabeled), with
   line-anchored regex so prose with stray ```` ``` ```` doesn't confuse us.
3. ``ast.parse`` the block and pick the strongest ``FunctionDef`` that
   matches the ENAMEL ``entry_point`` (using a small "function score").
   Filter out docstring-only / ``pass``-only stubs.
4. If the target function is trivial but some *other* function in the
   block is non-trivial, keep that one and add a thin wrapper that
   delegates ``entry_point(*args, **kwargs)`` to it.
5. If the whole block is unparsable, salvage individual ``def`` blocks,
   then fall back to slicing the target function by indentation, then to
   trimming the first ``def entry_point(`` block, and finally to treating
   the completion as a body to indent and re-attach under the prompt.
6. Validate the final candidate with ``ast.parse`` again and walk a small
   ladder of retries before giving up and emitting the prompt only.
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
# ENAMEL prompt templates & instruction wrapper
# ---------------------------------------------------------------------------

_ENAMEL_PROMPT_TEMPLATE = "{prompt}"
_FULL_FUNCTION_INSTRUCTION = (
    "Return only ONE complete Python function.\n"
    "Use exactly the given function signature.\n"
    "Do not output imports, tests, explanations, markdown, or extra functions.\n"
    "\n"
)

# Minimal single-turn instruct prompt (matches CLS eval/export_qwen_baseline.py).
# Raw ENAMEL ``pure`` + chat still fails on some Instruct checkpoints (e.g.
# deepseek-coder-6.7b-instruct): the model echoes docstrings instead of bodies.
_INSTRUCT_WRAP_TEMPLATE = """You are a Python programmer. Implement the function described below.

Task (verbatim from the benchmark):
-----
{prompt}
-----

Return ONLY the implementation in a single Python markdown code block.
The implementation must define a top-level function named exactly
`{entry_point}`. Include any imports it relies on. Do not print, do
not add example or test code, do not write anything outside the
fenced code block."""


# ---------------------------------------------------------------------------
# Code extraction / normalization helpers (ported from the L4E bridge)
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> str:
    """Return the last fenced code block, with line-anchored fences.

    The earlier version used ``r"```python\\s*(.*?)```"`` which would happily
    chew up prose containing a stray triple backtick. Anchoring the opening
    fence to the start of a line is critical for instruct-tuned models that
    sprinkle ``` inside their explanations.
    """
    matches = re.findall(r"(?:^|\n)```python\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"(?:^|\n)```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _strip_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _function_score(fn: ast.FunctionDef) -> int:
    """Rough "this function actually does something" score."""
    body = _strip_docstring_body(fn.body[:])
    if not body:
        return 0
    score = 0
    module = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(
            node,
            (
                ast.Return,
                ast.For,
                ast.While,
                ast.If,
                ast.Assign,
                ast.AugAssign,
                ast.Call,
                ast.Try,
                ast.With,
                ast.Raise,
                ast.Assert,
            ),
        ):
            score += 1
    return score


def _is_doc_or_pass_only(fn: ast.FunctionDef) -> bool:
    body = _strip_docstring_body(fn.body[:])
    if not body:
        return True
    return all(isinstance(node, ast.Pass) for node in body)


def _extract_function_by_indent(code: str, target_name: str) -> str:
    """Slice the target function block by indentation, ignoring trailing prose.

    Useful when models append "Here's an example usage:" or test cases after
    the function: ``ast.parse`` chokes, but the indentation tells us where
    the function ends.
    """
    needle = f"def {target_name}("
    idx = code.rfind(needle)
    if idx < 0:
        return ""
    lines = code[idx:].splitlines()
    if not lines:
        return ""
    kept = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            kept.append(line)
            continue
        if not line.startswith((" ", "\t")):
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _extract_def_blocks(text: str) -> list[str]:
    """Split free-form text into individual ``def`` blocks for AST salvage."""
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", line):
            start = i
            i += 1
            while i < len(lines):
                cur = lines[i]
                if re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", cur):
                    break
                if cur.strip().startswith(("```", "# Examples", "# Test", "if __name__")):
                    break
                i += 1
            blocks.append("\n".join(lines[start:i]).strip())
        else:
            i += 1
    return [b for b in blocks if b]


def _choose_best_function_from_blocks(
    blocks: list[str], entry_point: str
) -> tuple[str, bool]:
    """Pick the best ``FunctionDef`` across salvaged blocks.

    Returns ``(source, wrapped)``: when ``wrapped`` is True a thin entry-point
    wrapper has been appended that delegates to the strongest function found.
    """
    funcs: list[ast.FunctionDef] = []
    for block in blocks:
        try:
            tree = ast.parse(block)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                funcs.append(node)
    if not funcs:
        return "", False

    target_funcs = [fn for fn in funcs if fn.name == entry_point]
    if target_funcs:
        best_target = max(target_funcs, key=_function_score)
        if _function_score(best_target) > 0 and not _is_doc_or_pass_only(best_target):
            return ast.unparse(best_target).strip(), False

    nontrivial = [
        fn for fn in funcs if _function_score(fn) > 0 and not _is_doc_or_pass_only(fn)
    ]
    if not nontrivial:
        if target_funcs:
            best_target = max(target_funcs, key=_function_score)
            return ast.unparse(best_target).strip(), False
        best_any = max(funcs, key=_function_score)
        return ast.unparse(best_any).strip(), best_any.name != entry_point

    best_fn = max(nontrivial, key=_function_score)
    best_src = ast.unparse(best_fn).strip()
    if best_fn.name == entry_point:
        return best_src, False
    wrapper = (
        f"\n\ndef {entry_point}(*args, **kwargs):\n"
        f"    return {best_fn.name}(*args, **kwargs)\n"
    )
    return (best_src + wrapper).strip(), True


def _trim_target_function_block(code: str, entry_point: str) -> str:
    pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", flags=re.M)
    match = pattern.search(code)
    if not match:
        return ""
    lines = code[match.start():].splitlines()
    kept: list[str] = []
    for i, ln in enumerate(lines):
        if i > 0 and re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", ln):
            break
        if ln.strip().startswith(("```", "# Examples", "# Test", "if __name__")):
            break
        kept.append(ln)
    return "\n".join(kept).strip()


def _normalize_body_to_prompt(prompt: str, body_text: str) -> str:
    """Treat ``body_text`` as a function body and re-attach it under prompt."""
    body = textwrap.dedent(body_text).strip("\n")
    if not body:
        return prompt
    indented = "\n".join(
        ("    " + ln) if ln.strip() else "" for ln in body.splitlines()
    )
    return prompt if not indented else (prompt + "\n" + indented)


def _normalize_solution(prompt: str, entry_point: str, completion: str) -> str:
    """Main normalization pipeline. Mirrors the L4E bridge logic 1:1."""
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    prompt_clean = prompt.rstrip("\n")
    code = _extract_code_block(completion)
    prompt = prompt_clean
    target_def = f"def {entry_point}("

    normalized = ""
    wrapped = False

    try:
        tree = ast.parse(code)
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        target_funcs = [fn for fn in funcs if fn.name == entry_point]

        if target_funcs:
            best_target = max(target_funcs, key=_function_score)
            if _function_score(best_target) > 0 and not _is_doc_or_pass_only(best_target):
                normalized = ast.unparse(best_target).strip()
            else:
                nontrivial = [
                    fn for fn in funcs
                    if _function_score(fn) > 0 and not _is_doc_or_pass_only(fn)
                ]
                if nontrivial:
                    best_fn = max(nontrivial, key=_function_score)
                    best_src = ast.unparse(best_fn).strip()
                    if best_fn.name == entry_point:
                        normalized = best_src
                    else:
                        wrapper = (
                            f"\n\ndef {entry_point}(*args, **kwargs):\n"
                            f"    return {best_fn.name}(*args, **kwargs)\n"
                        )
                        normalized = (best_src + wrapper).strip()
                        wrapped = True
                else:
                    normalized = ast.unparse(best_target).strip()
        elif funcs:
            best_fn = max(funcs, key=_function_score)
            best_src = ast.unparse(best_fn).strip()
            wrapper = (
                f"\n\ndef {entry_point}(*args, **kwargs):\n"
                f"    return {best_fn.name}(*args, **kwargs)\n"
            )
            normalized = (best_src + wrapper).strip()
            wrapped = True
    except Exception:
        normalized = ""

    if not normalized:
        best_from_blocks, wrapped_from_blocks = _choose_best_function_from_blocks(
            _extract_def_blocks(code), entry_point
        )
        if best_from_blocks:
            normalized = best_from_blocks
            wrapped = wrapped or wrapped_from_blocks

    if not normalized:
        salvaged = _extract_function_by_indent(code, entry_point)
        if salvaged:
            try:
                tree2 = ast.parse(salvaged)
                tgt = [
                    n for n in tree2.body
                    if isinstance(n, ast.FunctionDef)
                    and n.name == entry_point
                    and _function_score(n) > 0
                    and not _is_doc_or_pass_only(n)
                ]
                if tgt:
                    normalized = ast.unparse(tgt[0]).strip()
            except Exception:
                pass

    if not normalized:
        if target_def in code:
            normalized = _trim_target_function_block(code, entry_point)
        elif re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\(", code, flags=re.M):
            normalized = code.strip()
        else:
            normalized = _normalize_body_to_prompt(prompt, code)

    if not normalized:
        normalized = _normalize_body_to_prompt(prompt, code)

    full = IMPORT_PKG + "\n" + normalized.strip() + "\n"
    try:
        ast.parse(full)
    except SyntaxError:
        trimmed = _trim_target_function_block(code, entry_point)
        if trimmed:
            retry = IMPORT_PKG + "\n" + trimmed.strip() + "\n"
            try:
                ast.parse(retry)
                full = retry
            except SyntaxError:
                body_retry = _normalize_body_to_prompt(prompt, code)
                retry2 = IMPORT_PKG + "\n" + body_retry.strip() + "\n"
                try:
                    ast.parse(retry2)
                    full = retry2
                except SyntaxError:
                    full = IMPORT_PKG + "\n" + prompt.strip() + "\n"
        else:
            body_retry = _normalize_body_to_prompt(prompt, code)
            retry2 = IMPORT_PKG + "\n" + body_retry.strip() + "\n"
            try:
                ast.parse(retry2)
                full = retry2
            except SyntaxError:
                full = IMPORT_PKG + "\n" + prompt.strip() + "\n"
    return full


def _minimal_solution(prompt: str, entry_point: str, completion: str) -> str:
    """Conservative extraction for ablation: fenced block only, no ast salvage or IMPORT_PKG."""
    completion = re.sub(
        r"<think>.*?</think>", "", completion, flags=re.DOTALL
    )
    code = _extract_code_block(completion)
    if not code.strip():
        code = completion.strip()
    target_def = f"def {entry_point}("
    if target_def in code:
        trimmed = _trim_target_function_block(code, entry_point)
        if trimmed:
            code = trimmed
    return code.strip() + "\n"


def _raw_solution(prompt: str, entry_point: str, completion: str) -> str:
    """Weakest ablation: append model text after the ENAMEL prompt with no structured extraction."""
    del entry_point  # unused; kept for a uniform signature with other extractors
    completion = re.sub(
        r"<think>.*?</think>", "", completion, flags=re.DOTALL
    )
    text = completion.strip()
    if text:
        return prompt.rstrip() + "\n" + text + "\n"
    return prompt.rstrip() + "\n"


# ---------------------------------------------------------------------------
# ENAMEL adapter
# ---------------------------------------------------------------------------

class ENAMELAdapter(DatasetAdapter):
    """Adapter for the ENAMEL dataset (CSV format).

    The ENAMEL CSV has columns: ``task_id, prompt, entry_point, ...``.
    The ``prompt`` column already contains the function signature + docstring.

    Parameters
    ----------
    csv_path:
        Path to ``dataset/enamel.csv`` (or wherever you stored it).
    protocol:
        Prompting protocol used by :meth:`format_prompt`.

        - ``"pure"`` (default): feed the raw ENAMEL prompt verbatim.
        - ``"l4e"``: prepend the L4E-style function-only instruction.
        - ``"instruct-wrap"``: fenced-code instruct template; use with chat mode
          when ``pure`` yields docstring-only stubs on an Instruct checkpoint.
    extract_mode:
        Post-processing in :meth:`extract_solution`.

        - ``"full"`` (default): ast alignment, salvage, and ``IMPORT_PKG`` header.
        - ``"minimal"``: last fenced block + trim to ``entry_point``; no ast/import header.
        - ``"raw"``: ``prompt`` + completion text, no code-block parsing.
    """

    dataset_name = "enamel"

    def __init__(
        self,
        csv_path: str = "dataset/enamel.csv",
        protocol: str = "pure",
        extract_mode: str = "full",
    ) -> None:
        self.csv_path = csv_path
        if protocol not in {"pure", "l4e", "instruct-wrap"}:
            raise ValueError(
                f"Unknown protocol {protocol!r}. "
                "Expected 'pure', 'l4e', or 'instruct-wrap'."
            )
        if extract_mode not in {"full", "minimal", "raw"}:
            raise ValueError(
                f"Unknown extract_mode {extract_mode!r}. "
                "Expected 'full', 'minimal', or 'raw'."
            )
        self.protocol = protocol
        self.extract_mode = extract_mode

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
        if self.protocol == "instruct-wrap":
            return _INSTRUCT_WRAP_TEMPLATE.format(
                prompt=task["prompt"],
                entry_point=task["entry_point"],
            )
        body = prompt_template.format(prompt=task["prompt"])
        if self.protocol == "l4e":
            return _FULL_FUNCTION_INSTRUCTION + body
        return body

    def extract_solution(self, task: Dict, completion: str) -> str:
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        if self.extract_mode == "minimal":
            return _minimal_solution(prompt, entry_point, completion)
        if self.extract_mode == "raw":
            return _raw_solution(prompt, entry_point, completion)
        return _normalize_solution(
            prompt=prompt,
            entry_point=entry_point,
            completion=completion,
        )


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
