"""
L4E "planning + coding" pipeline reproduction (no case-iter, no code execution).

Stages per task (single vLLM model, batched across tasks at each stage):
    1. task_description_gen   re-describe the problem
    2. algorithm_generation   five algorithm sketches in one shot
    3. (optional) knowledge_db   per-algorithm practical optimisation tips
    4. algorithm_to_code (x5)    one Python implementation per algorithm sketch
    5. fast_code_choice           LLM-vote select the best candidate

This module deliberately does NOT import any LLM4EFFI Python code, does NOT
invoke their open_llm_generation.py / tools/*.py, and does NOT execute the
generated code (that is L4E's `case-iter` stage, intentionally out of scope
for the "planning + coding" reproduction).

The five prompt strings below are copy-pasted verbatim from
``L4E/LLM4EFFI_official/prompt/prompt.py`` for protocol faithfulness; only the
``{ques}`` / ``{plan_and_algo}`` / ``{original_question}`` placeholders are
filled in by Python (via helper functions, NOT ``str.format``, because the
prompts contain literal curly braces meant to be shown to the LLM).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Verbatim prompts from L4E/LLM4EFFI_official/prompt/prompt.py
# ---------------------------------------------------------------------------

_TASK_DESCRIPTION_GEN_SYSTEM = """
    As a professional algorithm engineer, please analyze this algorithm problem according to the following categories.Do not generate any example implementation:
1. Entry point function name:
2. Input/Output conditions
3. Edge Cases and Parameters type(Int String...)
4. expected behavior
    """

_ALGORITHM_GEN_SYSTEM = """As a professional algorithm engineer, you can effectively design multiple algorithms to solve the problem with low time complexity and output them in pseudo algorithm format, and pseudo algorithm is a nonlinear, high-level programming language for algorithmic logic. It combines natural language and programming structures to express the steps and sums of algorithms. The main purpose of process algorithms is to clearly display the core ideas and logic of the algorithm without relying on specific programming language syntax. Please design an 5 excellent algorithm solution based on the problem description provided. The time complexity of the algorithm needs to be as small as possible, and try to output 10 algorithms  in the form of a pseudo-algorithm in the following format:
    PS: DO NOT provide implementation example!
```algorithm1
{{algorithm key description:this algorithm using xxx,the key is to make sure xxx}}
{ pseudo algorithm: ..}
```
```algorithm2
{algorithm key description:this algorithm using xxx,the key is to make sure xxx}
{ pseudo algorithm: ..}
```
```algorithm3
{algorithm key description:this algorithm using xxx,the key is to make sure xxx}
{ pseudo algorithm: ..}
```
```algorithm4
{algorithm key description:this algorithm using xxx,the key is to make sure xxx}
{ pseudo algorithm: ..}
```
```algorithm5
{algorithm key description:this algorithm using xxx,the key is to make sure xxx}
{ pseudo algorithm: ..}
```
    """

_KNOWLEDGE_DB_SYSTEM = """
As a professional Python algorithm programming expert, please provide suggestions for improving code efficiency based on the potential inefficiencies mentioned above. For example:
1.\tUsing xxx instead of xxx can significantly improve code efficiency.
Please provide at least 20 suggestions.
    """

_ALGO2CODE_SYSTEM = """
As a professional Python algorithm engineer, please convert the selected algorithm into corresponding Python code. Ensure the code is complete and well-formatted. When converting to a standardized format, be sure to follow the guidelines specified in the "original question format":
1.\tIf "from typing import List" appears in the original question, please retain it.
2.\tUse the same function name as given in the original question format; do not rename it.
3.\tYou may incorporate practical optimization details drawn from the knowledge base.
The final output format should be as follows:
```python
{code}
```
    """

_FAST_CODE_CHOICE_SYSTEM = """
As a professional Python algorithm engineer, please help me choose the most efficient Python code from the following codes. It is worth mentioning that it is necessary to consider the time complexity and practical level comprehensively:
INPUT:
{"1":"def ...()....",
"2": "def ...()..."
}
OUTPUT:
```text
{key}
```
EXAMPLE:
INPUT:
{"1":"def ...()....",
"2": "def ...()..."
}
OUTPUT:
```text
1
```
    """


# ---------------------------------------------------------------------------
# Helpers (avoid str.format because the L4E prompts contain literal "{...}")
# ---------------------------------------------------------------------------

def _td_user(ques: str) -> str:
    return f"""
        The algorithm problem description is as follows: {ques}
        """


def _algo2code_user(plan_and_algo: str, original_question: str) -> str:
    return f"""
      Selected plan and algorithm: {plan_and_algo}
      original question format: {original_question}
       """


def _join_system_user(system: str, user: str) -> str:
    """Collapse a (system, user) pair into a single user message.

    L4E's GPTReply ``getreply(system, user, history)`` is faithful to the
    OpenAI Chat API; locally we route everything through one user turn and
    rely on ``apply_chat_template`` to add the role boundaries.
    """
    return f"{system.strip()}\n\n{user.strip()}"


def _parse_algorithms(reply: str, num_algos: int = 5) -> list[str]:
    """Pull ``algorithm1..N`` fenced blocks out of an LLM reply.

    The L4E prompt asks for ``\\`\\`\\`algorithm{i}\\n...\\`\\`\\``` blocks; we
    are tolerant of leading whitespace and language tags like ``algorithm1 ``.
    """
    out: list[str] = []
    for i in range(1, num_algos + 1):
        pattern = re.compile(rf"```algorithm{i}\b[^\n]*\n(.*?)```", re.DOTALL)
        m = pattern.search(reply)
        out.append(m.group(1).strip() if m else "")
    return out


def _parse_select_index(reply: str, max_index: int) -> int:
    """Extract the chosen ``1..max_index`` index from the LLM-vote stage."""
    m = re.search(r"```text\s*\n?\s*(\d+)\s*\n?```", reply, flags=re.DOTALL)
    if not m:
        m = re.search(r"\b([1-9]\d*)\b", reply)
    if not m:
        return 0
    try:
        idx = int(m.group(1)) - 1
    except ValueError:
        return 0
    if idx < 0 or idx >= max_index:
        return 0
    return idx


def _wrap_python_fence(code_block: str) -> str:
    """Ensure the chosen candidate text is wrapped in a python fence.

    The downstream ENAMEL extractor (``_normalize_solution`` in
    ``dataset_adapter.py``) walks fenced blocks first; wrapping here means an
    LLM reply that is already fenced stays unchanged, while a reply that came
    back as raw code still hits the extractor's "last fenced block" branch.
    """
    if re.search(r"(?:^|\n)```", code_block):
        return code_block
    return f"```python\n{code_block.strip()}\n```"


# ---------------------------------------------------------------------------
# L4EPipeline
# ---------------------------------------------------------------------------

class L4EPipeline:
    """Run the L4E "planning + coding" stages with a single vLLM model.

    Parameters
    ----------
    llm:
        A ``vllm.LLM`` instance shared across all stages.
    sampling_params:
        A ``vllm.SamplingParams``; we reuse the same params at every stage so
        the experiment is single-knob.
    chat_template_fn:
        Optional ``Callable[[str], str]`` that wraps a raw user message into
        the model's chat template (typically the tokenizer's
        ``apply_chat_template`` invocation). Required for Instruct/Chat
        checkpoints; pass ``None`` for base-completion models.
    use_knowledge:
        When True, run an extra "knowledge base" LLM call per algorithm and
        concatenate the result to the algo->code prompt. Faithful to L4E's
        ``codegen_process4`` but multiplies LLM calls by ~2x.
    num_algos:
        Algorithm candidates produced per task (default 5, matching L4E).
    progress:
        Optional ``Callable[[str], None]`` for stage-level status logging.
    """

    def __init__(
        self,
        llm,
        sampling_params,
        *,
        chat_template_fn: Callable[[str], str] | None = None,
        use_knowledge: bool = False,
        num_algos: int = 5,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.sampling_params = sampling_params
        self.chat_template_fn = chat_template_fn
        self.use_knowledge = use_knowledge
        self.num_algos = num_algos
        self._progress = progress or (lambda msg: None)

    # ---- low-level batched chat ----
    def _chat(self, prompts: Sequence[str]) -> list[str]:
        if not prompts:
            return []
        materialised: list[str] = list(prompts)
        if self.chat_template_fn is not None:
            materialised = [self.chat_template_fn(p) for p in materialised]
        outputs = self.llm.generate(materialised, self.sampling_params)
        return [out.outputs[0].text for out in outputs]

    # ---- stages ----
    def _stage_task_description(self, tasks: Sequence[dict]) -> list[str]:
        prompts = [
            _join_system_user(_TASK_DESCRIPTION_GEN_SYSTEM, _td_user(t["prompt"]))
            for t in tasks
        ]
        self._progress(f"[L4E-pipe] stage 1/4: task description ({len(prompts)} calls)")
        return self._chat(prompts)

    def _stage_algorithms(self, augmented_questions: Sequence[str]) -> list[list[str]]:
        prompts = [
            _join_system_user(_ALGORITHM_GEN_SYSTEM, q)
            for q in augmented_questions
        ]
        self._progress(f"[L4E-pipe] stage 2/4: algorithm planning ({len(prompts)} calls)")
        replies = self._chat(prompts)
        return [_parse_algorithms(r, self.num_algos) for r in replies]

    def _stage_knowledge(
        self,
        algos_per_task: Sequence[Sequence[str]],
    ) -> dict[tuple[int, int], str]:
        if not self.use_knowledge:
            return {}
        prompts: list[str] = []
        index: list[tuple[int, int]] = []
        for ti, algos in enumerate(algos_per_task):
            for ai, algo in enumerate(algos):
                if not algo:
                    continue
                prompts.append(_join_system_user(_KNOWLEDGE_DB_SYSTEM, algo))
                index.append((ti, ai))
        self._progress(f"[L4E-pipe] stage 2.5/4: knowledge base ({len(prompts)} calls)")
        replies = self._chat(prompts)
        return {idx: r for idx, r in zip(index, replies)}

    def _stage_algo_to_code(
        self,
        tasks: Sequence[dict],
        algos_per_task: Sequence[Sequence[str]],
        knowledge: dict[tuple[int, int], str],
    ) -> list[list[str]]:
        prompts: list[str] = []
        index: list[tuple[int, int]] = []
        for ti, (task, algos) in enumerate(zip(tasks, algos_per_task)):
            for ai, algo in enumerate(algos):
                if not algo:
                    continue
                plan_and_algo = algo
                kb = knowledge.get((ti, ai), "")
                if kb:
                    plan_and_algo = (
                        f"{plan_and_algo}\n\nKnowledge base of practical tips:\n{kb}"
                    )
                prompts.append(
                    _join_system_user(
                        _ALGO2CODE_SYSTEM,
                        _algo2code_user(plan_and_algo, task["prompt"]),
                    )
                )
                index.append((ti, ai))
        self._progress(f"[L4E-pipe] stage 3/4: algorithm -> code ({len(prompts)} calls)")
        replies = self._chat(prompts)
        codes: list[list[str]] = [[] for _ in tasks]
        for (ti, _ai), reply in zip(index, replies):
            codes[ti].append(reply)
        return codes

    def _stage_select(self, codes_per_task: Sequence[Sequence[str]]) -> list[int]:
        prompts: list[str] = []
        prompt_index: list[int] = []
        for ti, codes in enumerate(codes_per_task):
            if len(codes) <= 1:
                continue
            blob_lines = []
            for i, c in enumerate(codes, start=1):
                blob_lines.append(f'"{i}":\n```python\n{c.strip()}\n```')
            blob = "{\n" + ",\n".join(blob_lines) + "\n}"
            prompts.append(_join_system_user(_FAST_CODE_CHOICE_SYSTEM, blob))
            prompt_index.append(ti)
        self._progress(f"[L4E-pipe] stage 4/4: LLM-vote select ({len(prompts)} calls)")
        replies = self._chat(prompts)
        chosen: list[int] = [0] * len(codes_per_task)
        for ti, reply in zip(prompt_index, replies):
            chosen[ti] = _parse_select_index(reply, len(codes_per_task[ti]))
        return chosen

    # ---- top-level entry ----
    def run(self, tasks: Sequence[dict]) -> list[str]:
        """Drive the pipeline for ``tasks`` and return one completion per task.

        Each returned completion is suitable to feed straight into
        ``ENAMELAdapter.extract_solution``; if a stage failed for a task we
        return an empty string so the extractor falls back to "prompt only".
        """
        if not tasks:
            return []

        td_replies = self._stage_task_description(tasks)
        augmented_qs = [
            (t["prompt"].strip() + "\n" + td.strip()).strip()
            for t, td in zip(tasks, td_replies)
        ]

        algos_per_task = self._stage_algorithms(augmented_qs)
        knowledge = self._stage_knowledge(algos_per_task)
        codes_per_task = self._stage_algo_to_code(tasks, algos_per_task, knowledge)
        chosen = self._stage_select(codes_per_task)

        finals: list[str] = []
        for ti, codes in enumerate(codes_per_task):
            if not codes:
                finals.append("")
                continue
            idx = chosen[ti]
            if idx < 0 or idx >= len(codes):
                idx = 0
            finals.append(_wrap_python_fence(codes[idx]))
        return finals
