"""
Iterative ENAMEL generation on PROBLEMSET: each round samples K completions
per problem, keeps the best under official Evaluator scoring, then compares
with the previous-round best and keeps the better one. Round 2+ always appends
the previous best code to the prompt for refinement.

Requires cached tests (same as demo.py), e.g. cache/eval~tests.pkl

Example:
    python generate_solutions_refine.py \\
        --model Qwen/Qwen2.5-Coder-3B-Instruct \\
        --refine-rounds 3 \\
        --candidates-per-round 3 \\
        --tests cache/eval~tests.pkl \\
        --output samples/refine-r3-k3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset_adapter import IMPORT_PKG, get_adapter
from enam import PROBLEMSET, Evaluator


def _parse_subset(value: str | None) -> list[int] | None:
    if not value or value.strip().lower() == "all":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _strip_imports(full_code: str) -> str:
    if full_code.startswith(IMPORT_PKG):
        return full_code[len(IMPORT_PKG) :].lstrip()
    return full_code


def _score_tuple(passed: bool, effs: list[float], hardness: np.ndarray) -> tuple[int, float]:
    if not passed:
        return (0, 0.0)
    e = float(np.average(np.array(effs, dtype=np.float64), weights=hardness))
    return (1, e)


def _summarize(scores: dict[int, tuple[int, float]]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    n = float(len(scores))
    pass_at_1 = sum(v[0] for v in scores.values()) / n
    eff_at_1 = sum(v[1] for v in scores.values()) / n
    return pass_at_1, eff_at_1


def _pick_best_candidate(
    evaluator: Evaluator,
    hardness: np.ndarray,
    problem_idx: int,
    candidates: list[str],
    verbose: bool,
) -> tuple[str, tuple[int, float]]:
    refs = evaluator.compute_refs(i=problem_idx)
    best_code = candidates[0]
    passed0, effs0, _ = evaluator.evaluate1(problem_idx, best_code, refs, verbose)
    best_t = _score_tuple(passed0, effs0, hardness)
    for code in candidates[1:]:
        passed, effs, _ = evaluator.evaluate1(problem_idx, code, refs, verbose)
        t = _score_tuple(passed, effs, hardness)
        if t > best_t:
            best_t = t
            best_code = code
    return best_code, best_t

def _better(
    a: tuple[str, tuple[int, float]],
    b: tuple[str, tuple[int, float]],
) -> tuple[str, tuple[int, float]]:
    """Return the better of (code, score_tuple); lexicographic on score."""
    _, ta = a
    _, tb = b
    return a if ta > tb else b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative ENAMEL generation with per-round best-of-K and official scoring."
    )
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", "-d", default="enamel", choices=["enamel"])
    parser.add_argument("--dataset-csv", default="dataset/enamel.csv")
    parser.add_argument(
        "--subset",
        default=None,
        help="Comma-separated problem ids. Default: PROBLEMSET only.",
    )
    parser.add_argument("--refine-rounds", type=int, default=3, help="Number of outer refinement rounds.")
    parser.add_argument(
        "--candidates-per-round",
        "-k",
        type=int,
        default=3,
        help="Parallel completions per problem each round (vLLM n=...).",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Problems per vLLM batch (not K).")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--protocol", choices=["pure", "l4e"], default="pure",
                        help="Prompting protocol (same semantics as generate_solutions.py).")
    parser.add_argument("--mode", choices=["auto", "chat", "completion"], default="auto",
                        help="Inference mode. 'chat' applies the tokenizer's chat template "
                             "(required for Instruct/Chat models). 'completion' uses raw text.")
    parser.add_argument(
        "--tests",
        type=str,
        default="cache/eval~tests.pkl",
        help="Pickled tests file (run demo.py once without --tests to create cache/eval~tests.pkl).",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="dataset/enamel.csv",
        help="CSV path passed to Evaluator (same as demo --dataset).",
    )
    parser.add_argument("--output", "-o", default="samples/refined.json")
    parser.add_argument("--eval-verbose", action="store_true", help="Forward verbose to evaluate1.")
    args = parser.parse_args()

    refine_rounds = max(3, args.refine_rounds)
    k = max(3, args.candidates_per_round)

    adapter = get_adapter(args.dataset, csv_path=args.dataset_csv, protocol=args.protocol)
    dataset = adapter.load_dataset()
    problemset = set(PROBLEMSET)
    subset = _parse_subset(args.subset)
    if subset is None:
        dataset = [t for t in dataset if t["problem_id"] in problemset]
    else:
        want = set(subset)
        dataset = [t for t in dataset if t["problem_id"] in want and t["problem_id"] in problemset]

    refine_tasks: list[dict] = sorted(dataset, key=lambda x: x["problem_id"])
    print(f"Tasks to refine (PROBLEMSET only): {len(refine_tasks)}")

    evaluator = Evaluator(
        problems=args.eval_dataset,
        subset=sorted(problemset),
        n_tests=[8, 4, 4, 4],
        n_reps=6,
        hardness=[0.0, 3.0, 3.0, 4.0],
        memory_giga=4.0,
        timeout_factor=2.0,
        tolerence_sec=0.01,
        seed=998244353,
    )
    if not evaluator.load_tests(fname=args.tests):
        raise FileNotFoundError(
            f"Tests not found: {args.tests!r}. Run once: python demo.py --load_name <any existing json> "
            f"(omit --tests) to build cache/eval~tests.pkl, or pass a valid --tests path."
        )
    hardness = np.array(evaluator.hardness, dtype=np.float64)

    # Decide inference mode (mirrors generate_solutions.py).
    mode = args.mode
    if mode == "auto":
        mid = args.model.lower()
        if any(tok in mid for tok in ("instruct", "chat", "-it", "/it-")):
            mode = "chat"
        else:
            mode = "completion"
    print(f"Protocol: {args.protocol}")
    print(f"Inference mode: {mode}")

    # Load tokenizer for chat-template wrapping (Instruct/Chat models *must*
    # go through apply_chat_template; see generate_solutions.py for details).
    tokenizer = None
    if mode == "chat":
        from transformers import AutoTokenizer
        print("Loading tokenizer for chat template...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )

    # Stop tokens follow the same rule as generate_solutions.py:
    # chat mode lets the model end naturally (EOS); completion mode stops on
    # the first "next top-level def/class/test".
    if mode == "chat":
        stop_tokens: list[str] = []
    else:
        stop_tokens = [
            "\n\ndef ",
            "\nclass ",
            "if __name__",
            "\nprint(",
            "\n# Test",
            "\nassert ",
            "\n# Example",
            "\n```",
        ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=k,
        stop=stop_tokens,
    )

    def _wrap_chat(prompts: list[str]) -> list[str]:
        if mode != "chat" or tokenizer is None:
            return prompts
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    prompt_template = adapter.get_prompt_template()
    solutions: dict[str, list[str]] = {str(t["problem_id"]): [] for t in refine_tasks}

    # Refine PROBLEMSET tasks
    best_code: dict[int, str | None] = {t["problem_id"]: None for t in refine_tasks}
    best_score: dict[int, tuple[int, float]] = {
        t["problem_id"]: (0, 0.0) for t in refine_tasks
    }

    round_history: list[dict[str, float | int]] = []
    for rnd in range(refine_rounds):
        desc = f"Refine round {rnd + 1}/{refine_rounds}"
        prompts_r: list[str] = []
        tasks_r: list[dict] = []
        for task in refine_tasks:
            pid = task["problem_id"]
            base = adapter.format_prompt(task, prompt_template)
            if rnd > 0 and best_code[pid] is not None:
                body = _strip_imports(best_code[pid])
                base = (
                    base
                    + "\n\n# Previous implementation (may be wrong or slow). "
                    "Output one improved complete function only.\n\n"
                    + body
                )
            prompts_r.append(base)
            tasks_r.append(task)

        wrapped_prompts_r = _wrap_chat(prompts_r)
        for i in tqdm(range(0, len(wrapped_prompts_r), args.batch_size), desc=desc):
            batch_p = wrapped_prompts_r[i : i + args.batch_size]
            batch_t = tasks_r[i : i + args.batch_size]
            outs = llm.generate(batch_p, sampling_params)
            for out, task in zip(outs, batch_t):
                pid = task["problem_id"]
                idx = int(pid)
                cands = [adapter.extract_solution(task, o.text) for o in out.outputs]
                round_best, round_t = _pick_best_candidate(
                    evaluator, hardness, idx, cands, args.eval_verbose
                )
                if best_code[pid] is None:
                    best_code[pid] = round_best
                    best_score[pid] = round_t
                else:
                    merged = _better((round_best, round_t), (best_code[pid], best_score[pid]))
                    best_code[pid], best_score[pid] = merged
        p1, e1 = _summarize(best_score)
        round_history.append({"round": rnd + 1, "pass@1_proxy": p1, "eff@1_proxy": e1})
        print(f"[Round {rnd + 1}] pass@1_proxy={p1:.6f}, eff@1_proxy={e1:.6f}")

    # By construction these are monotonic non-decreasing (keep-better policy).
    for i in range(1, len(round_history)):
        prev = round_history[i - 1]
        cur = round_history[i]
        if cur["pass@1_proxy"] < prev["pass@1_proxy"] or cur["eff@1_proxy"] < prev["eff@1_proxy"]:
            raise RuntimeError("Round metrics decreased unexpectedly; monotonic guarantee violated.")

    for task in refine_tasks:
        pid = task["problem_id"]
        code = best_code[pid]
        if code is None:
            code = adapter.extract_solution(task, "")
        solutions[str(pid)] = [code]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(solutions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(solutions)} tasks → {out_path}")
    hist_path = out_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps(round_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved round history → {hist_path}")


if __name__ == "__main__":
    main()
