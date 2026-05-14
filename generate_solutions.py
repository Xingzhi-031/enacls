"""
generate_solutions.py — batch solution generation via vLLM.

Generates code solutions for ENAMEL tasks and saves them as a samples JSON
compatible with enam.evaluate / demo.py.

Example usage (1 GPU, 1 solution per task, all problems):
    python generate_solutions.py \
        --model Qwen/Qwen3.5-4B \
        --dataset enamel \
        --num-solutions 1 \
        --output samples/qwen3.5-4b-n1.json

Example usage (subset of problems):
    python generate_solutions.py \
        --model Qwen/Qwen3.5-4B \
        --dataset enamel \
        --num-solutions 1 \
        --subset 0,1,2,3,4 \
        --output samples/qwen3.5-4b-n1-5q.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dataset_adapter import get_adapter


def _parse_subset(value: str) -> list[int] | None:
    if not value or value.strip().lower() == "all":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate code solutions via vLLM.")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--dataset", "-d", default="enamel",
                        choices=["enamel"],
                        help="Dataset to generate solutions for.")
    parser.add_argument("--dataset-csv", default="dataset/enamel.csv",
                        help="Path to ENAMEL CSV (enamel dataset only).")
    parser.add_argument("--num-solutions", "-n", type=int, default=1,
                        help="Number of solutions to generate per task.")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Number of tasks per vLLM batch.")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory fraction for vLLM (0.0-1.0).")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max new tokens per completion.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling probability.")
    parser.add_argument("--subset", default=None,
                        help="Comma-separated problem indices to run (default: all).")
    parser.add_argument("--protocol", choices=["pure", "l4e"], default="pure",
                        help="Prompting protocol. 'pure' = raw ENAMEL prompt "
                             "(official leaderboard setup, default). "
                             "'l4e' = prepend the L4E 'Return only ONE complete "
                             "Python function...' instruction.")
    parser.add_argument("--mode", choices=["auto", "chat", "completion"], default="auto",
                        help="Inference mode. 'chat' applies the tokenizer's chat "
                             "template (required for Instruct/Chat models). "
                             "'completion' feeds raw text (for base models). "
                             "'auto' (default) picks chat when the model id "
                             "contains 'instruct' / 'chat' / '-it'.")
    parser.add_argument("--output", "-o", default="samples/generated.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    # Load dataset
    adapter = get_adapter(
        args.dataset,
        csv_path=args.dataset_csv,
        protocol=args.protocol,
    )
    print(f"Protocol: {args.protocol}")
    print(f"Dataset: {adapter.dataset_name}")
    dataset = adapter.load_dataset()

    # Apply subset filter
    subset = _parse_subset(args.subset)
    if subset is not None:
        dataset = [t for t in dataset if t["problem_id"] in set(subset)]
    print(f"Tasks to process: {len(dataset)}")

    prompt_template = adapter.get_prompt_template()

    # Decide inference mode (chat vs raw completion).
    mode = args.mode
    if mode == "auto":
        mid = args.model.lower()
        if any(tok in mid for tok in ("instruct", "chat", "-it", "/it-")):
            mode = "chat"
        else:
            mode = "completion"
    print(f"Inference mode: {mode}")

    # Build all prompts (num_solutions copies per task for diverse sampling)
    raw_prompts: list[str] = []
    task_indices: list[int] = []  # which task each prompt belongs to
    for task in dataset:
        p = adapter.format_prompt(task, prompt_template)
        for _ in range(args.num_solutions):
            raw_prompts.append(p)
            task_indices.append(task["problem_id"])

    # Apply chat template if needed. Instruct/Chat models *must* go through
    # apply_chat_template, otherwise they degenerate into HumanEval-style
    # "docstring -> doctest -> assert" continuations and never write a body
    # (observed on Qwen2.5-Coder-14B-Instruct in the L4E run).
    if mode == "chat":
        from transformers import AutoTokenizer
        print("Applying tokenizer chat template...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts: list[str] = []
        for p in raw_prompts:
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(chat_text)
    else:
        prompts = raw_prompts

    # Load vLLM
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )

    # Stop tokens differ by mode:
    # - chat: let the model emit a fenced code block followed by prose; the
    #   adapter's extractor picks the last fenced block. Heavy stop tokens
    #   would chop the response in the middle of the block.
    # - completion: stop on the first "next top-level def/class/test" so we
    #   keep only the body of the target function.
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
        n=1,  # one completion per prompt entry (we already replicated prompts)
        stop=stop_tokens,
    )

    # Generate in batches
    print(f"Generating {len(prompts)} completions (batch_size={args.batch_size})...")
    all_completions: list[str] = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for out in outputs:
            all_completions.append(out.outputs[0].text)

    # Assemble solutions dict: {problem_id: [solution, ...]}
    solutions: dict[str, list[str]] = {}
    for task in dataset:
        solutions[str(task["problem_id"])] = []

    for prompt_idx, completion in enumerate(all_completions):
        pid = str(task_indices[prompt_idx])
        task = next(t for t in dataset if str(t["problem_id"]) == pid)
        solution = adapter.extract_solution(task, completion)
        solutions[pid].append(solution)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(solutions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(solutions)} tasks → {out_path}")


if __name__ == "__main__":
    main()
