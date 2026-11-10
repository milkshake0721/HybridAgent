#!/usr/bin/env python3
"""Unified TensorRT (Torch-TensorRT + Torch Inductor) and SGLang high-concurrency runner for
Gemma-4-26B-A4B-HybridAgent (BF16) and Gemma-4-26B-A4B-HybridAgent-4bit (W4A16).

Usage Examples:
  1. Run with TensorRT-accelerated HybridEngine (single or multi-question batch):
     python3 sglang_tensorrt_runner.py \
       --backend tensorrt \
       --model milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit \
       --questions "If 3x + 7 = 22, what is 5x - 2?" "What is 10 + 20?" \
       --batch-size 2

  2. Run BF16 model with TensorRT-accelerated HybridEngine:
     python3 sglang_tensorrt_runner.py \
       --backend tensorrt \
       --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
       --question "A rectangle has a perimeter of 30 cm and an area of 54 cm^2. What are the lengths of its sides?"

  3. Run with SGLang backend (`SGLangHybridEngine` + TensorRT `HybridLink`):
     python3 sglang_tensorrt_runner.py \
       --backend sglang \
       --model milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit \
       --question "If 3x + 7 = 22, what is the value of 5x - 2?"

  4. Print the SGLang OpenAI-compatible Server launch command (`sglang.launch_server`):
     python3 sglang_tensorrt_runner.py \
       --backend sglang-serve-cmd \
       --model milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import shlex
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_agent import (
    HybridEngine,
    SGLangHybridEngine,
    SYSTEM_PROMPT,
    get_sglang_server_command,
    load_hybrid_agent,
    prepare_sglang_model_dir,
    resolve_model_dir,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HybridAgent TensorRT & SGLang Runner")
    p.add_argument(
        "--backend",
        choices=("tensorrt", "sglang", "eager", "sglang-serve-cmd"),
        default="tensorrt",
        help="Execution backend: 'tensorrt' (Torch-TensorRT HybridLink + Zero-Sync MoE), 'sglang', 'eager', or 'sglang-serve-cmd'",
    )
    p.add_argument(
        "--model",
        default="milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit",
        help="Local path or Hugging Face repo ID (supports both BF16 and 4-bit compressed-tensors)",
    )
    p.add_argument(
        "--question",
        default="If 3x + 7 = 22, what is the value of 5x - 2?",
        help="Question to solve",
    )
    p.add_argument(
        "--questions",
        nargs="*",
        default=None,
        help="Multiple questions to solve concurrently in a batch",
    )
    p.add_argument(
        "--questions-file",
        "--questions_file",
        dest="questions_file",
        default=None,
        help="Optional JSONL or text file of questions to run in batch",
    )
    p.add_argument(
        "--batch-size",
        "--batch_size",
        "--concurrency",
        "--question_batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="Concurrent question batch size (default: 1)",
    )
    p.add_argument(
        "--no-tools",
        "--no_tools",
        "--disable-tools",
        "--disable_tools",
        dest="no_tools",
        action="store_true",
        help="Disable <python> and <search> tool calls",
    )
    p.add_argument("--sglang-url", default=None, help="Optional running SGLang server URL (e.g. http://127.0.0.1:30000)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-latent-steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args(argv)


def _collect_questions(args: argparse.Namespace) -> list[str]:
    if args.questions:
        return list(args.questions)
    if getattr(args, "questions_file", None):
        p = Path(args.questions_file)
        if p.is_file():
            items: list[str] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("{"):
                    rec = json.loads(s)
                    items.append(str(rec.get("question") or rec.get("problem") or ""))
                else:
                    items.append(s)
            if items:
                return items
    return [args.question]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.backend == "sglang-serve-cmd":
        local_dir = prepare_sglang_model_dir(resolve_model_dir(args.model))
        cmd = get_sglang_server_command(local_dir)
        print("=== SGLang Server Launch Command ===")
        print(" ".join(shlex.quote(c) for c in cmd))
        return

    q_list = _collect_questions(args)
    batch_size = max(1, int(args.batch_size))

    if args.backend == "sglang":
        print(f"=== Initializing SGLangHybridEngine on {args.model} ===")
        t0 = time.time()
        engine = SGLangHybridEngine(
            args.model,
            sglang_url=args.sglang_url,
            max_new_tokens=args.max_new_tokens,
            max_latent_steps=args.max_latent_steps,
            temperature=args.temperature,
        )
        print(f"Initialized in {time.time() - t0:.1f}s (HybridLink backend: {engine.trt_backend_name})")
        try:
            t_q = time.time()
            if len(q_list) > 1 or batch_size > 1:
                trajs = engine.run_batch(q_list, max_concurrency=batch_size, system_prompt=SYSTEM_PROMPT)
            else:
                trajs = [engine.run(q_list[0], system_prompt=SYSTEM_PROMPT)]
            dt = max(1e-6, time.time() - t_q)
            for traj in trajs:
                for r_idx, r in enumerate(traj.rounds):
                    print(f"[Round {r_idx}] latent_steps={r.latent_steps} | text={r.text!r}")
                print(f"Final Answer: {traj.final_answer!r} ({dt:.2f}s | {len(trajs)/dt:.2f} questions/s)")
        finally:
            engine.shutdown()
        return

    print(f"=== Loading {args.model} with backend={args.backend!r} ===")
    t0 = time.time()
    model, tokenizer = load_hybrid_agent(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        backend=args.backend,
        max_batch_size=max(64, batch_size, len(q_list)),
    )
    acc_info = getattr(model, "_acceleration_info", None)
    vram_gb = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"Loaded in {time.time() - t0:.1f}s | VRAM: {vram_gb:.2f} GB")
    if acc_info:
        print(f"Acceleration info: {acc_info}")

    max_tc = 0 if args.no_tools else 6
    engine = HybridEngine(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        max_latent_steps=args.max_latent_steps,
        temperature=args.temperature,
        max_tool_calls=max_tc,
        max_python_calls=max_tc,
        max_search_calls=max_tc,
        default_batch_size=batch_size,
    )
    t_q = time.time()
    if len(q_list) > 1 or batch_size > 1:
        trajs = engine.run_batch(q_list, batch_size=batch_size, system_prompt=SYSTEM_PROMPT)
    else:
        trajs = [engine.run(q_list[0], system_prompt=SYSTEM_PROMPT)]
    dt = max(1e-6, time.time() - t_q)
    for traj in trajs:
        for r_idx, r in enumerate(traj.rounds):
            print(f"[Round {r_idx}] latent_steps={r.latent_steps} | text={r.text!r}")
        print(f"\nRendered Output:\n{traj.output_rendered}")
        print(f"\nFinal Answer: {traj.final_answer!r} ({dt:.2f}s | {len(trajs)/dt:.2f} questions/s)")


if __name__ == "__main__":
    main()
