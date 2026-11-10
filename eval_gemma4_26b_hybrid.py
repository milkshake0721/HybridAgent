#!/usr/bin/env python3
"""Evaluate Gemma-4-26B-A4B-HybridAgent (BF16 or 4-bit) on the 10-question math/logic benchmark
using high-concurrency batched TensorRT execution, or optionally delegate to the 5-benchmark
HuggingFace evaluation suite (`eval_hf_benchmarks.py`).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_agent.answer import extract_answer
from hybrid_agent.engine import HybridEngine
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit"
DEFAULT_OUT = "eval_results.json"

BENCHMARK = [
    {"id": 1, "q": "If 3x + 7 = 22, what is the value of 5x - 2?", "expected": "23"},
    {"id": 2, "q": "What is the sum of all integers from 1 to 20 inclusive?", "expected": "210"},
    {"id": 3, "q": "A rectangle has perimeter 30 and area 54. What is the length of its longer side?", "expected": "9"},
    {"id": 4, "q": "How many distinct rearrangements of the letters in the word 'LEVEL' are there?", "expected": "30"},
    {"id": 5, "q": "If f(x) = x^2 - 3x + 4, what is f(f(2))?", "expected": "2"},
    {"id": 6, "q": "A train travels 120 km at 60 km/h, and then another 120 km at 40 km/h. What is the average speed for the entire trip in km/h?", "expected": "48"},
    {"id": 7, "q": "What is the remainder when 2^10 is divided by 7?", "expected": "2"},
    {"id": 8, "q": "In a triangle, two angles measure 45 degrees and 65 degrees. What is the measure of the third angle in degrees?", "expected": "70"},
    {"id": 9, "q": "Find the positive solution to x^2 + 4x - 21 = 0.", "expected": "3"},
    {"id": 10, "q": "Three fair coins are tossed simultaneously. What is the probability of getting exactly two heads? Express your answer as a simplified fraction a/b.", "expected": "3/8"},
]


def normalize_math_answer(ans: str) -> str:
    """Normalize LaTeX fractions (e.g. \\frac{3}{8} -> 3/8) and units for exact comparison."""
    s = ans.strip()
    m = re.match(r"^\\d?frac\{([^{}]+)\}\{([^{}]+)\}$", s)
    if m:
        return f"{m.group(1).strip()}/{m.group(2).strip()}"
    nums = re.findall(r"-?\d+(?:/\d+)?", s)
    return nums[-1].strip() if nums else s


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Gemma-4-26B-A4B-HybridAgent with Batched TensorRT")
    p.add_argument("pos_model", nargs="?", default=None, help="Positional model path or repo ID")
    p.add_argument("pos_output", nargs="?", default=None, help="Positional output JSON path")
    p.add_argument("pos_backend", nargs="?", default=None, help="Positional backend ('tensorrt', 'eager', 'sglang')")
    p.add_argument("pos_batch_size", nargs="?", type=int, default=None, help="Positional batch size")
    p.add_argument("--model", default=None, help="Model path or Hugging Face repo ID")
    p.add_argument("--output", "--out", dest="output", default=None, help="Output JSON file path")
    p.add_argument("--backend", choices=("tensorrt", "eager", "sglang"), default=None)
    p.add_argument(
        "--batch-size",
        "--batch_size",
        "--concurrency",
        dest="batch_size",
        type=int,
        default=None,
        help="Concurrent question batch size (default: 10)",
    )
    p.add_argument("--no-tools", "--no_tools", dest="no_tools", action="store_true")
    p.add_argument(
        "--hf-benchmarks",
        "--benchmarks",
        dest="hf_benchmarks",
        nargs="*",
        default=None,
        help="Optional 5-benchmark HF suite names to run via eval_hf_benchmarks",
    )
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/benchmarks")
    p.add_argument("--dataset-dir", "--dataset_dir", dest="dataset_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--synthetic-smoke", "--dry-run", dest="synthetic_smoke", action="store_true")
    args = p.parse_args(argv)

    args.model = args.model or args.pos_model or DEFAULT_MODEL
    args.output = args.output or args.pos_output or DEFAULT_OUT
    args.backend = args.backend or args.pos_backend or "tensorrt"
    args.batch_size = args.batch_size or args.pos_batch_size or len(BENCHMARK)
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)

    if args.hf_benchmarks is not None:
        import eval_hf_benchmarks

        hf_args = argparse.Namespace(
            model=args.model,
            backend=args.backend,
            benchmarks=args.hf_benchmarks if args.hf_benchmarks else ["all"],
            batch_size=args.batch_size,
            limit=args.limit,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            synthetic_smoke=args.synthetic_smoke,
            seed=42,
            vram_gb=96.0,
            max_new_tokens=4096,
            max_latent_steps=32,
            temperature=0.0,
        )
        return eval_hf_benchmarks.run_suite(hf_args)

    source = args.model
    out_json = args.output
    backend = args.backend
    batch_size = max(1, int(args.batch_size))
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    max_tc = 0 if args.no_tools else 6
    model, tokenizer = load_hybrid_agent(
        source,
        dtype=torch.bfloat16,
        device="cuda",
        backend=backend,
        max_batch_size=max(32, batch_size),
    )
    engine = HybridEngine(
        model,
        tokenizer,
        max_new_tokens=1024,
        max_latent_steps=64,
        temperature=0.0,
        top_p=1.0,
        max_tool_calls=max_tc,
        max_python_calls=max_tc,
        max_search_calls=max_tc,
        default_batch_size=batch_size,
    )

    questions = [item["q"] for item in BENCHMARK]
    t0 = time.time()
    trajectories = engine.run_batch(questions, batch_size=batch_size, system_prompt=SYSTEM_PROMPT)
    total_dt = max(1e-6, time.time() - t0)
    avg_dt = total_dt / max(1, len(BENCHMARK))

    results = []
    correct = 0
    for item, traj in zip(BENCHMARK, trajectories):
        raw_ans = traj.final_answer or extract_answer(traj.output)
        pred = normalize_math_answer(raw_ans)
        ok = pred == item["expected"]
        correct += int(ok)
        latent_steps = sum(r.latent_steps for r in traj.rounds)
        print(
            f"[Q{item['id']:02d}] {'PASS' if ok else 'FAIL'} | expected={item['expected']} "
            f"pred={pred} (raw={raw_ans!r}) | latent_steps={latent_steps} | {avg_dt:.1f}s"
        )
        results.append(
            {
                "id": item["id"],
                "question": item["q"],
                "expected": item["expected"],
                "predicted": pred,
                "raw_answer": raw_ans,
                "correct": ok,
                "latent_steps": latent_steps,
                "time_s": round(avg_dt, 2),
            }
        )

    accuracy = correct / len(BENCHMARK)
    payload = {
        "accuracy": accuracy,
        "backend": backend,
        "batch_size": batch_size,
        "wall_time_s": round(total_dt, 2),
        "questions_per_sec": round(len(BENCHMARK) / total_dt, 2),
        "results": results,
    }
    print(f"\n=== Accuracy: {correct}/{len(BENCHMARK)} ({accuracy * 100:.1f}%) in {total_dt:.2f}s ===")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results to {out_json}")
    return payload


if __name__ == "__main__":
    main()
