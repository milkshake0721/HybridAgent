#!/usr/bin/env python3
"""Unified High-Concurrency TensorRT CLI for the 5-Benchmark HuggingFace Evaluation Suite on G4 GPU.

Evaluates `milkshake721/Gemma-4-26B-A4B-HybridAgent` (BF16) and
`milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit` (W4A16) on Google Cloud G4 GPU
(`NVIDIA RTX PRO 6000 Blackwell`, 96 GB GDDR7 VRAM) across:
  - MMLU Pro (`mmlu_pro`): Target 82.6%
  - AIME 2026 (No Tools) (`aime_2026`): Target 88.3%
  - GPQA Diamond (`gpqa_diamond`): Target 82.3%
  - LiveCodeBench v6 (`livecodebench_v6`): Target 77.1%
  - MMMU Pro (多模態) (`mmmu_pro`): Target 73.8%

Example:
  python3 eval_hf_benchmarks.py \
    --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
    --backend tensorrt \
    --benchmarks all \
    --batch-size 32 \
    --output-dir outputs/benchmarks
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_agent import HybridEngine, load_hybrid_agent
from hybrid_agent.benchmarks import (
    BENCHMARK_SPECS,
    evaluate_all_benchmarks,
    recommend_g4_concurrency,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-Benchmark HuggingFace Evaluation Suite on G4 GPU")
    p.add_argument(
        "--model",
        default="milkshake721/Gemma-4-26B-A4B-HybridAgent",
        help="Model repo ID or local checkpoint directory",
    )
    p.add_argument(
        "--backend",
        choices=("tensorrt", "eager", "sglang"),
        default="tensorrt",
        help="Execution backend (default: tensorrt)",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to run: 'all' or subset of {mmlu_pro,aime_2026,gpqa_diamond,livecodebench_v6,mmmu_pro}",
    )
    p.add_argument(
        "--batch-size",
        "--batch_size",
        "--concurrency",
        dest="batch_size",
        type=int,
        default=32,
        help="Concurrent question batch size on G4 GPU (default: 32)",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional max items per benchmark")
    p.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        dest="dataset_dir",
        default=None,
        help="Optional local directory containing <benchmark>.jsonl files",
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="outputs/benchmarks",
        help="Directory for per-benchmark JSON, summary.json, and report.md",
    )
    p.add_argument(
        "--synthetic-smoke",
        "--synthetic_smoke",
        "--dry-run",
        "--dry_run",
        dest="synthetic_smoke",
        action="store_true",
        help="Run using built-in synthetic benchmark items (offline smoke test)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling/sampling")
    p.add_argument("--vram-gb", "--vram_gb", dest="vram_gb", type=float, default=96.0, help="Target GPU VRAM in GB")
    p.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=4096)
    p.add_argument("--max-latent-steps", "--max_latent_steps", dest="max_latent_steps", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args(argv)


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    g4_rec = recommend_g4_concurrency(args.model, vram_gb=float(getattr(args, "vram_gb", 96.0)))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(g4_rec["batch_size"])

    model, tokenizer = load_hybrid_agent(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        backend=args.backend,
        max_batch_size=max(32, batch_size, int(g4_rec["max_concurrency"])),
        load_processor=True,
    )
    engine = HybridEngine(
        model,
        tokenizer,
        processor=getattr(model, "processor", None),
        max_new_tokens=int(getattr(args, "max_new_tokens", 4096)),
        max_latent_steps=int(getattr(args, "max_latent_steps", 32)),
        temperature=float(getattr(args, "temperature", 0.0)),
        default_batch_size=batch_size,
    )

    summary = evaluate_all_benchmarks(
        engine,
        benchmarks=args.benchmarks,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=batch_size,
        synthetic_smoke=bool(args.synthetic_smoke),
        seed=int(args.seed),
        model_name=str(args.model),
        backend=str(args.backend),
    )
    return dict(summary)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_suite(args)
    print(f"=== Completed 5-Benchmark Suite ({len(summary.get('benchmarks', {}))} benchmarks) ===")
    for b_name, b_data in summary.get("benchmarks", {}).items():
        spec = BENCHMARK_SPECS.get(b_name)
        disp = spec.display_name if spec else b_name
        print(
            f"  • {disp} ({b_name}): {b_data['accuracy']:.1f}% "
            f"(target: {b_data['reference_score']:.1f}%, delta: {b_data['delta_vs_reference']:+.1f}%, "
            f"{b_data['questions_per_sec']:.2f} q/s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
