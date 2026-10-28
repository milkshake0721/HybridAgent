#!/usr/bin/env python3
"""Verify local or Hugging Face Hub weights for Gemma-4-26B-A4B-HybridAgent (BF16 & 4-bit)
with optional TensorRT (`--backend tensorrt`) acceleration."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_agent.engine import HybridEngine
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.prompts import SYSTEM_PROMPT

DEFAULT_REPO = "milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit"


def main():
    p = argparse.ArgumentParser(description="Test Gemma-4-26B-A4B-HybridAgent (BF16 or 4-bit)")
    p.add_argument("source", nargs="?", default=DEFAULT_REPO, help="Local dir or HF repo ID")
    p.add_argument("--backend", default="tensorrt", choices=("tensorrt", "eager"), help="Backend mode")
    args = p.parse_args()

    print(f"=== Loading Gemma-4-26B-A4B-HybridAgent from: {args.source} (backend={args.backend!r}) ===")
    t0 = time.time()
    model, tokenizer = load_hybrid_agent(
        args.source,
        dtype=torch.bfloat16,
        device="cuda",
        backend=args.backend,
    )

    embed_layer = model.backbone.get_input_embeddings()
    embed_scale = getattr(embed_layer, "embed_scale", None)
    acc_info = getattr(model, "_acceleration_info", None)
    print(
        f"Loaded in {time.time() - t0:.1f}s! Vocab size: {len(tokenizer)}, "
        f"embed_scale: {float(embed_scale) if embed_scale is not None else None:.4f}"
    )
    print(f"GPU VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    if acc_info:
        print(f"Acceleration active: {acc_info}")

    engine = HybridEngine(
        model,
        tokenizer,
        max_new_tokens=1024,
        max_latent_steps=64,
        temperature=0.0,
        top_p=1.0,
    )

    questions = [
        "If 3x + 7 = 22, what is the value of 5x - 2?",
        "A rectangle has a perimeter of 30 cm and an area of 54 cm^2. What are the lengths of its sides?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\n--- Test {i}: {q} ---")
        t_q = time.time()
        traj = engine.run(q, system_prompt=SYSTEM_PROMPT)
        dt = time.time() - t_q
        for r_idx, r in enumerate(traj.rounds):
            print(f"[Round {r_idx}] latent_steps={r.latent_steps} | text={r.text!r}")
        print(f"Final Answer: {traj.final_answer!r} ({dt:.2f}s)")


if __name__ == "__main__":
    main()
