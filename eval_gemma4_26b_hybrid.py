#!/usr/bin/env python3
"""Evaluate Gemma-4-26B-A4B-HybridAgent (BF16 or 4-bit) on the 10-question math/logic benchmark."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

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


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    out_json = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    backend = sys.argv[3] if len(sys.argv) > 3 else "tensorrt"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    model, tokenizer = load_hybrid_agent(source, dtype=torch.bfloat16, device="cuda", backend=backend)
    engine = HybridEngine(model, tokenizer, max_new_tokens=1024, max_latent_steps=64, temperature=0.0, top_p=1.0)

    results = []
    correct = 0
    for item in BENCHMARK:
        t0 = time.time()
        traj = engine.run(item["q"], system_prompt=SYSTEM_PROMPT)
        dt = time.time() - t0
        raw_ans = traj.final_answer or extract_answer(traj.output)
        pred = normalize_math_answer(raw_ans)
        ok = pred == item["expected"]
        correct += int(ok)
        latent_steps = sum(r.latent_steps for r in traj.rounds)
        print(f"[Q{item['id']:02d}] {'PASS' if ok else 'FAIL'} | expected={item['expected']} pred={pred} (raw={raw_ans!r}) | latent_steps={latent_steps} | {dt:.1f}s")
        results.append({
            "id": item["id"],
            "question": item["q"],
            "expected": item["expected"],
            "predicted": pred,
            "raw_answer": raw_ans,
            "correct": ok,
            "latent_steps": latent_steps,
            "time_s": round(dt, 2),
        })

    print(f"\n=== Accuracy: {correct}/{len(BENCHMARK)} ({correct / len(BENCHMARK) * 100:.1f}%) ===")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"accuracy": correct / len(BENCHMARK), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results to {out_json}")


if __name__ == "__main__":
    main()
