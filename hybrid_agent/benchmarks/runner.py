"""High-concurrency TensorRT benchmark runner, G4 GPU auto-tuner, and Markdown/JSON reporter."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import torch

from .datasets import (
    BENCHMARK_SPECS,
    REFERENCE_SCORES,
    build_synthetic_benchmark_items,
    load_benchmark_dataset,
)
from .graders import grade_benchmark_item
from .prompts import (
    build_benchmark_prompt,
    get_benchmark_tool_policy,
)


@dataclass
class ItemEvalResult:
    id: Any
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    latent_steps: int = 0
    tool_calls: int = 0
    python_calls: int = 0
    search_calls: int = 0
    output: str = ""
    options: list[str] | None = None
    subset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "correct": bool(self.correct),
            "latent_steps": self.latent_steps,
            "tool_calls": self.tool_calls,
            "python_calls": self.python_calls,
            "search_calls": self.search_calls,
            "output": self.output,
        }
        if self.subset is not None:
            d["subset"] = self.subset
        d.update(self.extra)
        return d


@dataclass
class BenchmarkResult:
    benchmark: str
    display_name: str
    total: int
    correct: int
    accuracy: float
    reference_score: float
    delta_vs_reference: float
    batch_size: int
    backend: str = "tensorrt"
    wall_time_s: float = 0.0
    questions_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    latent_steps_avg: float = 0.0
    total_latent_steps: int = 0
    total_tool_calls: int = 0
    peak_vram_gb: float = 0.0
    sub_scores: dict[str, float] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "display_name": self.display_name,
            "num_samples": self.total,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "reference_score": self.reference_score,
            "delta_vs_reference": self.delta_vs_reference,
            "batch_size": self.batch_size,
            "backend": self.backend,
            "wall_time_s": self.wall_time_s,
            "questions_per_sec": self.questions_per_sec,
            "tokens_per_sec": self.tokens_per_sec,
            "latent_steps_avg": self.latent_steps_avg,
            "total_latent_steps": self.total_latent_steps,
            "total_tool_calls": self.total_tool_calls,
            "peak_vram_gb": self.peak_vram_gb,
            "sub_scores": dict(self.sub_scores),
            "throughput": {
                "questions_per_sec": self.questions_per_sec,
                "tokens_per_sec": self.tokens_per_sec,
                "latent_steps_avg": self.latent_steps_avg,
                "wall_time_s": self.wall_time_s,
                "peak_vram_gb": self.peak_vram_gb,
            },
            "predictions": self.predictions,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class SuiteEvalResult(dict):
    """Suite evaluation result supporting both dict access (`summary['benchmarks']`) and attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"SuiteEvalResult has no attribute {key!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def recommend_g4_concurrency(
    model_id_or_path: str | Path,
    vram_gb: float = 96.0,
    quantized_4bit: bool | None = None,
) -> dict[str, Any]:
    """Recommend optimal TensorRT `batch_size` and `max_concurrency` for Google Cloud G4 GPU
    (`NVIDIA RTX PRO 6000 Blackwell`, 96 GB GDDR7 VRAM) or smaller GPUs.
    """
    s = str(model_id_or_path)
    p = Path(s)
    if quantized_4bit is None:
        if p.is_dir():
            from ..quantization import is_pack_quantized_dir

            quantized_4bit = is_pack_quantized_dir(p)
        else:
            quantized_4bit = "4bit" in s.lower() or "int4" in s.lower() or "w4a16" in s.lower()

    weight_vram_gb = 19.00 if quantized_4bit else 54.73
    runtime_overhead_gb = 4.5
    usable_kv_gb = max(1.0, float(vram_gb) - weight_vram_gb - runtime_overhead_gb)

    # Gemma 4 26B-A4B uses 25 sliding-window layers + 5 global GQA-2 layers (~0.30-0.35 GB per 4k-token trajectory)
    kv_per_slot_gb = 0.30 if quantized_4bit else 0.35
    raw_slots = max(1, int(usable_kv_gb / kv_per_slot_gb))

    if vram_gb >= 80.0:
        base_batch = (
            min(128, max(32, (raw_slots // 8) * 8))
            if quantized_4bit
            else min(64, max(32, (raw_slots // 8) * 8))
        )
    elif vram_gb >= 40.0:
        base_batch = 32 if quantized_4bit else 8
    else:
        base_batch = 16 if quantized_4bit else 1

    per_benchmark = {
        "mmlu_pro": base_batch,
        "aime_2026": max(1, base_batch // 2),
        "gpqa_diamond": max(1, base_batch // 2),
        "livecodebench_v6": max(1, base_batch // 2),
        "mmmu_pro": max(1, base_batch // 2),
    }
    return {
        "batch_size": base_batch,
        "max_concurrency": base_batch,
        "quantized_4bit": bool(quantized_4bit),
        "vram_gb": float(vram_gb),
        "weight_vram_gb": weight_vram_gb,
        "kv_headroom_gb": round(usable_kv_gb, 2),
        "per_benchmark": per_benchmark,
        **per_benchmark,
    }


def evaluate_benchmark(
    benchmark_name: str,
    engine: Any,
    *,
    dataset_items: Sequence[dict[str, Any]] | None = None,
    dataset_dir: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    batch_size: int = 32,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Evaluate `engine` (`HybridEngine`) on one of the 5 benchmarks (`mmlu_pro`, `aime_2026`,
    `gpqa_diamond`, `livecodebench_v6`, `mmmu_pro`) using high-concurrency `engine.run_batch`.
    """
    if benchmark_name not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark {benchmark_name!r}; expected one of {list(BENCHMARK_SPECS)}")
    spec = BENCHMARK_SPECS[benchmark_name]
    items = load_benchmark_dataset(
        benchmark_name,
        dataset_items=dataset_items,
        dataset_dir=dataset_dir,
        split=split,
        limit=limit,
        seed=seed,
    )
    prompts = [build_benchmark_prompt(benchmark_name, it) for it in items]
    images = [it.get("images") for it in items] if spec.multimodal else None

    run_overrides = get_benchmark_tool_policy(benchmark_name)

    t0 = time.time()
    trajectories = engine.run_batch(
        prompts,
        images=images,
        batch_size=batch_size,
        max_concurrency=batch_size,
        **run_overrides,
    )
    wall = max(1e-6, time.time() - t0)

    predictions: list[dict[str, Any]] = []
    correct_count = 0
    total_gen_steps = 0
    total_latent_steps = 0
    total_tool_calls = 0
    subset_counts: dict[str, list[int]] = {}

    for item, traj in zip(items, trajectories):
        raw_res = getattr(traj, "raw_result", {}) or {}
        out_text = getattr(traj, "output", "") or ""
        lat_steps = int(getattr(traj, "latent_steps", raw_res.get("latent_steps", 0)))
        gen_steps = int(raw_res.get("generation_steps", max(1, lat_steps + len(out_text) // 4)))
        py_calls = int(raw_res.get("python_calls", 0))
        sr_calls = int(raw_res.get("search_calls", 0))
        t_calls = py_calls + sr_calls

        total_gen_steps += gen_steps
        total_latent_steps += lat_steps
        total_tool_calls += t_calls

        ok, pred, extra_meta = grade_benchmark_item(benchmark_name, out_text, item)
        gold = "pass@1" if benchmark_name == "livecodebench_v6" else str(item.get("answer", "")).strip()

        correct_count += int(ok)
        sub_name = item.get("subset") or item.get("category")
        if sub_name:
            counts = subset_counts.setdefault(str(sub_name), [0, 0])
            counts[0] += int(ok)
            counts[1] += 1

        predictions.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "options": item.get("options"),
                "ground_truth": gold,
                "prediction": pred,
                "correct": bool(ok),
                "latent_steps": lat_steps,
                "tool_calls": t_calls,
                "python_calls": py_calls,
                "search_calls": sr_calls,
                "output": out_text,
                **({"subset": item.get("subset")} if item.get("subset") else {}),
                **extra_meta,
            }
        )

    total = len(items)
    sub_scores = {
        k: round((v[0] / v[1]) * 100.0, 2) for k, v in subset_counts.items() if v[1] > 0
    }
    acc_pct = round((correct_count / total) * 100.0, 2) if total > 0 else 0.0
    ref = REFERENCE_SCORES[benchmark_name]
    vram_gb = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

    result = BenchmarkResult(
        benchmark=benchmark_name,
        display_name=spec.display_name,
        total=total,
        correct=correct_count,
        accuracy=acc_pct,
        reference_score=ref,
        delta_vs_reference=round(acc_pct - ref, 2),
        batch_size=int(batch_size),
        wall_time_s=round(wall, 4),
        questions_per_sec=round(total / wall, 2) if total > 0 else 0.0,
        tokens_per_sec=round(total_gen_steps / wall, 2) if total_gen_steps > 0 else 0.0,
        latent_steps_avg=round(total_latent_steps / total, 2) if total > 0 else 0.0,
        total_latent_steps=total_latent_steps,
        total_tool_calls=total_tool_calls,
        peak_vram_gb=round(vram_gb, 3),
        sub_scores=sub_scores,
        predictions=predictions,
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def format_benchmark_report_markdown(summary: dict[str, Any]) -> str:
    """Format a Markdown benchmark comparison table comparing achieved scores against the 5 reference targets."""
    model_name = summary.get("model", "milkshake721/Gemma-4-26B-A4B-HybridAgent")
    backend = summary.get("backend", "tensorrt")
    batch_size = summary.get("batch_size", 32)
    benchmarks_map = summary.get("benchmarks", {})

    md_rows: list[str] = [
        "# HybridAgent 5-Benchmark Evaluation Report (G4 GPU TensorRT)",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Backend**: `{backend}`",
        f"- **Batch Size**: `{batch_size}`",
        "",
        "| Benchmark | Achieved Score | Target Reference | Delta | Samples | Throughput (q/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b_name, b_data in benchmarks_map.items():
        disp = b_data.get("display_name", b_name)
        acc = float(b_data.get("accuracy", 0.0))
        ref = float(b_data.get("reference_score", REFERENCE_SCORES.get(b_name, 0.0)))
        delta = float(b_data.get("delta_vs_reference", acc - ref))
        tot = int(b_data.get("total", 0))
        qps = float(b_data.get("questions_per_sec", 0.0))
        md_rows.append(
            f"| {disp} (`{b_name}`) | {acc:.1f}% | {ref:.1f}% | {delta:+.1f}% | {tot} | {qps:.2f} |"
        )
    return "\n".join(md_rows) + "\n"


format_benchmark_markdown_report = format_benchmark_report_markdown


def evaluate_all_benchmarks(
    engine: Any,
    *,
    benchmarks: Sequence[str] | None = None,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 32,
    synthetic_smoke: bool = False,
    seed: int = 42,
    model_name: str = "milkshake721/Gemma-4-26B-A4B-HybridAgent",
    backend: str = "tensorrt",
) -> SuiteEvalResult:
    """Evaluate `engine` across all selected benchmarks and write JSON + Markdown reports."""
    bench_list: list[str] = []
    for b in (benchmarks or ["all"]):
        for sub in str(b).split(","):
            sub = sub.strip()
            if sub == "all":
                for k in BENCHMARK_SPECS:
                    if k not in bench_list:
                        bench_list.append(k)
            elif sub in BENCHMARK_SPECS and sub not in bench_list:
                bench_list.append(sub)
    if not bench_list:
        bench_list = list(BENCHMARK_SPECS.keys())

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_map = build_synthetic_benchmark_items() if synthetic_smoke else {}
    results_dict: dict[str, Any] = {}

    for b_name in bench_list:
        ds_items = synthetic_map.get(b_name) if synthetic_smoke else None
        per_out = (out_dir / f"{b_name}.json") if out_dir is not None else None
        res = evaluate_benchmark(
            b_name,
            engine,
            dataset_items=ds_items,
            dataset_dir=dataset_dir,
            limit=limit,
            batch_size=int(batch_size),
            output_path=per_out,
            seed=int(seed),
        )
        res.backend = backend
        if per_out is not None:
            per_out.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        results_dict[b_name] = res.to_dict()

    summary_payload = SuiteEvalResult(
        {
            "model": str(model_name),
            "backend": str(backend),
            "batch_size": int(batch_size),
            "reference_scores": dict(REFERENCE_SCORES),
            "g4_concurrency": recommend_g4_concurrency(model_name, vram_gb=96.0),
            "benchmarks": results_dict,
            **results_dict,
        }
    )

    if out_dir is not None:
        for fname in ("summary.json", "benchmark_results.json"):
            (out_dir / fname).write_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        md_text = format_benchmark_report_markdown(summary_payload)
        for mname in ("report.md", "SUMMARY.md"):
            (out_dir / mname).write_text(md_text, encoding="utf-8")

    return summary_payload
