"""5-Benchmark HuggingFace Evaluation Suite for HybridAgent on Google Cloud G4 GPU.

Benchmarks and Official Gemma 4 26B A4B Reference Targets:
  1. MMLU Pro (`mmlu_pro`, `TIGER-Lab/MMLU-Pro`, 10-choice A-J): 82.6%
  2. AIME 2026 No Tools (`aime_2026`, `MathArena/aime_2026`, integer 000-999): 88.3%
  3. GPQA Diamond (`gpqa_diamond`, `Idavidrein/gpqa` / `fingertap/GPQA-Diamond`, 4-choice A-D): 82.3%
  4. LiveCodeBench v6 (`livecodebench_v6`, `livecodebench/code_generation_lite` `release_v6`, pass@1): 77.1%
  5. MMMU Pro (`mmmu_pro`, `MMMU/MMMU_Pro` `standard (10 options)` & `vision`, 10-choice A-J): 73.8%
"""
from __future__ import annotations

import sys

from . import datasets, graders, prompts, runner
from .datasets import (
    BENCHMARK_NAMES,
    BENCHMARK_SPECS,
    REFERENCE_SCORES,
    BenchmarkItem,
    BenchmarkSpec,
    build_synthetic_benchmark_items,
    compress_lcb_private_tests,
    create_synthetic_benchmark_datasets,
    decode_lcb_private_tests,
    load_benchmark_dataset,
    normalize_aime_2026_item,
    normalize_gpqa_diamond_item,
    normalize_livecodebench_v6_item,
    normalize_mmlu_pro_item,
    normalize_mmmu_pro_item,
)
from .graders import (
    extract_code_block,
    extract_mcq_choice,
    grade_aime_2026,
    grade_benchmark_item,
    grade_gpqa_diamond,
    grade_livecodebench_v6,
    grade_mcq,
    grade_mmlu_pro,
    grade_mmmu_pro,
    normalize_aime_integer,
)
from .prompts import (
    BENCHMARK_SYSTEM_PROMPTS,
    NO_TOOLS_SYSTEM_PROMPT,
    build_benchmark_prompt,
    get_benchmark_system_prompt,
    get_benchmark_tool_policy,
)
from .runner import (
    BenchmarkResult,
    ItemEvalResult,
    SuiteEvalResult,
    evaluate_all_benchmarks,
    evaluate_benchmark,
    format_benchmark_markdown_report,
    format_benchmark_report_markdown,
    recommend_g4_concurrency,
)

if "hybrid_agent" in sys.modules:
    setattr(sys.modules["hybrid_agent"], "benchmarks", sys.modules[__name__])

__all__ = [
    "BENCHMARK_NAMES",
    "BENCHMARK_SPECS",
    "BENCHMARK_SYSTEM_PROMPTS",
    "NO_TOOLS_SYSTEM_PROMPT",
    "REFERENCE_SCORES",
    "BenchmarkItem",
    "BenchmarkResult",
    "BenchmarkSpec",
    "ItemEvalResult",
    "SuiteEvalResult",
    "build_benchmark_prompt",
    "build_synthetic_benchmark_items",
    "compress_lcb_private_tests",
    "create_synthetic_benchmark_datasets",
    "datasets",
    "decode_lcb_private_tests",
    "evaluate_all_benchmarks",
    "evaluate_benchmark",
    "extract_code_block",
    "extract_mcq_choice",
    "format_benchmark_markdown_report",
    "format_benchmark_report_markdown",
    "get_benchmark_system_prompt",
    "get_benchmark_tool_policy",
    "grade_aime_2026",
    "grade_benchmark_item",
    "grade_gpqa_diamond",
    "grade_livecodebench_v6",
    "grade_mcq",
    "grade_mmlu_pro",
    "grade_mmmu_pro",
    "graders",
    "load_benchmark_dataset",
    "normalize_aime_2026_item",
    "normalize_aime_integer",
    "normalize_gpqa_diamond_item",
    "normalize_livecodebench_v6_item",
    "normalize_mmlu_pro_item",
    "normalize_mmmu_pro_item",
    "prompts",
    "recommend_g4_concurrency",
    "runner",
]
