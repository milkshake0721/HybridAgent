"""Benchmark-specific prompt builders and tool policies for the 5 HuggingFace benchmarks."""
from __future__ import annotations

from typing import Any

from ..prompts import SYSTEM_PROMPT
from .datasets import BENCHMARK_SPECS

NO_TOOLS_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant solving competition problems WITHOUT external tools.\n"
    "Do NOT invoke <python> or <search> tags under any circumstances.\n"
    "Reason step by step inside <think>...</think> (using <latent>...</latent> for internal continuous reasoning) "
    "and place your final integer answer (0-999) inside <answer>\\boxed{...}</answer>."
)

BENCHMARK_SYSTEM_PROMPTS: dict[str, str] = {
    "mmlu_pro": SYSTEM_PROMPT,
    "aime_2026": NO_TOOLS_SYSTEM_PROMPT,
    "gpqa_diamond": SYSTEM_PROMPT,
    "livecodebench_v6": SYSTEM_PROMPT,
    "mmmu_pro": SYSTEM_PROMPT,
}


def get_benchmark_system_prompt(benchmark_name: str, *, allow_tools: bool | None = None) -> str:
    """Return the system prompt for `benchmark_name` (`NO_TOOLS_SYSTEM_PROMPT` when tools are disabled)."""
    spec = BENCHMARK_SPECS.get(benchmark_name)
    tools_enabled = allow_tools if allow_tools is not None else (spec.allow_tools if spec is not None else True)
    if not tools_enabled or benchmark_name == "aime_2026":
        return NO_TOOLS_SYSTEM_PROMPT
    return BENCHMARK_SYSTEM_PROMPTS.get(benchmark_name, SYSTEM_PROMPT)


def get_benchmark_tool_policy(
    benchmark_name: str,
    *,
    allow_tools_override: bool | None = None,
) -> dict[str, Any]:
    """Return `HybridEngine.run_batch` overrides enforcing benchmark tool permissions."""
    spec = BENCHMARK_SPECS.get(benchmark_name)
    allow_tools = (
        allow_tools_override
        if allow_tools_override is not None
        else (spec.allow_tools if spec is not None else True)
    )
    if not allow_tools or benchmark_name == "aime_2026":
        return {
            "system_prompt": NO_TOOLS_SYSTEM_PROMPT,
            "max_tool_calls": 0,
            "max_python_calls": 0,
            "max_search_calls": 0,
            "terminate_on_tool_cap": True,
        }
    return {}


def build_benchmark_prompt(benchmark_name: str, item: dict[str, Any]) -> str:
    """Format a normalized benchmark item into a model user prompt."""
    q = str(item.get("question", ""))
    opts = item.get("options") or []
    if benchmark_name in ("mmlu_pro", "gpqa_diamond", "mmmu_pro") and opts:
        opt_lines = [f"({chr(65 + i)}) {opt}" for i, opt in enumerate(opts)]
        max_letter = chr(65 + len(opts) - 1)
        return (
            f"{q}\n\nOptions:\n"
            + "\n".join(opt_lines)
            + f"\n\nPut the single correct option letter (A-{max_letter}) inside \\boxed{{}}."
        )
    if benchmark_name == "aime_2026":
        return f"{q}\n\nSolve without any tools and put the final integer answer (0-999) inside \\boxed{{}}."
    if benchmark_name == "livecodebench_v6":
        starter = str(item.get("starter_code") or "").strip()
        if starter:
            return (
                f"{q}\n\nStarter Code:\n```python\n{starter}\n```\n"
                "Provide the complete Python solution inside a ```python ... ``` block."
            )
        return (
            f"{q}\n\n"
            "Provide a complete Python 3 solution reading from stdin and printing to stdout "
            "inside a ```python ... ``` block."
        )
    return q
