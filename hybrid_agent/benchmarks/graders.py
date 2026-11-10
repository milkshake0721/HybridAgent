"""Deterministic answer extractors and graders for the 5 HuggingFace benchmarks."""
from __future__ import annotations

import math
import re
import subprocess
import sys
from typing import Any

from ..answer import extract_answer
from ..constants import ANSWER_CLOSE, ANSWER_OPEN


def extract_mcq_choice(text: str, num_choices: int = 10) -> str:
    """Extract a single multiple-choice letter in `A`..`chr(65 + num_choices - 1)` from model output."""
    max_ch = chr(65 + max(1, num_choices) - 1)
    valid_letters = f"A-{max_ch}"
    cand = extract_answer(text).strip()
    if cand:
        cand_clean = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", cand).strip()
        cand_clean = cand_clean.strip("()[]{}.: ")
        if len(cand_clean) == 1 and "A" <= cand_clean.upper() <= max_ch:
            return cand_clean.upper()
        m_lead = re.match(rf"^\(?([{valid_letters}])\)?(?:[\s.:\-]|$)", cand_clean, flags=re.IGNORECASE)
        if m_lead:
            return m_lead.group(1).upper()
        m_box = re.search(rf"\b([{valid_letters}])\b", cand_clean)
        if m_box:
            return m_box.group(1).upper()
        m_box_ci = re.search(rf"\b([{valid_letters}])\b", cand_clean, flags=re.IGNORECASE)
        if m_box_ci and len(cand_clean) <= 3:
            return m_box_ci.group(1).upper()

    # Check unclosed \boxed{X at end of text
    m_unclosed = re.findall(r"\\boxed\{([^}]*)$", text)
    if m_unclosed:
        u = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", m_unclosed[-1]).strip()
        u = u.strip("()[]{}.: ")
        if len(u) == 1 and "A" <= u.upper() <= max_ch:
            return u.upper()

    patterns = [
        rf"(?:answer|choice|option)\s*(?:is|:)\s*\*?\*?\(?([{valid_letters}])\)?\*?\*?",
        rf"\(([{valid_letters}])\)\s*\.?\s*$",
        rf"\b([{valid_letters}])\s*\.?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).upper()
    if cand and len(cand) == 1 and "A" <= cand.upper() <= max_ch:
        return cand.upper()
    return ""


def grade_mcq(output_or_pred: str, gold: str, *, num_choices: int = 10) -> tuple[bool, str]:
    """Grade a multiple-choice output against `gold` letter."""
    pred = extract_mcq_choice(output_or_pred, num_choices=num_choices)
    gold_clean = str(gold).strip().upper()
    return (pred == gold_clean and bool(gold_clean)), pred


def grade_mmlu_pro(output_or_pred: str, gold: str, *, num_choices: int = 10) -> tuple[bool, str]:
    return grade_mcq(output_or_pred, gold, num_choices=num_choices)


def grade_gpqa_diamond(output_or_pred: str, gold: str, *, num_choices: int = 4) -> tuple[bool, str]:
    return grade_mcq(output_or_pred, gold, num_choices=num_choices)


def grade_mmmu_pro(output_or_pred: str, gold: str, *, num_choices: int = 10) -> tuple[bool, str]:
    return grade_mcq(output_or_pred, gold, num_choices=num_choices)


def normalize_aime_integer(val_str: str) -> int | None:
    """Normalize an AIME answer string (including leading zeros `007`, LaTeX fractions, or powers) to an int."""
    if val_str is None:
        return None
    s = str(val_str).strip()
    if not s:
        return None
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", s).strip()
    if s.startswith("\\boxed{") and not s.endswith("}"):
        s = s[7:].strip()
    elif s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1].strip()
    s = s.replace(",", "")
    s_expr = s.replace("\\times", "*").replace("\\cdot", "*").replace("^", "**")
    s_expr = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s_expr)
    if re.fullmatch(r"[0-9+\-*/().\s]+", s_expr):
        # Handle leading zeros like "007" or "042" or "000" before Python eval (which rejects 007 in Python 3)
        s_expr_no_lz = re.sub(r"\b0+(\d)", r"\1", s_expr)
        try:
            val = eval(compile(s_expr_no_lz, "<aime>", "eval"), {"__builtins__": {}}, {})
            if isinstance(val, (int, float)) and math.isfinite(val) and abs(val - round(val)) < 1e-6:
                return int(round(val))
        except Exception:
            pass
    nums = re.findall(r"-?\d+", s)
    if nums:
        return int(nums[-1])
    return None


def grade_aime_2026(output_or_pred: str, gold: str) -> tuple[bool, str]:
    """Grade an AIME 2026 response, returning `(is_correct, normalized_3digit_str)`."""
    raw = (
        extract_answer(output_or_pred)
        if ("<answer>" in output_or_pred or "\\boxed" in output_or_pred)
        else output_or_pred
    )
    if not raw and "\\boxed{" in output_or_pred:
        raw = output_or_pred.split("\\boxed{")[-1].rstrip("}")
    pred_int = normalize_aime_integer(raw or output_or_pred)
    gold_int = normalize_aime_integer(gold)
    if pred_int is not None and gold_int is not None:
        formatted = f"{pred_int:03d}" if 0 <= pred_int <= 999 else str(pred_int)
        return (pred_int == gold_int), formatted

    # Fallback symbolic math_verify comparison if available
    try:
        from hybrid_rl.reward import math_verify_equal

        if raw and math_verify_equal(str(raw), str(gold)):
            formatted = f"{gold_int:03d}" if (gold_int is not None and 0 <= gold_int <= 999) else str(raw).strip()
            return True, formatted
    except Exception:
        pass

    return (str(raw).strip() == str(gold).strip() and bool(str(gold).strip())), str(raw).strip()


def extract_code_block(text: str) -> str:
    """Extract candidate Python solution from `<answer>` (ignoring `<think>` / `<python>` tool blocks)."""
    target = text
    if ANSWER_OPEN in text:
        end_idx = text.rfind(ANSWER_CLOSE)
        start_idx = text.rfind(ANSWER_OPEN, 0, end_idx if end_idx != -1 else len(text))
        if start_idx != -1:
            target = text[start_idx + len(ANSWER_OPEN) : (end_idx if end_idx != -1 else len(text))]
    else:
        target = re.sub(r"<think>.*?</think>", "", target, flags=re.DOTALL)
        target = re.sub(r"<python>.*?</python>", "", target, flags=re.DOTALL)
        target = re.sub(r"<result>.*?</result>", "", target, flags=re.DOTALL)

    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", target, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    m_unclosed = re.search(r"```(?:python|py)?\s*\n(.*)$", target, flags=re.DOTALL | re.IGNORECASE)
    if m_unclosed:
        return m_unclosed.group(1).strip()
    return target.strip()


def grade_livecodebench_v6(
    output_text: str,
    item: dict[str, Any],
    *,
    timeout_per_test: float = 1.5,
) -> tuple[bool, str, dict[str, Any]]:
    """Execute extracted candidate code against LiveCodeBench v6 public + private test cases in a subprocess."""
    code = extract_code_block(output_text)
    test_cases = item.get("test_cases") or []
    fn_name = item.get("fn_name")
    if not code:
        return False, "", {"reason": "empty_code", "passed_tests": 0, "total_tests": len(test_cases)}
    if not test_cases:
        return True, code, {"reason": "no_tests", "passed_tests": 0, "total_tests": 0}

    passed_count = 0
    for tc in test_cases:
        tc_in = str(tc.get("input", ""))
        tc_out = str(tc.get("output", "")).strip()
        is_func = tc.get("testtype") == "functional" or bool(fn_name)
        if is_func and fn_name:
            arg_lines = [line for line in tc_in.splitlines() if line.strip()]
            harness = (
                "from typing import *\n"
                "import collections, heapq, itertools, math, functools, bisect\n\n"
                f"{code}\n\n"
                "import json, ast\n"
                "def _parse_val(s):\n"
                "    try:\n"
                "        return ast.literal_eval(s)\n"
                "    except Exception:\n"
                "        return json.loads(s)\n"
                f"_args = [_parse_val(line) for line in {arg_lines!r}]\n"
                f"_expected = _parse_val({tc_out!r})\n"
                "if 'Solution' in globals():\n"
                f"    _res = getattr(Solution(), {fn_name!r})(*_args)\n"
                "else:\n"
                f"    _res = globals()[{fn_name!r}](*_args)\n"
                "assert _res == _expected, f'Expected {_expected}, got {_res}'\n"
                "print('LCB_PASS')\n"
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", harness],
                    input="",
                    capture_output=True,
                    text=True,
                    timeout=timeout_per_test,
                )
                if proc.returncode == 0 and "LCB_PASS" in proc.stdout:
                    passed_count += 1
                else:
                    return False, code, {
                        "reason": "wrong_answer",
                        "stderr": proc.stderr[:300],
                        "passed_tests": passed_count,
                        "total_tests": len(test_cases),
                    }
            except subprocess.TimeoutExpired:
                return False, code, {
                    "reason": "timeout",
                    "passed_tests": passed_count,
                    "total_tests": len(test_cases),
                }
        else:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    input=tc_in,
                    capture_output=True,
                    text=True,
                    timeout=timeout_per_test,
                )
                out_lines = [line.rstrip() for line in proc.stdout.strip().splitlines()]
                exp_lines = [line.rstrip() for line in tc_out.splitlines()]
                if proc.returncode == 0 and (proc.stdout.strip() == tc_out or out_lines == exp_lines):
                    passed_count += 1
                else:
                    return False, code, {
                        "reason": "wrong_answer",
                        "stdout": proc.stdout[:200],
                        "stderr": proc.stderr[:200],
                        "passed_tests": passed_count,
                        "total_tests": len(test_cases),
                    }
            except subprocess.TimeoutExpired:
                return False, code, {
                    "reason": "timeout",
                    "passed_tests": passed_count,
                    "total_tests": len(test_cases),
                }

    return True, code, {"reason": "passed", "passed_tests": passed_count, "total_tests": len(test_cases)}


def grade_benchmark_item(
    benchmark_name: str,
    output_text: str,
    item: dict[str, Any],
    *,
    timeout_per_test: float = 1.5,
) -> tuple[bool, str, dict[str, Any]]:
    """Unified dispatcher for grading a single item from any of the 5 benchmarks."""
    if benchmark_name == "mmlu_pro":
        num_c = max(10, len(item.get("options") or []))
        ok, pred = grade_mmlu_pro(output_text, str(item.get("answer", "")), num_choices=num_c)
        return ok, pred, {}
    if benchmark_name == "gpqa_diamond":
        num_c = max(4, len(item.get("options") or []))
        ok, pred = grade_gpqa_diamond(output_text, str(item.get("answer", "")), num_choices=num_c)
        return ok, pred, {}
    if benchmark_name == "mmmu_pro":
        num_c = max(10, len(item.get("options") or []))
        ok, pred = grade_mmmu_pro(output_text, str(item.get("answer", "")), num_choices=num_c)
        return ok, pred, {}
    if benchmark_name == "aime_2026":
        ok, pred = grade_aime_2026(output_text, str(item.get("answer", "")))
        return ok, pred, {}
    if benchmark_name == "livecodebench_v6":
        return grade_livecodebench_v6(output_text, item, timeout_per_test=timeout_per_test)
    pred = extract_answer(output_text)
    gold = str(item.get("answer", "")).strip()
    return (pred == gold), pred, {}
