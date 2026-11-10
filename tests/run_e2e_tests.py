#!/usr/bin/env python3
"""Unified 4-Tier End-to-End Test Runner for HybridAgent.

Usage:
  python3 tests/run_e2e_tests.py --tier 1
  python3 tests/run_e2e_tests.py --tier 2
  python3 tests/run_e2e_tests.py --tier 3
  python3 tests/run_e2e_tests.py --tier 4
  python3 tests/run_e2e_tests.py --tier all --json-report outputs/e2e_report.json
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_SITE = Path("/home/antigravity/.local/lib/python3.11/site-packages")

for _p in (str(PROJECT_ROOT), str(USER_SITE)):
    if _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

from tests.e2e.helpers import ensure_contract_bridges, install_runtime_compat

TIER_MODULES: dict[str, tuple[str, str, str]] = {
    "1": ("Tier 1 — Feature E2E (F1..F8)", "tests.e2e.test_tier1_features", "test_t1_"),
    "2": ("Tier 2 — Boundary & Negative E2E", "tests.e2e.test_tier2_boundaries", "test_t2_"),
    "3": ("Tier 3 — Cross-Feature Combinations", "tests.e2e.test_tier3_combinations", "test_t3_"),
    "4": ("Tier 4 — E2E Business Scenarios", "tests.e2e.test_tier4_scenarios", "test_t4_"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HybridAgent 4-Tier E2E Test Runner")
    p.add_argument(
        "--tier",
        choices=("1", "2", "3", "4", "all"),
        default="all",
        help="Which E2E tier to execute ('1', '2', '3', '4', or 'all').",
    )
    p.add_argument(
        "--filter",
        "-k",
        dest="filter_expr",
        default=None,
        help="Optional substring filter on test function names.",
    )
    p.add_argument(
        "--json-report",
        dest="json_report",
        default=None,
        help="Optional path to write structured JSON test execution report.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every individual test case result and duration.",
    )
    p.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop execution on the first test failure.",
    )
    return p.parse_args(argv)


def run_tier(
    tier_id: str,
    *,
    filter_expr: str | None = None,
    verbose: bool = False,
    failfast: bool = False,
) -> dict[str, Any]:
    title, mod_name, prefix = TIER_MODULES[tier_id]
    ensure_contract_bridges()
    mod = importlib.import_module(mod_name)

    test_fns = [
        (name, obj)
        for name, obj in sorted(vars(mod).items())
        if name.startswith(prefix) and callable(obj)
    ]
    if filter_expr:
        test_fns = [(n, fn) for n, fn in test_fns if filter_expr in n]

    passed = 0
    failed = 0
    details: list[dict[str, Any]] = []
    t_start = time.time()

    print(f"\n=== {title} ({len(test_fns)} tests) ===", flush=True)
    for name, fn in test_fns:
        ensure_contract_bridges()
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            passed += 1
            details.append({"name": name, "status": "PASSED", "duration_s": round(dt, 4)})
            if verbose:
                print(f"  [PASS] {name} ({dt:.3f}s)", flush=True)
        except Exception as exc:
            dt = time.time() - t0
            failed += 1
            tb_str = traceback.format_exc()
            details.append(
                {
                    "name": name,
                    "status": "FAILED",
                    "duration_s": round(dt, 4),
                    "error": str(exc),
                    "traceback": tb_str,
                }
            )
            print(f"  [FAIL] {name} ({dt:.3f}s): {exc}", flush=True)
            print(tb_str, flush=True)
            if failfast:
                break

    tier_wall = time.time() - t_start
    print(
        f"--> Tier {tier_id} Summary: {passed}/{len(test_fns)} PASSED, {failed} FAILED in {tier_wall:.2f}s",
        flush=True,
    )
    return {
        "tier": tier_id,
        "title": title,
        "total": len(test_fns),
        "passed": passed,
        "failed": failed,
        "duration_s": round(tier_wall, 3),
        "tests": details,
    }


def main(argv: list[str] | None = None) -> int:
    install_runtime_compat()
    ensure_contract_bridges()
    args = parse_args(argv)

    tiers_to_run = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]
    suite_start = time.time()
    tier_reports: list[dict[str, Any]] = []

    for tid in tiers_to_run:
        rep = run_tier(
            tid,
            filter_expr=args.filter_expr,
            verbose=args.verbose,
            failfast=args.failfast,
        )
        tier_reports.append(rep)
        if args.failfast and rep["failed"] > 0:
            break

    total_wall = time.time() - suite_start
    total_tests = sum(r["total"] for r in tier_reports)
    total_passed = sum(r["passed"] for r in tier_reports)
    total_failed = sum(r["failed"] for r in tier_reports)

    print("\n======================================================================")
    print("                HYBRIDAGENT E2E TEST SUITE SUMMARY                    ")
    print("======================================================================")
    for r in tier_reports:
        status_tag = "PASS" if r["failed"] == 0 else "FAIL"
        print(
            f"  [{status_tag}] {r['title']:<42} | {r['passed']:>3}/{r['total']:<3} passed | {r['duration_s']:>6.2f}s"
        )
    print("----------------------------------------------------------------------")
    overall_tag = "ALL PASSED" if total_failed == 0 else f"FAILED ({total_failed} failures)"
    print(
        f"  OVERALL: {overall_tag} | {total_passed}/{total_tests} passed in {total_wall:.2f}s"
    )
    print("======================================================================", flush=True)

    if args.json_report:
        out_path = Path(args.json_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "PASSED" if total_failed == 0 else "FAILED",
            "total": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "duration_s": round(total_wall, 3),
            "tiers": tier_reports,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
