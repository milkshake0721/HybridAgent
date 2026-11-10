"""Milestone M3 Unit Test Suite (`tests/unit/test_m3_benchmarks.py` — 20 Tests).

Covers Features F7 (High-Concurrency Batch Inference CLI & Runners) and F8 (5-Benchmark
HuggingFace Evaluation Suite on G4 GPU: MMLU Pro, AIME 2026 No Tools, GPQA Diamond,
LiveCodeBench v6, and MMMU Pro).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from PIL import Image

from tests.e2e.helpers import (
    PROJECT_ROOT,
    build_synthetic_benchmark_items,
    build_tiny_hybrid_model,
    compress_lcb_private_tests,
    create_synthetic_benchmark_datasets,
    create_temp_checkpoint_dir,
    ensure_contract_bridges,
    install_runtime_compat,
    run_cli_command,
)

install_runtime_compat()
ensure_contract_bridges()

import eval_gemma4_26b_hybrid
import eval_hf_benchmarks
from hybrid_agent import HybridEngine, enable_tensorrt_acceleration
import hybrid_agent.benchmarks as benchmarks
import hybrid_inference.run_batch as run_batch_cli
import sglang_tensorrt_runner


class TestM3BenchmarksAndRunners(unittest.TestCase):
    """20-test unit suite verifying M3 Features F7 and F8."""

    def test_01_benchmark_specs_and_reference_scores_contract(self) -> None:
        expected_refs = {
            "mmlu_pro": 82.6,
            "aime_2026": 88.3,
            "gpqa_diamond": 82.3,
            "livecodebench_v6": 77.1,
            "mmmu_pro": 73.8,
        }
        self.assertEqual(benchmarks.REFERENCE_SCORES, expected_refs)
        self.assertEqual(set(benchmarks.BENCHMARK_SPECS.keys()), set(expected_refs.keys()))
        for key, ref in expected_refs.items():
            spec = benchmarks.BENCHMARK_SPECS[key]
            self.assertEqual(spec.name, key)
            self.assertAlmostEqual(spec.reference_score, ref)

        self.assertFalse(benchmarks.BENCHMARK_SPECS["aime_2026"].allow_tools)
        self.assertTrue(benchmarks.BENCHMARK_SPECS["mmlu_pro"].allow_tools)
        self.assertTrue(benchmarks.BENCHMARK_SPECS["gpqa_diamond"].allow_tools)
        self.assertTrue(benchmarks.BENCHMARK_SPECS["livecodebench_v6"].allow_tools)
        self.assertTrue(benchmarks.BENCHMARK_SPECS["mmmu_pro"].allow_tools)
        self.assertTrue(benchmarks.BENCHMARK_SPECS["mmmu_pro"].multimodal)
        self.assertEqual(benchmarks.BENCHMARK_SPECS["mmlu_pro"].num_choices, 10)
        self.assertEqual(benchmarks.BENCHMARK_SPECS["gpqa_diamond"].num_choices, 4)
        self.assertEqual(benchmarks.BENCHMARK_SPECS["mmmu_pro"].num_choices, 10)

        for submod in ("datasets", "prompts", "graders", "runner"):
            self.assertTrue(hasattr(benchmarks, submod))

    def test_02_recommend_g4_concurrency_bf16_vs_4bit_and_vram_tiers(self) -> None:
        rec_bf16 = benchmarks.recommend_g4_concurrency(
            "milkshake721/Gemma-4-26B-A4B-HybridAgent", vram_gb=96.0
        )
        rec_4bit = benchmarks.recommend_g4_concurrency(
            "milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit", vram_gb=96.0
        )
        self.assertFalse(rec_bf16["quantized_4bit"])
        self.assertTrue(rec_4bit["quantized_4bit"])
        self.assertGreaterEqual(rec_bf16["batch_size"], 32)
        self.assertGreater(rec_4bit["batch_size"], rec_bf16["batch_size"])
        for b_key in benchmarks.REFERENCE_SCORES:
            self.assertGreaterEqual(rec_4bit[b_key], 16)
            self.assertEqual(rec_4bit["per_benchmark"][b_key], rec_4bit[b_key])

        rec_24gb = benchmarks.recommend_g4_concurrency(
            "milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit", vram_gb=24.0
        )
        self.assertLess(rec_24gb["batch_size"], rec_4bit["batch_size"])

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_4b = create_temp_checkpoint_dir(Path(tmp), quantized_4bit=True)
            rec_dir = benchmarks.recommend_g4_concurrency(ckpt_4b, vram_gb=96.0)
            self.assertTrue(rec_dir["quantized_4bit"])

    def test_03_mmlu_pro_normalizer_na_filtering_and_answer_index(self) -> None:
        raw = {
            "question_id": 42,
            "question": "Which conservation law holds?",
            "options": "['Energy', 'N/A', 'Momentum', 'n/a', 'Charge']",
            "answer": "",
            "answer_index": 2,
            "category": "physics",
        }
        norm = benchmarks.datasets.normalize_mmlu_pro_item(raw)
        self.assertEqual(norm["id"], 42)
        self.assertEqual(norm["options"], ["Energy", "Momentum", "Charge"])
        self.assertEqual(norm["answer"], "C")
        prompt = benchmarks.prompts.build_benchmark_prompt("mmlu_pro", norm)
        self.assertIn("(A) Energy", prompt)
        self.assertIn("(B) Momentum", prompt)
        self.assertIn("(C) Charge", prompt)
        self.assertNotIn("N/A", prompt)

    def test_04_mcq_choice_extractor_10_options_and_edge_cases(self) -> None:
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("<answer>\\boxed{J}</answer>", num_choices=10), "J"
        )
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("<answer>\\boxed{\\text{D}}</answer>", num_choices=10), "D"
        )
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("<answer>\\boxed{(F)}</answer>", num_choices=10), "F"
        )
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("<answer>\\boxed{H", num_choices=10), "H"
        )
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("After checking all options, the answer is (J).", num_choices=10),
            "J",
        )
        # With num_choices=4 (A-D), J is out of range
        self.assertEqual(
            benchmarks.graders.extract_mcq_choice("The answer is (D).", num_choices=4), "D"
        )

    def test_05_aime_2026_normalizer_and_3digit_integer_grader(self) -> None:
        raw = {"id": "aime_1", "problem": "Find n.", "answer": "007"}
        norm = benchmarks.datasets.normalize_aime_2026_item(raw)
        self.assertEqual(norm["question"], "Find n.")
        self.assertEqual(norm["answer"], "007")

        ok_007, pred_007 = benchmarks.graders.grade_aime_2026("<answer>\\boxed{007}</answer>", "7")
        self.assertTrue(ok_007)
        self.assertEqual(pred_007, "007")

        ok_000, pred_000 = benchmarks.graders.grade_aime_2026("<answer>\\boxed{000}</answer>", "0")
        self.assertTrue(ok_000)
        self.assertEqual(pred_000, "000")

        ok_unc, pred_unc = benchmarks.graders.grade_aime_2026("<answer>\\boxed{42", "42")
        self.assertTrue(ok_unc)
        self.assertEqual(pred_unc, "042")

        ok_frac, pred_frac = benchmarks.graders.grade_aime_2026("<answer>\\boxed{\\frac{250}{2}}</answer>", "125")
        self.assertTrue(ok_frac)
        self.assertEqual(pred_frac, "125")

        ok_pow, pred_pow = benchmarks.graders.grade_aime_2026("<answer>\\boxed{5^3}</answer>", "125")
        self.assertTrue(ok_pow)
        self.assertEqual(pred_pow, "125")

    def test_06_gpqa_diamond_deterministic_seeded_shuffle(self) -> None:
        raw = {
            "Record ID": "rec_gpqa_99",
            "Question": "What is the fine-structure constant approximately equal to?",
            "Correct Answer": "1/137",
            "Incorrect Answer 1": "1/2",
            "Incorrect Answer 2": "1/10",
            "Incorrect Answer 3": "137",
        }
        g1 = benchmarks.datasets.normalize_gpqa_diamond_item(raw, seed=42)
        g2 = benchmarks.datasets.normalize_gpqa_diamond_item(raw, seed=42)
        self.assertEqual(g1["options"], g2["options"])
        self.assertEqual(g1["answer"], g2["answer"])
        gold_idx = ord(g1["answer"]) - 65
        self.assertEqual(g1["options"][gold_idx], "1/137")

        # Pre-shuffled options are preserved as-is
        pre = {
            "id": "gpqa_pre",
            "Question": "Pre-shuffled?",
            "options": ["A_opt", "B_opt", "C_opt", "D_opt"],
            "answer": "C",
        }
        g_pre = benchmarks.datasets.normalize_gpqa_diamond_item(pre, seed=42)
        self.assertEqual(g_pre["options"], ["A_opt", "B_opt", "C_opt", "D_opt"])
        self.assertEqual(g_pre["answer"], "C")

    def test_07_livecodebench_v6_zlib_pickle_base64_and_json_decoding(self) -> None:
        pub = [{"input": "[1, 2]\n3", "output": "[0, 1]", "testtype": "functional"}]
        priv = [{"input": "[4, 5]\n9", "output": "[0, 1]", "testtype": "functional"}]
        raw = {
            "question_id": "lcb_test_7",
            "question_title": "Two Sum",
            "question_content": "Return indices.",
            "starter_code": "class Solution:\n    def twoSum(self, nums: list[int], target: int) -> list[int]:\n",
            "public_test_cases": json.dumps(pub),
            "private_test_cases": compress_lcb_private_tests(priv),
            "metadata": "{}",
        }
        norm = benchmarks.datasets.normalize_livecodebench_v6_item(raw)
        self.assertEqual(norm["fn_name"], "twoSum")
        self.assertEqual(len(norm["test_cases"]), 2)
        self.assertEqual(norm["test_cases"][1]["input"], "[4, 5]\n9")

    def test_08_livecodebench_v6_code_block_extractor_ignores_think_and_python_tools(self) -> None:
        text = (
            "<think>Scratch:\n```python\nprint('draft')\n```\n</think>"
            "<python>print(1+1)</python><result>2</result>"
            "<think>Now final.</think>"
            "<answer>First block:\n```python\nprint('old')\n```\n"
            "Final block:\n```python\nimport sys\nprint(sys.stdin.read().strip())\n```</answer>"
        )
        code = benchmarks.graders.extract_code_block(text)
        self.assertNotIn("draft", code)
        self.assertNotIn("old", code)
        self.assertIn("sys.stdin.read()", code)

        unclosed = "<answer>```python\nprint(123)\n"
        self.assertEqual(benchmarks.graders.extract_code_block(unclosed), "print(123)")

    def test_09_livecodebench_v6_sandbox_grader_stdin_functional_and_timeout(self) -> None:
        stdin_item = {
            "fn_name": None,
            "test_cases": [
                {"input": "4 5\n", "output": "20\n", "testtype": "stdin"},
                {"input": "3 7\n", "output": "21\n", "testtype": "stdin"},
            ],
        }
        ok_stdin, _, meta_stdin = benchmarks.graders.grade_livecodebench_v6(
            "<answer>```python\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a * b)\n```</answer>",
            stdin_item,
        )
        self.assertTrue(ok_stdin)
        self.assertEqual(meta_stdin["reason"], "passed")
        self.assertEqual(meta_stdin["passed_tests"], 2)

        fn_item = {
            "fn_name": "addPair",
            "test_cases": [
                {"input": "10\n25", "output": "35", "testtype": "functional"},
            ],
        }
        ok_fn, _, meta_fn = benchmarks.graders.grade_livecodebench_v6(
            "<answer>```python\nclass Solution:\n    def addPair(self, a: int, b: int) -> int:\n        return a + b\n```</answer>",
            fn_item,
        )
        self.assertTrue(ok_fn)
        self.assertEqual(meta_fn["reason"], "passed")

        ok_wrong, _, meta_wrong = benchmarks.graders.grade_livecodebench_v6(
            "<answer>```python\nprint(999)\n```</answer>",
            stdin_item,
        )
        self.assertFalse(ok_wrong)
        self.assertEqual(meta_wrong["reason"], "wrong_answer")

        t0 = time.time()
        ok_to, _, meta_to = benchmarks.graders.grade_livecodebench_v6(
            "<answer>```python\nwhile True:\n    pass\n```</answer>",
            stdin_item,
            timeout_per_test=0.4,
        )
        self.assertFalse(ok_to)
        self.assertEqual(meta_to["reason"], "timeout")
        self.assertLess(time.time() - t0, 2.5)

    def test_10_mmmu_pro_multimodal_normalizer_standard10_and_vision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "sample.png"
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(img_path)

            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color=(40, 50, 60)).save(buf, format="PNG")
            raw_bytes = buf.getvalue()

            raw_std = {
                "id": "mmmu_1",
                "subset": "standard (10 options)",
                "question": "Compare <image 1> and <image 2>.",
                "options": [f"Opt {i}" for i in range(10)],
                "answer": "e",
                "image_1": str(img_path),
                "image_2": {"bytes": raw_bytes},
            }
            norm_std = benchmarks.datasets.normalize_mmmu_pro_item(raw_std)
            self.assertEqual(norm_std["answer"], "E")
            self.assertEqual(len(norm_std["images"]), 2)
            self.assertTrue(all(isinstance(im, Image.Image) for im in norm_std["images"]))

    def test_11_load_benchmark_dataset_fallback_chain_offline_and_local_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ds_paths = create_synthetic_benchmark_datasets(tmp_p)
            local_dir = ds_paths["mmlu_pro"].parent

            # 1) Explicit dataset_items with limit
            explicit = benchmarks.load_benchmark_dataset(
                "mmlu_pro",
                dataset_items=build_synthetic_benchmark_items()["mmlu_pro"],
                limit=3,
            )
            self.assertEqual(len(explicit), 3)

            # 2) Local dataset_dir
            from_local = benchmarks.load_benchmark_dataset("mmmu_pro", dataset_dir=local_dir)
            self.assertEqual(len(from_local), 5)
            self.assertTrue(all(it["images"] is not None for it in from_local))

            # 3) Mocked HF Hub fallback from gated Idavidrein/gpqa -> fingertap/GPQA-Diamond
            calls: list[str] = []

            def _fake_load_dataset(repo: str, *args, **kwargs):
                calls.append(repo)
                if repo == "Idavidrein/gpqa":
                    raise PermissionError("403 GatedRepoError")
                return [
                    {
                        "id": "hf_fallback_0",
                        "Question": "Fallback question?",
                        "Correct Answer": "Yes",
                        "Incorrect Answer 1": "No1",
                        "Incorrect Answer 2": "No2",
                        "Incorrect Answer 3": "No3",
                    }
                ]

            mock_hf = types.ModuleType("datasets")
            mock_hf.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
            with (
                mock.patch.dict(sys.modules, {"datasets": mock_hf}),
                mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}),
            ):
                hf_items = benchmarks.load_benchmark_dataset("gpqa_diamond")
                self.assertEqual(len(hf_items), 1)
                self.assertEqual(hf_items[0]["id"], "hf_fallback_0")
                self.assertIn("Idavidrein/gpqa", calls)
                self.assertIn("fingertap/GPQA-Diamond", calls)

    def test_12_evaluate_benchmark_mmlu_pro_and_gpqa_diamond_end_to_end(self) -> None:
        model = build_tiny_hybrid_model()
        engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

        res_mmlu = benchmarks.evaluate_benchmark(
            "mmlu_pro",
            engine,
            dataset_items=build_synthetic_benchmark_items()["mmlu_pro"],
            batch_size=4,
        )
        self.assertEqual(res_mmlu.accuracy, 100.0)
        self.assertEqual(res_mmlu.reference_score, 82.6)
        self.assertGreater(res_mmlu.questions_per_sec, 0.0)
        self.assertEqual(res_mmlu["num_samples"], 5)

        res_gpqa = benchmarks.evaluate_benchmark(
            "gpqa_diamond",
            engine,
            dataset_items=build_synthetic_benchmark_items()["gpqa_diamond"],
            batch_size=4,
        )
        self.assertEqual(res_gpqa.accuracy, 100.0)
        self.assertEqual(res_gpqa.reference_score, 82.3)

    def test_13_evaluate_benchmark_aime_2026_strict_zero_tools_enforcement(self) -> None:
        scripts = {
            "AIME_TOOL_ATTEMPT": (
                "Thinking.<latent:3></latent></think>"
                "<python>print(999)</python><result></result>"
                "<think>Done.</think><answer>\\boxed{042}</answer>"
            )
        }
        model = build_tiny_hybrid_model(scripted_responses=scripts)
        engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

        res = benchmarks.evaluate_benchmark(
            "aime_2026",
            engine,
            dataset_items=[{"id": "a0", "problem": "AIME_TOOL_ATTEMPT", "answer": "42"}],
            batch_size=1,
        )
        self.assertEqual(res.total_tool_calls, 0)
        self.assertEqual(res.predictions[0]["python_calls"], 0)
        self.assertEqual(res.predictions[0]["search_calls"], 0)
        self.assertGreaterEqual(res.total_latent_steps, 3)

    def test_14_evaluate_benchmark_livecodebench_v6_and_mmmu_pro_multimodal(self) -> None:
        model = build_tiny_hybrid_model(with_vision=True)
        enable_tensorrt_acceleration(model, max_batch_size=16)
        engine = HybridEngine(model, model.backbone.tokenizer, processor=model.processor, temperature=0.0)

        res_lcb = benchmarks.evaluate_benchmark(
            "livecodebench_v6",
            engine,
            dataset_items=build_synthetic_benchmark_items()["livecodebench_v6"],
            batch_size=2,
        )
        self.assertEqual(res_lcb.accuracy, 100.0)
        self.assertEqual(res_lcb.reference_score, 77.1)

        res_mmmu = benchmarks.evaluate_benchmark(
            "mmmu_pro",
            engine,
            dataset_items=build_synthetic_benchmark_items()["mmmu_pro"],
            batch_size=5,
        )
        self.assertEqual(res_mmmu.accuracy, 100.0)
        self.assertEqual(res_mmmu.reference_score, 73.8)
        self.assertGreaterEqual(model.backbone.vision_tower.call_count, 1)
        self.assertIn("standard (10 options)", res_mmmu.sub_scores)
        self.assertIn("vision", res_mmmu.sub_scores)

    def test_15_eval_hf_benchmarks_cli_parse_args_and_run_suite_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ckpt = create_temp_checkpoint_dir(tmp_p, include_vision=True)
            out_dir = tmp_p / "hf_out"

            args = eval_hf_benchmarks.parse_args(
                [
                    "--model",
                    str(ckpt),
                    "--backend",
                    "tensorrt",
                    "--benchmarks",
                    "mmlu_pro,aime_2026",
                    "gpqa_diamond",
                    "--batch_size",
                    "4",
                    "--synthetic-smoke",
                    "--output_dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(args.batch_size, 4)
            self.assertTrue(args.synthetic_smoke)

            summary = eval_hf_benchmarks.run_suite(args)
            self.assertIn("mmlu_pro", summary["benchmarks"])
            self.assertIn("aime_2026", summary["benchmarks"])
            self.assertIn("gpqa_diamond", summary["benchmarks"])
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertTrue((out_dir / "report.md").is_file())

    def test_16_run_batch_cli_multi_question_batching_no_tools_and_resume(self) -> None:
        self.assertFalse(getattr(run_batch_cli, "_e2e_bridged", False))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ckpt = create_temp_checkpoint_dir(tmp_p)
            in_jsonl = tmp_p / "in.jsonl"
            out_jsonl = tmp_p / "out.jsonl"
            questions = [{"id": i, "question": f"Problem {i}"} for i in range(4)]
            in_jsonl.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")

            args = run_batch_cli.parse_args(
                [
                    "--model",
                    str(ckpt),
                    "--input",
                    str(in_jsonl),
                    "--output",
                    str(out_jsonl),
                    "--question_batch_size",
                    "2",
                    "--num_samples",
                    "2",
                    "--no_tools",
                ]
            )
            self.assertEqual(args.backend, "tensorrt")
            self.assertEqual(args.question_batch_size, 2)
            self.assertEqual(args.max_tool_calls, 0)

            run_batch_cli.main(
                [
                    "--model",
                    str(ckpt),
                    "--input",
                    str(in_jsonl),
                    "--output",
                    str(out_jsonl),
                    "--question_batch_size",
                    "2",
                    "--num_samples",
                    "2",
                    "--no_tools",
                    "--temperature",
                    "0.0",
                ]
            )
            lines = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 8)
            self.assertTrue(all(r["num_tool_calls"] == 0 for r in lines))

    def test_17_sglang_tensorrt_runner_cli_multi_question_and_serve_cmd(self) -> None:
        self.assertFalse(getattr(sglang_tensorrt_runner, "_e2e_bridged", False))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ckpt_4bit = create_temp_checkpoint_dir(tmp_p, quantized_4bit=True)
            args = sglang_tensorrt_runner.parse_args(
                [
                    "--backend",
                    "tensorrt",
                    "--model",
                    str(ckpt_4bit),
                    "--questions",
                    "What is 1+1?",
                    "Solve 2*3",
                    "--batch-size",
                    "2",
                    "--no-tools",
                ]
            )
            self.assertEqual(args.batch_size, 2)
            self.assertTrue(args.no_tools)
            self.assertEqual(args.questions, ["What is 1+1?", "Solve 2*3"])

            proc = run_cli_command(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "sglang_tensorrt_runner.py"),
                    "--backend",
                    "sglang-serve-cmd",
                    "--model",
                    str(ckpt_4bit),
                ]
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("sglang.launch_server", proc.stdout)
            self.assertIn("--quantization compressed-tensors", proc.stdout)

    def test_18_eval_gemma4_26b_hybrid_batched_tensorrt_and_math_normalizer(self) -> None:
        self.assertEqual(eval_gemma4_26b_hybrid.normalize_math_answer("\\frac{3}{8}"), "3/8")
        self.assertEqual(eval_gemma4_26b_hybrid.normalize_math_answer("\\dfrac{15}{4}"), "15/4")
        self.assertEqual(eval_gemma4_26b_hybrid.normalize_math_answer("48 km/h"), "48")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ckpt = create_temp_checkpoint_dir(tmp_p)
            out_json = tmp_p / "eval10.json"
            payload = eval_gemma4_26b_hybrid.main([str(ckpt), str(out_json), "tensorrt"])
            self.assertEqual(payload["accuracy"], 1.0)
            self.assertEqual(len(payload["results"]), 10)
            self.assertTrue(out_json.is_file())

    def test_19_run_batch_inference_sh_script_syntax_and_flag_forwarding(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "run_batch_inference.sh"
        self.assertTrue(script_path.is_file())
        syntax_check = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
        self.assertEqual(syntax_check.returncode, 0, msg=syntax_check.stderr)
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("--backend", content)
        self.assertIn("--question_batch_size", content)

    def test_20_evaluate_all_benchmarks_and_markdown_report_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            model = build_tiny_hybrid_model(with_vision=True)
            engine = HybridEngine(model, model.backbone.tokenizer, processor=model.processor, temperature=0.0)
            out_dir = tmp_p / "all_bench_out"

            suite_res = benchmarks.evaluate_all_benchmarks(
                engine,
                benchmarks=["all"],
                output_dir=out_dir,
                batch_size=4,
                synthetic_smoke=True,
            )
            md_text = (out_dir / "report.md").read_text(encoding="utf-8")
            for b_name, ref_val in benchmarks.REFERENCE_SCORES.items():
                self.assertIn(b_name, suite_res["benchmarks"])
                self.assertEqual(suite_res["benchmarks"][b_name]["accuracy"], 100.0)
                self.assertIn(f"`{b_name}`", md_text)
                self.assertIn(f"{ref_val:.1f}%", md_text)


if __name__ == "__main__":
    unittest.main()
