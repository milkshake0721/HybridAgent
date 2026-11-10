"""Tier 4 End-to-End Business Workflow & User Journey Scenarios (8 Tests: `test_t4_s01` .. `test_t4_s08`).

Simulates complete real-world user journeys from checkpoint creation/loading and G4 GPU
concurrency auto-tuning through multi-shard crash recovery, multimodal STEM analysis,
competitive programming sandboxes, heterogeneous 16-row lockstep stress tests, and all 4 CLI runners.
Compatible with both `pytest` and `unittest` (`python3 tests/run_e2e_tests.py --tier 4`).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from PIL import Image
import torch

from tests.e2e.helpers import (
    PROJECT_ROOT,
    SpyTensorRTEngine,
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

import eval_hf_benchmarks
from hybrid_agent import (
    HybridEngine,
    HybridGenerationEngine,
    HybridLink,
    SGLangHybridEngine,
    ToolExecutor,
    enable_tensorrt_acceleration,
    extract_answer,
    get_sglang_server_command,
    load_hybrid_agent,
    prepare_sglang_model_dir,
)
from hybrid_agent.backends import _TensorRTHybridLinkWrapper
import hybrid_agent.benchmarks as benchmarks
from hybrid_agent.tools import PythonTool
import hybrid_inference.run_batch as run_batch_cli


def test_t4_s01_e2e_bf16_checkpoint_to_tensorrt_5_benchmark_suite_journey() -> None:
    """Scenario S01: BF16 checkpoint (without processor_config.json) -> G4 concurrency auto-tuning ->
    TensorRT acceleration -> 5-benchmark evaluation suite -> JSON & Markdown report verification."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt_bf16 = create_temp_checkpoint_dir(
            tmp_p,
            name="Gemma-4-26B-A4B-HybridAgent",
            quantized_4bit=False,
            include_processor_config=False,
            include_vision=True,
        )
        ds_paths = create_synthetic_benchmark_datasets(tmp_p)
        out_dir = tmp_p / "g4_eval_outputs"

        g4_cfg = benchmarks.recommend_g4_concurrency(ckpt_bf16, vram_gb=96.0)
        assert g4_cfg["quantized_4bit"] is False
        assert g4_cfg["batch_size"] >= 32

        proc = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_hf_benchmarks.py"),
                "--model",
                str(ckpt_bf16),
                "--backend",
                "tensorrt",
                "--benchmarks",
                "all",
                "--batch-size",
                "4",
                "--dataset-dir",
                str(ds_paths["mmlu_pro"].parent),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        report_md = (out_dir / "report.md").read_text(encoding="utf-8")

        expected_targets = {
            "mmlu_pro": 82.6,
            "aime_2026": 88.3,
            "gpqa_diamond": 82.3,
            "livecodebench_v6": 77.1,
            "mmmu_pro": 73.8,
        }
        for b_name, ref_target in expected_targets.items():
            assert b_name in summary["benchmarks"]
            b_res = summary["benchmarks"][b_name]
            assert b_res["reference_score"] == ref_target
            assert b_res["accuracy"] == 100.0
            assert b_res["questions_per_sec"] > 0
            assert (out_dir / f"{b_name}.json").is_file()
            assert f"{ref_target:.1f}%" in report_md


def test_t4_s02_e2e_4bit_checkpoint_high_concurrency_tensorrt_and_sglang_dual_backend_journey() -> None:
    """Scenario S02: 4-bit compressed-tensors checkpoint -> TensorRT HybridEngine.run_batch (B=16) ->
    prepare_sglang_model_dir -> SGLangHybridEngine.run_batch (B=16) parity verification."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt_4bit = create_temp_checkpoint_dir(
            tmp_p,
            name="Gemma-4-26B-A4B-HybridAgent-4bit",
            quantized_4bit=True,
            include_processor_config=False,
            include_vision=True,
        )
        model, tok = load_hybrid_agent(ckpt_4bit, backend="tensorrt", max_batch_size=64)
        trt_engine = HybridEngine(model, tok, temperature=0.0)

        questions = [f"Problem {i}" for i in range(16)]
        trt_trajs = trt_engine.run_batch(questions, batch_size=16)
        assert len(trt_trajs) == 16
        assert [t.final_answer for t in trt_trajs] == [f"ans_{i}" for i in range(16)]

        # Prepare directory for SGLang and verify server command
        sgl_dir = prepare_sglang_model_dir(ckpt_4bit)
        cmd = get_sglang_server_command(sgl_dir, port=30010)
        assert "--quantization" in cmd and "compressed-tensors" in cmd

        # Run SGLangHybridEngine on the same 16 questions
        sgl_engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
        sgl_engine.device = torch.device("cpu")
        sgl_engine.dtype = torch.bfloat16
        sgl_engine.max_new_tokens = 128
        sgl_engine.min_latent_steps = 4
        sgl_engine.max_latent_steps = 16
        sgl_engine.temperature = 0.0
        sgl_engine.top_p = 1.0
        sgl_engine.latent_mode = "hybrid"
        sgl_engine.latent_open_id = 262144
        sgl_engine.latent_close_id = 262145
        sgl_engine._hybridlink_lock = threading.Lock()
        sgl_engine._fallback_hybrid_engine = trt_engine
        sgl_engine._sgl_engine = None

        sgl_trajs = sgl_engine.run_batch(questions, max_concurrency=16)
        assert len(sgl_trajs) == 16
        assert [t.final_answer for t in sgl_trajs] == [t.final_answer for t in trt_trajs]
        assert [t["latent_steps"] for t in sgl_trajs] == [t.latent_steps for t in trt_trajs]


def test_t4_s03_e2e_distributed_sharded_batch_inference_crash_and_resume_journey() -> None:
    """Scenario S03: 12-question x 2-sample distributed job across 2 shards with simulated crash and --resume."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        in_jsonl = tmp_p / "batch_12.jsonl"
        s0_out = tmp_p / "shard_0.jsonl"
        s1_out = tmp_p / "shard_1.jsonl"

        questions = [{"id": i, "question": f"Problem {i}", "category": "math"} for i in range(12)]
        in_jsonl.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")

        # Step 1: Run Shard 0 with --limit 4 first (simulating partial completion of ids 0, 2 plus a corrupted tail)
        run_batch_cli.main(
            [
                "--model",
                str(ckpt),
                "--input",
                str(in_jsonl),
                "--output",
                str(s0_out),
                "--backend",
                "tensorrt",
                "--num_shards",
                "2",
                "--shard_id",
                "0",
                "--num_samples",
                "2",
                "--question_batch_size",
                "2",
                "--limit",
                "4",
                "--temperature",
                "0.0",
            ]
        )
        partial_lines = [l for l in s0_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(partial_lines) == 4  # ids 0 and 2, samples 0 and 1

        # Append a corrupted incomplete line simulating power loss mid-write, then verify read_done
        s0_out.write_text("\n".join(partial_lines) + '\n{"id": 4, "sample_idx": 0, "broken', encoding="utf-8")
        done_pairs = run_batch_cli.read_done(str(s0_out))
        assert done_pairs == {(0, 0), (0, 1), (2, 0), (2, 1)}

        # Restore clean lines and resume Shard 0 for all 12 questions, plus run Shard 1
        s0_out.write_text("\n".join(partial_lines) + "\n", encoding="utf-8")
        run_batch_cli.main(
            [
                "--model",
                str(ckpt),
                "--input",
                str(in_jsonl),
                "--output",
                str(s0_out),
                "--backend",
                "tensorrt",
                "--num_shards",
                "2",
                "--shard_id",
                "0",
                "--num_samples",
                "2",
                "--question_batch_size",
                "3",
                "--resume",
                "--temperature",
                "0.0",
            ]
        )
        run_batch_cli.main(
            [
                "--model",
                str(ckpt),
                "--input",
                str(in_jsonl),
                "--output",
                str(s1_out),
                "--backend",
                "tensorrt",
                "--num_shards",
                "2",
                "--shard_id",
                "1",
                "--num_samples",
                "2",
                "--question_batch_size",
                "3",
                "--temperature",
                "0.0",
            ]
        )

        recs_0 = [json.loads(l) for l in s0_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs_1 = [json.loads(l) for l in s1_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_recs = recs_0 + recs_1
        assert len(all_recs) == 24
        assert {(r["id"], r["sample_idx"]) for r in all_recs} == {(i, s) for i in range(12) for s in range(2)}
        for r in all_recs:
            assert r["prediction"] == f"ans_{r['id']}"
            assert r["latent_steps"] == 4


def test_t4_s04_e2e_multimodal_stem_research_assistant_journey() -> None:
    """Scenario S04: Multimodal STEM workflow combining multi-image inputs, grayscale/RGBA images,
    latent readout rendering, and Python tool calculations in a single batch."""
    scripts = {
        "STEM_MULTI_IMG": "Compare two plots.<latent:4>PLOT</latent>Done.</think><answer>\\boxed{B}</answer>",
        "STEM_GRAY_CALC": "Read grayscale meter.<latent:3>MTR</latent>Calc.</think><python>print(12 * 15)</python><result></result><think>Done.</think><answer>\\boxed{180}</answer>",
        "STEM_RGBA": "Read RGBA spectrum.<latent:3>SPC</latent>Done.</think><answer>\\boxed{G}</answer>",
        "STEM_TEXT_ONLY": "Pure text equation.<latent:3>TXT</latent>Done.</think><answer>\\boxed{42}</answer>",
    }
    model = build_tiny_hybrid_model(with_vision=True, scripted_responses=scripts)
    enable_tensorrt_acceleration(model, max_batch_size=32)
    engine = HybridEngine(model, model.backbone.tokenizer, processor=model.processor, temperature=0.0, record_latent_readout=True)

    images = [
        [Image.new("RGB", (64, 64), color=(200, 50, 50)), Image.new("RGB", (64, 64), color=(50, 50, 200))],
        Image.new("L", (32, 32), color=180),
        Image.new("RGBA", (48, 48), color=(30, 180, 90, 255)),
        None,
    ]
    questions = ["STEM_MULTI_IMG", "STEM_GRAY_CALC", "STEM_RGBA", "STEM_TEXT_ONLY"]

    trajs = engine.run_batch(questions, images=images, batch_size=4, record_latent_readout=True)
    assert [t.final_answer for t in trajs] == ["B", "180", "G", "42"]
    assert trajs[1].raw_result["python_calls"] == 1
    assert "<result>180</result>" in trajs[1].output
    for tr in trajs:
        assert tr.latent_steps >= 3
        assert len(tr.raw_result["latent_readouts"]) == 1
        assert "<latent></latent>" not in tr.output_rendered


def test_t4_s05_e2e_aime_2026_and_math_competition_benchmark_journey() -> None:
    """Scenario S05: Competition math evaluation across both `eval_gemma4_26b_hybrid.py` (10-Q benchmark)
    and `aime_2026` (strict 0-tool Olympiad integer benchmark)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        eval_json = tmp_p / "eval_10q.json"

        proc = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_gemma4_26b_hybrid.py"),
                str(ckpt),
                str(eval_json),
                "tensorrt",
            ]
        )
        assert proc.returncode == 0
        data_10q = json.loads(eval_json.read_text(encoding="utf-8"))
        assert data_10q["accuracy"] == 1.0
        assert data_10q["results"][-1]["expected"] == "3/8"
        assert data_10q["results"][-1]["predicted"] == "3/8"

        model, tok = load_hybrid_agent(ckpt, backend="tensorrt")
        engine = HybridEngine(model, tok, temperature=0.0)
        aime_items = build_synthetic_benchmark_items()["aime_2026"]
        aime_res = benchmarks.evaluate_benchmark("aime_2026", engine, dataset_items=aime_items, batch_size=5)

        assert aime_res.accuracy == 100.0
        assert aime_res.reference_score == 88.3
        assert aime_res.total_tool_calls == 0


def test_t4_s06_e2e_livecodebench_v6_competitive_programming_pipeline_journey() -> None:
    """Scenario S06: LiveCodeBench v6 pipeline where model uses <python> scratchpad during <think>
    before emitting final code solution in <answer>, plus negative grading of buggy/timeout code."""
    lcb_code = "import sys\ndata = sys.stdin.read().split()\nprint(int(data[0]) * int(data[1]))\n"
    scripts = {
        "Multiply two integers from stdin": (
            "Test in scratchpad.<latent:3></latent></think>"
            "<python>print(6 * 7)</python><result></result>"
            f"<think>Verified.</think><answer>```python\n{lcb_code}```</answer>"
        )
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    enable_tensorrt_acceleration(model, max_batch_size=16)
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

    pub_tests = [{"input": "6 7\n", "output": "42\n", "testtype": "stdin"}]
    priv_tests = [{"input": "12 11\n", "output": "132\n", "testtype": "stdin"}]
    dataset = [
        {
            "question_id": "lcb_mul_0",
            "question_title": "Multiply Two Integers",
            "question_content": "Multiply two integers from stdin and print the product.",
            "starter_code": "",
            "public_test_cases": json.dumps(pub_tests),
            "private_test_cases": compress_lcb_private_tests(priv_tests),
            "metadata": "{}",
        }
    ]

    res = benchmarks.evaluate_benchmark("livecodebench_v6", engine, dataset_items=dataset, batch_size=1)
    assert res.accuracy == 100.0
    assert res.total_tool_calls == 1
    assert res.predictions[0]["passed_tests"] == 2


def test_t4_s07_e2e_heterogeneous_workload_stress_test_journey() -> None:
    """Scenario S07: 16-row heterogeneous lockstep batch stress test combining anchor latent exits,
    forced max_latent_steps exits, concurrent Python/Search tools, multimodal images, and tool_cap."""
    scripts: dict[str, object] = {}
    questions: list[str] = []
    images: list[object] = []

    # Rows 0..3: Multi-span <latent> reasoning (anchor exit)
    for i in range(4):
        q = f"STRESS_LAT_{i}"
        questions.append(q)
        images.append(None)
        scripts[q] = f"S1.<latent:3></latent>S2.<latent:2></latent>Done.</think><answer>\\boxed{{LAT_{i}}}</answer>"

    # Rows 4..7: Forced max_latent_steps exit (never_signal_exit=True)
    for i in range(4, 8):
        q = f"STRESS_FORCED_{i}"
        questions.append(q)
        images.append(None)
        scripts[q] = {
            "turns": [f"Force.<latent></latent>Done.</think><answer>\\boxed{{FORCE_{i}}}</answer>"],
            "latent_steps": 10,
            "never_signal_exit": True,
        }

    # Rows 8..11: Python & Search tool calls with multi-token FEED prefill
    for i in range(8, 12):
        q = f"STRESS_TOOL_{i}"
        questions.append(q)
        images.append(None)
        if i % 2 == 0:
            scripts[q] = f"Py.</think><python>print({i} * 100)</python><result></result><think>Done.</think><answer>\\boxed{{{i * 100}}}</answer>"
        else:
            scripts[q] = f"Sr.</think><search>topic {i}</search><result></result><think>Done.</think><answer>\\boxed{{SR_{i}}}</answer>"

    # Rows 12..13: Multimodal PIL.Image inputs
    for i in range(12, 14):
        q = f"STRESS_MM_{i}"
        questions.append(q)
        images.append(Image.new("RGB", (64, 64), color=(10 * i, 80, 160)))
        scripts[q] = f"Vision.<latent:3></latent>Done.</think><answer>\\boxed{{MM_{i}}}</answer>"

    # Rows 14..15: Immediate direct answers
    for i in range(14, 16):
        q = f"STRESS_FAST_{i}"
        questions.append(q)
        images.append(None)
        scripts[q] = f"Fast.</think><answer>\\boxed{{FAST_{i}}}</answer>"

    model = build_tiny_hybrid_model(quantized_4bit=True, with_vision=True, scripted_responses=scripts)
    enable_tensorrt_acceleration(model, max_batch_size=32)
    tok = model.backbone.tokenizer

    class _StressSearchTool:
        name = "stress_search"
        trigger_tag = "search"

        def execute(self, content: str, **kwargs) -> str:
            return f"Page 1: {content}"

    tools = ToolExecutor()
    tools.register_tool(PythonTool(timeout=10))
    tools.register_tool(_StressSearchTool())  # type: ignore[arg-type]

    engine = HybridEngine(
        model,
        tok,
        tool_executor=tools,
        processor=model.processor,
        min_latent_steps=2,
        max_latent_steps=5,
        temperature=0.0,
    )

    def _forbid_nonzero(*args, **kwargs):
        raise RuntimeError("Tensor.nonzero() forbidden")

    with mock.patch.object(torch.Tensor, "nonzero", _forbid_nonzero):
        trajs = engine.run_batch(questions, images=images, batch_size=16)

    assert len(trajs) == 16
    for i in range(4):
        assert trajs[i].final_answer == f"LAT_{i}"
        assert trajs[i].latent_spans == 2
        assert trajs[i].latent_steps == 5
    for i in range(4, 8):
        assert trajs[i].final_answer == f"FORCE_{i}"
        assert trajs[i].raw_result["forced_exits"] == 1
        assert trajs[i].latent_steps == 5
    for i in range(8, 12):
        expected = str(i * 100) if i % 2 == 0 else f"SR_{i}"
        assert trajs[i].final_answer == expected
    for i in range(12, 14):
        assert trajs[i].final_answer == f"MM_{i}"
    for i in range(14, 16):
        assert trajs[i].final_answer == f"FAST_{i}"


def test_t4_s08_e2e_cli_suite_and_checkpoint_roundtrip_persistence_journey() -> None:
    """Scenario S08: Full checkpoint save_pretrained -> load_hybrid_agent roundtrip -> execute all 4 CLI scripts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        model = build_tiny_hybrid_model(with_vision=True)
        tok = model.backbone.tokenizer

        saved_dir = tmp_p / "saved_hybrid_agent"
        model.save_pretrained(saved_dir, tokenizer=tok)
        # Write fixture marker so subprocesses reconstruct deterministic ScriptedHybridBackbone
        (saved_dir / "hybrid_e2e_fixture.json").write_text(
            json.dumps(
                {
                    "quantized_4bit": False,
                    "include_vision": True,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_experts": 8,
                    "top_k": 2,
                    "vocab_size": len(tok),
                    "scripted_responses": {},
                }
            ),
            encoding="utf-8",
        )

        reloaded_model, reloaded_tok = load_hybrid_agent(saved_dir, backend="tensorrt", load_processor=True)
        for p1, p2 in zip(model.hybridlink.parameters(), reloaded_model.hybridlink.parameters()):
            assert torch.allclose(p1, p2)

        # 1. CLI #1: sglang_tensorrt_runner.py
        p1 = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "sglang_tensorrt_runner.py"),
                "--backend",
                "tensorrt",
                "--model",
                str(saved_dir),
                "--question",
                "If 3x + 7 = 22, what is the value of 5x - 2?",
            ]
        )
        assert p1.returncode == 0
        assert "23" in p1.stdout

        # 2. CLI #2: eval_gemma4_26b_hybrid.py
        eval_out = tmp_p / "eval_out.json"
        p2 = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_gemma4_26b_hybrid.py"),
                str(saved_dir),
                str(eval_out),
                "tensorrt",
            ]
        )
        assert p2.returncode == 0
        assert json.loads(eval_out.read_text(encoding="utf-8"))["accuracy"] == 1.0

        # 3. CLI #3: hybrid_inference/run_batch.py
        q_in = tmp_p / "q.jsonl"
        q_out = tmp_p / "q_out.jsonl"
        q_in.write_text(json.dumps({"id": 0, "question": "What is 1+1?"}) + "\n", encoding="utf-8")
        p3 = run_cli_command(
            [
                sys.executable,
                "-m",
                "hybrid_inference.run_batch",
                "--model",
                str(saved_dir),
                "--input",
                str(q_in),
                "--output",
                str(q_out),
                "--backend",
                "tensorrt",
                "--temperature",
                "0.0",
            ]
        )
        assert p3.returncode == 0
        assert json.loads(q_out.read_text(encoding="utf-8").splitlines()[0])["prediction"] == "2"

        # 4. CLI #4: eval_hf_benchmarks.py
        b_out = tmp_p / "b_out"
        p4 = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_hf_benchmarks.py"),
                "--model",
                str(saved_dir),
                "--benchmarks",
                "mmlu_pro",
                "aime_2026",
                "--synthetic-smoke",
                "--output-dir",
                str(b_out),
            ]
        )
        assert p4.returncode == 0
        assert (b_out / "summary.json").is_file()


class Tier4ScenariosTestSuite(unittest.TestCase):
    """Unittest wrapper dynamically binding all 8 `test_t4_s*` functions for unittest discovery."""


for _fn_name, _fn_obj in list(globals().items()):
    if _fn_name.startswith("test_t4_s") and callable(_fn_obj):
        setattr(Tier4ScenariosTestSuite, _fn_name, staticmethod(_fn_obj))


if __name__ == "__main__":
    unittest.main()
