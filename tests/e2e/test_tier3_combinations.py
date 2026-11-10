"""Tier 3 Cross-Feature Combination End-to-End Tests (16 Tests: `test_t3_c01` .. `test_t3_c16`).

Verifies multi-feature interactions across M1 (TensorRT HybridLink, BF16/4-bit MoE, SGLang),
M2 (Multi-question lockstep engine, hot-path optimizations, multimodal vision), and
M3 (Batch CLIs, G4 concurrency auto-tuning, and the 5-benchmark evaluation suite).
Compatible with both `pytest` and `unittest` (`python3 tests/run_e2e_tests.py --tier 3`).
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock

from PIL import Image
import torch
import torch.nn.functional as F

from tests.e2e.helpers import (
    SpyTensorRTEngine,
    build_synthetic_benchmark_items,
    build_tiny_hybrid_model,
    create_temp_checkpoint_dir,
    ensure_contract_bridges,
    install_runtime_compat,
)

install_runtime_compat()
ensure_contract_bridges()

import eval_hf_benchmarks
from hybrid_agent import (
    HybridEngine,
    HybridGenerationEngine,
    HybridLink,
    HybridMultimodalProcessor,
    SGLangHybridEngine,
    ToolExecutor,
    enable_tensorrt_acceleration,
    load_hybrid_agent,
)
from hybrid_agent.backends import _TensorRTHybridLinkWrapper
import hybrid_agent.benchmarks as benchmarks
import hybrid_inference.run_batch as run_batch_cli


def test_t3_c01_tensorrt_hybridlink_and_bf16_moe_fastpath_coexistence() -> None:
    """F1 + F2: Dynamic-batch TensorRT HybridLink coexists with zero-sync BF16 MoE fast path."""
    torch.manual_seed(3001)
    model = build_tiny_hybrid_model(hidden_size=64, num_layers=2, quantized_4bit=False)
    info = enable_tensorrt_acceleration(model, max_batch_size=64, opt_batch_size=16, min_batch_size=1)

    assert info["bf16_experts_zero_sync_layers"] == 2
    assert isinstance(model.hybridlink, _TensorRTHybridLinkWrapper)
    assert model.hybridlink.max_batch_size == 64

    def _forbid_nonzero(*args, **kwargs):
        raise RuntimeError("Tensor.nonzero() forbidden")

    with mock.patch.object(torch.Tensor, "nonzero", _forbid_nonzero):
        for B in (1, 16, 64):
            input_ids = torch.randint(10, 100, (B, 4))
            out = model(input_ids=input_ids)
            last_h = out.hidden_states[-1][:, -1, :]
            lat_vec = model.run_hybridlink(last_h, output_dtype=torch.bfloat16)
            assert lat_vec.shape == (B, 64)
            assert lat_vec.dtype == torch.bfloat16


def test_t3_c02_tensorrt_hybridlink_and_4bit_w4a16_moe_coexistence() -> None:
    """F1 + F3: 4-bit pack-quantized model loads with TensorRT HybridLink and runs T=32 expert-sorted MoE."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_4bit = create_temp_checkpoint_dir(Path(tmp), quantized_4bit=True, hidden_size=64, num_layers=2)
        model, tok = load_hybrid_agent(ckpt_4bit, backend="tensorrt", max_batch_size=64)

        assert isinstance(model.hybridlink, _TensorRTHybridLinkWrapper)
        assert model.hybridlink.max_batch_size == 64
        info = getattr(model, "_acceleration_info", {})
        assert info.get("int4_ffn_compiled_layers", 0) == 2

        # Run T=32 (> SORT_EXPERT_THRESHOLD=8) through 4-bit backbone and TensorRT HybridLink
        input_ids = torch.randint(10, 100, (32, 2))
        out = model(input_ids=input_ids)
        lat = model.run_hybridlink(out.hidden_states[-1][:, -1, :])
        assert lat.shape == (32, 64)


def test_t3_c03_multi_question_engine_routes_concurrent_latent_rows_into_single_trt_hybridlink_call() -> None:
    """F1 + F4: When 3 of 4 rows in a multi-question batch are in LATENT mode, HybridLink is called once with shape (3, H)."""
    scripts = {
        "Q_LAT_0": "<latent:3></latent>Done.</think><answer>\\boxed{A0}</answer>",
        "Q_LAT_1": "<latent:3></latent>Done.</think><answer>\\boxed{A1}</answer>",
        "Q_DEC_2": "Plain text only.</think><answer>\\boxed{A2}</answer>",
        "Q_LAT_3": "<latent:3></latent>Done.</think><answer>\\boxed{A3}</answer>",
    }
    model = build_tiny_hybrid_model(hidden_size=64, scripted_responses=scripts)
    tok = model.backbone.tokenizer
    spy = SpyTensorRTEngine(model.hybridlink)
    model.hybridlink = _TensorRTHybridLinkWrapper(model.hybridlink, spy, hidden_size=64, max_batch_size=64)

    engine = HybridEngine(model, tok, temperature=0.0, min_latent_steps=2, max_latent_steps=6)
    trajs = engine.run_batch(["Q_LAT_0", "Q_LAT_1", "Q_DEC_2", "Q_LAT_3"], batch_size=4)

    assert [t.final_answer for t in trajs] == ["A0", "A1", "A2", "A3"]
    assert [t.latent_steps for t in trajs] == [3, 3, 0, 3]
    # All 3 latent rows entered <latent> on step 0 and stepped together with batch size 3!
    assert (3, 64) in spy.call_shapes


def test_t3_c04_multi_question_lockstep_batch_exercises_bf16_zero_sync_moe_and_preallocated_mask() -> None:
    """F2 + F4 + F5: 6-question lockstep batch with BF16 zero-sync MoE and pre-allocated attention mask."""
    model = build_tiny_hybrid_model(hidden_size=64, num_layers=2, quantized_4bit=False)
    enable_tensorrt_acceleration(model, max_batch_size=64)
    tok = model.backbone.tokenizer

    questions = [f"Problem {i}" for i in range(6)]
    gen_engine = HybridGenerationEngine(model, tok, questions, batch_size=1, temperature=0.0)

    def _forbid_nonzero(*args, **kwargs):
        raise RuntimeError("Tensor.nonzero() forbidden")

    with mock.patch.object(torch.Tensor, "nonzero", _forbid_nonzero):
        gen_engine._init_generation()
        buf_ptr = gen_engine._attn_mask_buf.data_ptr()
        while any(r.mode != "DONE" for r in gen_engine.rows):
            gen_engine._decode_step(rng=None)
            assert gen_engine._attn_mask_buf.data_ptr() == buf_ptr
        for r in gen_engine.rows:
            gen_engine._flush_new_model_text(r)

    answers = [
        HybridEngine._row_to_trajectory(gen_engine, r).final_answer
        for r in gen_engine.rows
    ]
    assert answers == [f"ans_{i}" for i in range(6)]


def test_t3_c05_multi_question_lockstep_batch_with_4bit_moe_expert_sorting_and_lazy_readout() -> None:
    """F3 + F4 + F5: 10-question lockstep batch (B=10 > SORT_EXPERT_THRESHOLD=8) on 4-bit MoE with lazy BF16 readout bank."""
    model = build_tiny_hybrid_model(hidden_size=64, num_layers=2, quantized_4bit=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0, record_latent_readout=True)

    questions = [f"Problem {i}" for i in range(10)]
    trajs = engine.run_batch(questions, batch_size=10, record_latent_readout=True)

    assert len(trajs) == 10
    for i, tr in enumerate(trajs):
        assert tr.final_answer == f"ans_{i}"
        assert tr.latent_steps == 4
        assert len(tr.raw_result["latent_readouts"]) == 1
        assert tr.raw_result["latent_readouts"][0]["pseudo_text"] != ""


def test_t3_c06_sglang_hybrid_engine_run_batch_with_tensorrt_hybridlink_and_python_tools() -> None:
    """F1 + F3 + F7: SGLangHybridEngine.run_batch with TensorRT HybridLink and <python> tool execution across 6 requests."""
    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 128
    engine.min_latent_steps = 3
    engine.max_latent_steps = 12
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = threading.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None

    base_link = HybridLink(64).to(dtype=torch.bfloat16).eval()
    spy = SpyTensorRTEngine(base_link)
    engine.hybridlink = _TensorRTHybridLinkWrapper(base_link, spy, hidden_size=64, max_batch_size=64)
    from hybrid_agent.tools import build_tool_executor

    engine.tools = build_tool_executor(require_search=False)

    lock = threading.Lock()
    steps_per_q: dict[int, int] = {}

    def _fake_sgl_gen(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
        q_idx = int(prompt.split("CALC_")[1].split("_")[0])
        with lock:
            st = steps_per_q.get(q_idx, 0)
            steps_per_q[q_idx] = st + 1
        if st == 0:
            return {
                "text": "<latent>",
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": 262144},
                    "hidden_states": [torch.randn(1, 64).tolist()],
                },
            }
        if st == 1:
            return {
                "text": f"</think><python>print({q_idx} * 11)</python>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": "</python>"}},
            }
        if st == 2:
            return {
                "text": "<think><latent>",
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": 262144},
                    "hidden_states": [torch.randn(1, 64).tolist()],
                },
            }
        return {
            "text": f"</think><answer>\\boxed{{{q_idx * 11}}}</answer>",
            "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
        }

    engine._sgl_generate = _fake_sgl_gen
    questions = [f"CALC_{i}_END" for i in range(6)]
    trajs = engine.run_batch(questions, max_concurrency=6)

    assert len(trajs) == 6
    assert engine.latent_mode == "hybrid"
    assert spy.call_count == 6 * 2 * 3  # 6 requests * 2 spans * 3 steps per span
    for i, tr in enumerate(trajs):
        assert tr.final_answer == str(i * 11)
        assert tr.latent_steps == 6
        assert tr.latent_spans == 2
        assert f"<result>{i * 11}</result>" in tr.output


def test_t3_c07_multi_question_concurrent_tool_barrier_with_multitoken_feed_prefill() -> None:
    """F4 + F5: 4-question batch with concurrent Python + Search tool barrier and multi-token FEED prefill."""
    scripts = {
        "Q_PY_LONG": "Calc.</think><python>print(10**12)</python><result></result><think>Done.</think><answer>\\boxed{1000000000000}</answer>",
        "Q_SEARCH": "Search.</think><search>Gemma 4 architecture</search><result></result><think>Done.</think><answer>\\boxed{Gemma4}</answer>",
        "Q_PY_SHORT": "Calc.</think><python>print(5)</python><result></result><think>Done.</think><answer>\\boxed{5}</answer>",
        "Q_NO_TOOL": "Direct.</think><answer>\\boxed{0}</answer>",
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    tok = model.backbone.tokenizer

    from hybrid_agent.tools import PythonTool

    class _MockSearchTool:
        name = "mock_search"
        trigger_tag = "search"

        def execute(self, content: str, **kwargs) -> str:
            return f"Page 1: {content} uses 128 routed experts."

    tools = ToolExecutor()
    tools.register_tool(PythonTool(timeout=10))
    tools.register_tool(_MockSearchTool())  # type: ignore[arg-type]

    engine = HybridEngine(model, tok, tool_executor=tools, temperature=0.0)
    trajs = engine.run_batch(["Q_PY_LONG", "Q_SEARCH", "Q_PY_SHORT", "Q_NO_TOOL"], batch_size=4)

    assert [t.final_answer for t in trajs] == ["1000000000000", "Gemma4", "5", "0"]
    assert trajs[0].raw_result["python_calls"] == 1
    assert trajs[1].raw_result["search_calls"] == 1
    assert trajs[2].raw_result["python_calls"] == 1
    assert trajs[3].raw_result["python_calls"] == 0
    assert model.backbone.multi_token_feed_call_count >= 1


def test_t3_c08_multimodal_batch_with_tensorrt_hybridlink_and_bf16_moe_fastpath() -> None:
    """F1 + F2 + F4 + F6: Multimodal batch with TensorRT HybridLink and BF16 MoE fast path."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(Path(tmp), include_vision=True, quantized_4bit=False)
        model, tok = load_hybrid_agent(ckpt, backend="tensorrt", max_batch_size=64, load_processor=True)
        spy = SpyTensorRTEngine(model.hybridlink)
        model.hybridlink = _TensorRTHybridLinkWrapper(model.hybridlink, spy, hidden_size=64, max_batch_size=64)

        engine = HybridEngine(model, tok, processor=model.processor, temperature=0.0)
        questions = ["What is 1+1?", "Solve 2*3", "Compute 10 - 4", "Value of 7?"]
        images = [Image.new("RGB", (64, 64), color=(20 * i, 100, 150)) for i in range(4)]

        trajs = engine.run_batch(questions, images=images, batch_size=4)
        assert [t.final_answer for t in trajs] == ["2", "6", "6", "7"]
        assert model.backbone.last_prefill_pixel_values is not None
        assert model.backbone.last_prefill_pixel_values.shape == (4, 3, 64, 64)
        assert (4, 64) in spy.call_shapes


def test_t3_c09_multimodal_4bit_checkpoint_missing_processor_config_with_batched_latent_tool_loop() -> None:
    """F3 + F4 + F5 + F6: 4-bit checkpoint without processor_config.json synthesizes processor and runs multimodal latent + tool batch."""
    with tempfile.TemporaryDirectory() as tmp:
        scripts = {
            "MM_4BIT_0": "Img 0.<latent:3></latent>Tool.</think><python>print(4 * 5)</python><result></result><think>Done.</think><answer>\\boxed{20}</answer>",
            "MM_4BIT_1": "Img 1.<latent:3></latent>Tool.</think><python>print(6 * 7)</python><result></result><think>Done.</think><answer>\\boxed{42}</answer>",
        }
        ckpt = create_temp_checkpoint_dir(
            Path(tmp),
            quantized_4bit=True,
            include_processor_config=False,
            include_vision=True,
            scripted_responses=scripts,
        )
        model, tok = load_hybrid_agent(ckpt, backend="tensorrt", load_processor=True)
        assert isinstance(model.processor, HybridMultimodalProcessor)

        engine = HybridEngine(model, tok, processor=model.processor, temperature=0.0)
        imgs = [Image.new("RGB", (64, 64), color=(10, 20, 30)), Image.new("RGB", (64, 64), color=(40, 50, 60))]
        trajs = engine.run_batch(["MM_4BIT_0", "MM_4BIT_1"], images=imgs, batch_size=2)

        assert [t.final_answer for t in trajs] == ["20", "42"]
        assert [t.latent_steps for t in trajs] == [3, 3]
        assert [t.raw_result["python_calls"] for t in trajs] == [1, 1]


def test_t3_c10_run_batch_cli_multi_question_tensorrt_with_latent_readouts_and_python_tools() -> None:
    """F1 + F4 + F5 + F7: `hybrid_inference.run_batch` CLI with TensorRT, question_batch_size=2, latent readouts, and Python tool calls."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        scripts = {
            "Q_CLI_0": "Step.<latent:3>XYZ</latent>Tool.</think><python>print(11 + 22)</python><result></result><think>Done.</think><answer>\\boxed{33}</answer>",
            "Q_CLI_1": "Step.<latent:3>ABC</latent>Tool.</think><python>print(33 + 44)</python><result></result><think>Done.</think><answer>\\boxed{77}</answer>",
        }
        ckpt = create_temp_checkpoint_dir(tmp_p, scripted_responses=scripts)
        in_jsonl = tmp_p / "in.jsonl"
        out_jsonl = tmp_p / "out.jsonl"
        in_jsonl.write_text(
            json.dumps({"id": 0, "question": "Q_CLI_0"}) + "\n" + json.dumps({"id": 1, "question": "Q_CLI_1"}) + "\n",
            encoding="utf-8",
        )

        run_batch_cli.main(
            [
                "--model",
                str(ckpt),
                "--input",
                str(in_jsonl),
                "--output",
                str(out_jsonl),
                "--backend",
                "tensorrt",
                "--question_batch_size",
                "2",
                "--num_samples",
                "2",
                "--min_latent_steps",
                "3",
                "--temperature",
                "0.0",
            ]
        )
        recs = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(recs) == 4
        for r in recs:
            assert r["latent_steps"] == 3
            assert r["num_python_calls"] == 1
            assert len(r["latent_readouts"]) == 1
            assert r["prediction"] in ("33", "77")


def test_t3_c11_mmlu_pro_and_gpqa_diamond_batched_tensorrt_evaluation_with_latent_and_python() -> None:
    """F1 + F2 + F4 + F8: MMLU Pro and GPQA Diamond evaluated via batched TensorRT engine with latent + Python tool trajectories."""
    scripts = {
        "mmlu_t3": "Analyze.<latent:4></latent>Verify.</think><python>print('F')</python><result></result><think>Done.</think><answer>\\boxed{F}</answer>",
        "gpqa_t3": "Quantum.<latent:4></latent>Verify.</think><python>print('C')</python><result></result><think>Done.</think><answer>\\boxed{C}</answer>",
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    enable_tensorrt_acceleration(model, max_batch_size=32)
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

    mmlu_items = [
        {
            "question_id": f"m_{i}",
            "question": f"mmlu_t3 item {i}",
            "options": [f"Opt {chr(65 + j)}" for j in range(10)],
            "answer": "F",
        }
        for i in range(4)
    ]
    gpqa_items = [
        {
            "id": f"g_{i}",
            "Question": f"gpqa_t3 item {i}",
            "options": ["Choice A", "Choice B", "Choice C", "Choice D"],
            "answer": "C",
        }
        for i in range(4)
    ]

    res_mmlu = benchmarks.evaluate_benchmark("mmlu_pro", engine, dataset_items=mmlu_items, batch_size=4)
    res_gpqa = benchmarks.evaluate_benchmark("gpqa_diamond", engine, dataset_items=gpqa_items, batch_size=4)

    assert res_mmlu.accuracy == 100.0
    assert res_mmlu.reference_score == 82.6
    assert res_mmlu.total_latent_steps == 16
    assert res_mmlu.total_tool_calls == 4

    assert res_gpqa.accuracy == 100.0
    assert res_gpqa.reference_score == 82.3
    assert res_gpqa.total_latent_steps == 16
    assert res_gpqa.total_tool_calls == 4


def test_t3_c12_aime_2026_batched_4bit_tensorrt_evaluation_with_strict_zero_tools_and_latent_reasoning() -> None:
    """F1 + F3 + F4 + F8: AIME 2026 on 4-bit TensorRT engine enforces 0 tools while executing latent steps."""
    model = build_tiny_hybrid_model(quantized_4bit=True)
    enable_tensorrt_acceleration(model, max_batch_size=32)
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

    items = build_synthetic_benchmark_items()["aime_2026"]
    res = benchmarks.evaluate_benchmark("aime_2026", engine, dataset_items=items, batch_size=5)

    assert res.accuracy == 100.0
    assert res.reference_score == 88.3
    assert res.total_tool_calls == 0
    assert res.latent_steps_avg == 4.0


def test_t3_c13_livecodebench_v6_batched_tensorrt_generation_and_multiprocess_sandbox_grading() -> None:
    """F1 + F4 + F5 + F8: LiveCodeBench v6 batched TensorRT generation + subprocess sandbox grading."""
    model = build_tiny_hybrid_model()
    enable_tensorrt_acceleration(model, max_batch_size=32)
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)

    items = build_synthetic_benchmark_items()["livecodebench_v6"]
    res = benchmarks.evaluate_benchmark("livecodebench_v6", engine, dataset_items=items, batch_size=2)

    assert res.accuracy == 100.0
    assert res.reference_score == 77.1
    assert res.total == 2
    assert res.correct == 2
    assert all(p.get("passed_tests", 0) >= 2 for p in res.predictions)


def test_t3_c14_mmmu_pro_multimodal_batched_tensorrt_evaluation_across_standard_and_vision_subsets() -> None:
    """F1 + F4 + F6 + F8: MMMU Pro evaluated across standard (multi-image) and vision subsets with TensorRT."""
    model = build_tiny_hybrid_model(with_vision=True)
    enable_tensorrt_acceleration(model, max_batch_size=32)
    engine = HybridEngine(model, model.backbone.tokenizer, processor=model.processor, temperature=0.0)

    items = build_synthetic_benchmark_items()["mmmu_pro"]
    res = benchmarks.evaluate_benchmark("mmmu_pro", engine, dataset_items=items, batch_size=5)

    assert res.accuracy == 100.0
    assert res.reference_score == 73.8
    assert res.total == 5
    assert res.correct == 5
    assert model.backbone.vision_tower.call_count >= 1


def test_t3_c15_g4_concurrency_autotuner_drives_tensorrt_profile_and_run_batch_across_5_benchmarks() -> None:
    """F1 + F3 + F4 + F7 + F8: recommend_g4_concurrency sizes TensorRT profile and batch size on 96GB G4 GPU."""
    rec_bf16 = benchmarks.recommend_g4_concurrency("milkshake721/Gemma-4-26B-A4B-HybridAgent", vram_gb=96.0)
    rec_4bit = benchmarks.recommend_g4_concurrency("milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit", vram_gb=96.0)

    assert rec_bf16["quantized_4bit"] is False
    assert rec_4bit["quantized_4bit"] is True
    assert rec_bf16["batch_size"] >= 32
    assert rec_4bit["batch_size"] >= rec_bf16["batch_size"]
    for b_key in ("mmlu_pro", "aime_2026", "gpqa_diamond", "livecodebench_v6", "mmmu_pro"):
        assert rec_4bit[b_key] >= 16

    model = build_tiny_hybrid_model(quantized_4bit=True)
    info = enable_tensorrt_acceleration(model, max_batch_size=rec_4bit["max_concurrency"])
    assert model.hybridlink.max_batch_size == rec_4bit["max_concurrency"]
    assert info["max_batch_size"] == rec_4bit["max_concurrency"]


def test_t3_c16_unified_5_benchmark_runner_end_to_end_json_and_markdown_artifact_consistency() -> None:
    """F1 + F2 + F4 + F6 + F7 + F8: Unified 5-benchmark suite cross-checks per-benchmark JSON, summary.json, and report.md."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p, include_vision=True)
        out_dir = tmp_p / "suite_out"

        args = eval_hf_benchmarks.parse_args(
            [
                "--model",
                str(ckpt),
                "--backend",
                "tensorrt",
                "--benchmarks",
                "all",
                "--batch-size",
                "4",
                "--synthetic-smoke",
                "--output-dir",
                str(out_dir),
            ]
        )
        summary = eval_hf_benchmarks.run_suite(args)

        assert (out_dir / "summary.json").is_file()
        assert (out_dir / "report.md").is_file()
        md_content = (out_dir / "report.md").read_text(encoding="utf-8")

        for b_name, ref_score in benchmarks.REFERENCE_SCORES.items():
            per_file = out_dir / f"{b_name}.json"
            assert per_file.is_file()
            per_data = json.loads(per_file.read_text(encoding="utf-8"))
            assert per_data["accuracy"] == 100.0
            assert per_data["reference_score"] == ref_score
            assert summary["benchmarks"][b_name]["accuracy"] == per_data["accuracy"]
            assert f"`{b_name}`" in md_content

        assert summary["benchmarks"]["aime_2026"]["total_tool_calls"] == 0


class Tier3CombinationsTestSuite(unittest.TestCase):
    """Unittest wrapper dynamically binding all 16 `test_t3_c*` functions for unittest discovery."""


for _fn_name, _fn_obj in list(globals().items()):
    if _fn_name.startswith("test_t3_c") and callable(_fn_obj):
        setattr(Tier3CombinationsTestSuite, _fn_name, staticmethod(_fn_obj))


if __name__ == "__main__":
    unittest.main()
