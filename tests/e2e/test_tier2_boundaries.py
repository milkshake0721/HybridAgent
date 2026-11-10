"""Tier 2 Boundary, Edge-Case & Negative End-to-End Tests (40 Tests: 5 per Feature F1..F8).

Verifies boundary conditions, shape limits, fallbacks, malformed inputs, and security/cap guardrails
across Features F1-F8.
Compatible with both `pytest` and `unittest` (`python3 tests/run_e2e_tests.py --tier 2`).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from tests.e2e.helpers import (
    PROJECT_ROOT,
    SPECIAL_TOKEN_MAP,
    Gemma4TextExperts,
    SpyTensorRTEngine,
    build_tiny_hybrid_model,
    build_tiny_tokenizer,
    compress_lcb_private_tests,
    create_temp_checkpoint_dir,
    ensure_contract_bridges,
    install_runtime_compat,
    reference_4bit_moe,
    reference_bf16_moe,
    run_cli_command,
)

install_runtime_compat()
ensure_contract_bridges()

from eval_gemma4_26b_hybrid import normalize_math_answer
from hybrid_agent import (
    HybridAgentModel,
    HybridEngine,
    HybridGenerationEngine,
    HybridLink,
    HybridMultimodalProcessor,
    SGLangHybridEngine,
    ToolExecutor,
    compile_hybridlink_tensorrt,
    extract_answer,
    latent_readout,
    load_hybrid_agent,
    patch_bf16_experts_fastpath,
)
import hybrid_agent.backends as backends_mod
from hybrid_agent.backends import (
    _TensorRTHybridLinkWrapper,
    _run_compiled_ffn_moe_bf16,
)
import hybrid_agent.benchmarks as benchmarks
from hybrid_agent.quantization import (
    Gemma4TextExperts4Bit,
    dequantize_int4_packed,
)
from hybrid_agent.readout import nearest_token_ids, normalized_embeddings
from hybrid_agent.sampling import sample_rows, token_log_probs
import hybrid_inference.run_batch as run_batch_cli


# ==============================================================================
# Feature F1 Boundaries: Batch Chunking, Empty/1D Tensors, Missing TRT & Dtype Casts (5 Tests)
# ==============================================================================


def test_t2_f1_01_wrapper_chunks_inputs_exceeding_max_batch_size() -> None:
    torch.manual_seed(1101)
    H = 64
    base_link = HybridLink(H).to(dtype=torch.bfloat16).eval()
    ref_link = HybridLink(H).to(dtype=torch.bfloat16).eval()
    ref_link.load_state_dict(base_link.state_dict())

    spy = SpyTensorRTEngine(ref_link)
    wrapper = _TensorRTHybridLinkWrapper(base_link, spy, hidden_size=H, max_batch_size=64)

    # 1. 2D [150, H] -> 64 + 64 + 22
    h_2d = torch.randn(150, H, dtype=torch.bfloat16)
    with torch.no_grad():
        out_2d = wrapper(h_2d)
        exp_2d = ref_link(h_2d)
    assert spy.call_sizes == [64, 64, 22]
    assert out_2d.shape == (150, H)
    assert torch.allclose(out_2d, exp_2d, atol=1e-4, rtol=1e-4)

    # 2. 3D [10, 20, H] (200 rows) -> 64 + 64 + 64 + 8
    spy.reset_stats()
    h_3d = torch.randn(10, 20, H, dtype=torch.bfloat16)
    with torch.no_grad():
        out_3d = wrapper(h_3d)
        exp_3d = ref_link(h_3d)
    assert spy.call_sizes == [64, 64, 64, 8]
    assert out_3d.shape == (10, 20, H)
    assert torch.allclose(out_3d, exp_3d, atol=1e-4, rtol=1e-4)


def test_t2_f1_02_wrapper_handles_empty_zero_row_and_1d_tensors_gracefully() -> None:
    H = 64
    base_link = HybridLink(H).to(dtype=torch.bfloat16).eval()
    spy = SpyTensorRTEngine(base_link)
    wrapper = _TensorRTHybridLinkWrapper(base_link, spy, hidden_size=H, max_batch_size=64)

    h_empty_2d = torch.empty((0, H), dtype=torch.bfloat16)
    out_empty_2d = wrapper(h_empty_2d)
    assert out_empty_2d.shape == (0, H)

    h_empty_3d = torch.empty((4, 0, H), dtype=torch.bfloat16)
    out_empty_3d = wrapper(h_empty_3d)
    assert out_empty_3d.shape == (4, 0, H)

    h_1d = torch.randn(H, dtype=torch.bfloat16)
    with torch.no_grad():
        out_1d = wrapper(h_1d)
        exp_1d = base_link(h_1d)
    assert out_1d.shape == (H,)
    assert torch.allclose(out_1d, exp_1d, atol=1e-4, rtol=1e-4)
    assert spy.call_count == 0


def test_t2_f1_03_compile_hybridlink_tensorrt_fallback_chain_when_trt_missing() -> None:
    H = 64
    link = HybridLink(H).to(dtype=torch.bfloat16).eval()

    with (
        mock.patch.dict(sys.modules, {"torch_tensorrt": None}),
        mock.patch("torch.compile", side_effect=RuntimeError("inductor unavailable")),
    ):
        wrapper, backend_name = compile_hybridlink_tensorrt(
            link, hidden_size=H, dtype=torch.bfloat16, device="cpu"
        )

    assert isinstance(wrapper, _TensorRTHybridLinkWrapper)
    assert backend_name == "eager"
    h = torch.randn(16, H, dtype=torch.bfloat16)
    with torch.no_grad():
        out = wrapper(h)
        exp = link(h)
    assert out.shape == (16, H)
    assert torch.allclose(out, exp, atol=1e-5)


def test_t2_f1_04_run_hybridlink_casts_mismatched_input_and_output_dtypes() -> None:
    model = build_tiny_hybrid_model(hidden_size=64, dtype=torch.bfloat16)
    h_fp32 = torch.randn(8, 64, dtype=torch.float32)

    out_fp16 = model.run_hybridlink(h_fp32, output_dtype=torch.float16)
    assert out_fp16.shape == (8, 64)
    assert out_fp16.dtype == torch.float16

    out_bf16 = model.run_hybridlink(h_fp32, output_dtype=torch.bfloat16)
    assert out_bf16.shape == (8, 64)
    assert out_bf16.dtype == torch.bfloat16


def test_t2_f1_05_wrapper_handles_non_contiguous_sliced_batch_views() -> None:
    H = 64
    base_link = HybridLink(H).to(dtype=torch.bfloat16).eval()
    spy = SpyTensorRTEngine(base_link)
    wrapper = _TensorRTHybridLinkWrapper(base_link, spy, hidden_size=H, max_batch_size=64)

    h_full = torch.randn(8, 10, H, dtype=torch.bfloat16)
    h_view = h_full[:, ::2, :]
    assert not h_view.is_contiguous()

    with torch.no_grad():
        out = wrapper(h_view)
        exp = base_link(h_view.contiguous())
    assert out.shape == (8, 5, H)
    assert spy.call_shapes == [(40, H)]
    assert torch.allclose(out, exp, atol=1e-4, rtol=1e-4)


# ==============================================================================
# Feature F2 Boundaries: Empty Tokens, Skewed Routing, Chunking & Fallback (5 Tests)
# ==============================================================================


def test_t2_f2_01_bf16_experts_empty_zero_token_input() -> None:
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    h_empty = torch.empty((0, 64), dtype=torch.bfloat16)
    idx_empty = torch.empty((0, 2), dtype=torch.long)
    w_empty = torch.empty((0, 2), dtype=torch.bfloat16)

    out = experts(h_empty, idx_empty, w_empty)
    assert out.shape == (0, 64)
    assert out.dtype == torch.bfloat16


def test_t2_f2_02_bf16_experts_all_tokens_routed_to_same_vs_disjoint_experts() -> None:
    torch.manual_seed(1202)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    T = 16
    h = torch.randn(T, 64, dtype=torch.bfloat16)
    # 1. All 16 tokens routed to identical experts [0, 1]
    idx_same = torch.tensor([[0, 1]] * T, dtype=torch.long)
    w_same = torch.full((T, 2), 0.5, dtype=torch.bfloat16)

    # 2. Disjoint cyclic expert routing
    idx_disjoint = torch.tensor([[(2 * i) % 8, (2 * i + 1) % 8] for i in range(T)], dtype=torch.long)
    w_disjoint = torch.full((T, 2), 0.5, dtype=torch.bfloat16)

    with torch.no_grad():
        out_same = experts(h, idx_same, w_same)
        ref_same = reference_bf16_moe(h, idx_same, w_same, experts.gate_up_proj, experts.down_proj)
        out_dis = experts(h, idx_disjoint, w_disjoint)
        ref_dis = reference_bf16_moe(h, idx_disjoint, w_disjoint, experts.gate_up_proj, experts.down_proj)

    assert torch.allclose(out_same, ref_same, atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_dis, ref_dis, atol=1e-2, rtol=1e-2)


def test_t2_f2_03_bf16_experts_large_token_count_microbatch_chunking() -> None:
    torch.manual_seed(1203)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    # 65 tokens = 4 chunks of 16 + 1 tail token
    T = 65
    h = torch.randn(T, 64, dtype=torch.bfloat16)
    idx = torch.randint(0, 8, (T, 2))
    w = F.softmax(torch.randn(T, 2), dim=-1).to(torch.bfloat16)

    with torch.no_grad():
        out = experts(h, idx, w)
        ref = reference_bf16_moe(h, idx, w, experts.gate_up_proj, experts.down_proj)
    assert out.shape == (65, 64)
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_t2_f2_04_bf16_experts_zero_weights_and_unnormalized_top_k_weights() -> None:
    torch.manual_seed(1204)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.float32)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    h = torch.randn(8, 64, dtype=torch.float32)
    idx = torch.randint(0, 8, (8, 2))
    w_zero = torch.zeros((8, 2), dtype=torch.float32)
    w_unit = torch.tensor([[1.0, 0.25]] * 8, dtype=torch.float32)
    w_scaled = w_unit * 2.5

    with torch.no_grad():
        out_zero = experts(h, idx, w_zero)
        out_unit = experts(h, idx, w_unit)
        out_scaled = experts(h, idx, w_scaled)

    assert torch.all(out_zero == 0.0)
    assert torch.allclose(out_scaled, out_unit * 2.5, atol=1e-5, rtol=1e-4)


def test_t2_f2_05_compiled_ffn_bf16_fallback_when_compile_fails() -> None:
    torch.manual_seed(1205)
    model = build_tiny_hybrid_model(hidden_size=64, num_layers=1, quantized_4bit=False, dtype=torch.float32)
    lyr = model.backbone.model.language_model.layers[0]
    residual = torch.randn(4, 2, 64, dtype=torch.float32)
    attn_out = torch.randn(4, 2, 64, dtype=torch.float32)

    args = (
        residual,
        attn_out,
        lyr.post_attention_layernorm.weight,
        lyr.pre_feedforward_layernorm.weight,
        lyr.mlp.gate_proj.weight,
        lyr.mlp.up_proj.weight,
        lyr.mlp.down_proj.weight,
        lyr.post_feedforward_layernorm_1.weight,
        lyr.router.scale,
        float(lyr.router.scalar_root_size),
        lyr.router.proj.weight,
        lyr.router.per_expert_scale,
        lyr.pre_feedforward_layernorm_2.weight,
        lyr.experts.gate_up_proj,
        lyr.experts.down_proj,
        lyr.post_feedforward_layernorm_2.weight,
        lyr.post_feedforward_layernorm.weight,
        lyr.layer_scalar,
        float(lyr.input_layernorm.eps),
        int(lyr.router.config.top_k_experts),
    )

    orig_failed = backends_mod._BF16_COMPILE_FAILED
    try:
        backends_mod._BF16_COMPILE_FAILED = False
        with (
            mock.patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True),
            mock.patch.object(backends_mod, "_compiled_ffn_moe_bf16", side_effect=RuntimeError("Simulated compile error")),
        ):
            out = _run_compiled_ffn_moe_bf16(*args)
            assert backends_mod._BF16_COMPILE_FAILED is True
            expected = backends_mod._ffn_moe_bf16_step(*args)
            assert torch.allclose(out, expected, atol=1e-5)
    finally:
        backends_mod._BF16_COMPILE_FAILED = orig_failed


# ==============================================================================
# Feature F3 Boundaries: T=8 vs T=9 Sort Boundary, Int4 Nibbles & SGLang Edge Cases (5 Tests)
# ==============================================================================


def test_t2_f3_01_4bit_moe_sort_threshold_boundary_t8_vs_t9() -> None:
    torch.manual_seed(1301)
    cfg = types.SimpleNamespace(
        num_experts=8,
        hidden_size=64,
        moe_intermediate_size=32,
        hidden_activation="gelu_pytorch_tanh",
    )
    experts_4bit = Gemma4TextExperts4Bit(cfg, dtype=torch.bfloat16)
    experts_4bit.gate_up_packed.random_(0, 2**31 - 1)
    experts_4bit.gate_up_scale.uniform_(-0.04, 0.04)
    experts_4bit.down_packed.random_(0, 2**31 - 1)
    experts_4bit.down_scale.uniform_(-0.04, 0.04)

    h9 = torch.randn(9, 64, dtype=torch.bfloat16)
    idx9 = torch.randint(0, 8, (9, 2))
    w9 = F.softmax(torch.randn(9, 2), dim=-1).to(torch.bfloat16)

    with torch.no_grad():
        out8 = experts_4bit(h9[:8], idx9[:8], w9[:8])
        out9 = experts_4bit(h9, idx9, w9)
    assert out8.shape == (8, 64)
    assert out9.shape == (9, 64)
    assert torch.allclose(out8, out9[:8], atol=1e-2, rtol=1e-2)


def test_t2_f3_02_dequantize_int4_packed_extreme_nibble_values_0_8_15() -> None:
    # 32 elements = 4 int32 words (8 nibbles per int32), 1 group of 32
    val_all_0 = 0x00000000  # nibble 0 -> 0 - 8 = -8
    val_all_8 = int(0x88888888) - (1 << 32)  # nibble 8 -> 8 - 8 = 0
    val_all_f = int(0xFFFFFFFF) - (1 << 32)  # nibble 15 -> 15 - 8 = +7

    packed = torch.tensor(
        [
            [val_all_0] * 4,
            [val_all_8] * 4,
            [val_all_f] * 4,
        ],
        dtype=torch.int32,
    )
    scale = torch.tensor([[0.5], [1.25], [2.0]], dtype=torch.float32)

    unpacked = dequantize_int4_packed(packed, scale)
    assert unpacked.shape == (3, 32)
    assert torch.allclose(unpacked[0], torch.full((32,), -4.0))
    assert torch.allclose(unpacked[1], torch.zeros(32))
    assert torch.allclose(unpacked[2], torch.full((32,), 14.0))


def test_t2_f3_03_sglang_hybrid_engine_text_only_mode_and_zero_max_latent_steps() -> None:
    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 64
    engine.min_latent_steps = 4
    engine.max_latent_steps = 16
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = threading.Lock() if "threading" in dir() else None
    import threading as _th

    engine._hybridlink_lock = _th.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None
    engine.hybridlink = HybridLink(64).to(dtype=torch.bfloat16).eval()
    engine.tools = None

    recorded_params: list[tuple[dict, bool]] = []

    def _fake_sgl_gen(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
        recorded_params.append((dict(sampling_params), return_hidden_states))
        return {
            "text": "Direct text.</think><answer>\\boxed{77}</answer>",
            "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
        }

    engine._sgl_generate = _fake_sgl_gen

    res1 = engine.run("Q_text_only", latent_mode="text_only")
    assert res1.latent_steps == 0
    assert recorded_params[-1][0].get("logit_bias") == {"262144": -100.0}
    assert recorded_params[-1][1] is False

    res2 = engine.run("Q_zero_max", max_latent_steps=0)
    assert res2.latent_steps == 0
    assert recorded_params[-1][0].get("logit_bias") == {"262144": -100.0}
    assert recorded_params[-1][1] is False


def test_t2_f3_04_sglang_hybrid_engine_missing_hidden_states_fallback_and_multi_round_reenable() -> None:
    import threading as _th

    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 64
    engine.min_latent_steps = 3
    engine.max_latent_steps = 12
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = _th.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None
    engine.hybridlink = HybridLink(2816).to(dtype=torch.bfloat16).eval()
    engine.tools = types.SimpleNamespace(execute=lambda tag, code: "4")

    step = 0

    def _fake_sgl_gen(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
        nonlocal step
        step += 1
        if step == 1:
            # Stopped at <latent> but hidden_states is None -> fallback zero tensor
            return {
                "text": "Think 1.<latent>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": 262144}, "hidden_states": None},
            }
        if step == 2:
            return {
                "text": "</think><python>print(2+2)</python>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": "</python>"}},
            }
        if step == 3:
            # After tool execution, <latent> is re-enabled!
            assert return_hidden_states is True
            return {
                "text": "<think>Think 2.<latent>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": 262144}, "hidden_states": None},
            }
        return {
            "text": "Done.</think><answer>\\boxed{4}</answer>",
            "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
        }

    engine._sgl_generate = _fake_sgl_gen
    traj = engine.run("Multi round with missing hidden_states")
    assert traj.final_answer == "4"
    assert traj.latent_spans == 2
    assert traj.latent_steps == 6
    assert "<result>4</result>" in traj.output


def test_t2_f3_05_sglang_hybrid_engine_run_batch_empty_list_and_single_item() -> None:
    import threading as _th

    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 64
    engine.min_latent_steps = 2
    engine.max_latent_steps = 8
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = _th.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None
    engine.hybridlink = HybridLink(64).to(dtype=torch.bfloat16).eval()
    engine.tools = None
    engine._sgl_generate = lambda **kw: {
        "text": "Done.</think><answer>\\boxed{1}</answer>",
        "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
    }

    assert engine.run_batch([]) == []
    single_res = engine.run_batch(["Single question"], max_concurrency=1)
    assert len(single_res) == 1
    assert single_res[0].final_answer == "1"


# ==============================================================================
# Feature F4 Boundaries: Invalid Inputs, Budget Truncation, Guardrails & Caps (5 Tests)
# ==============================================================================


def test_t2_f4_01_run_batch_empty_questions_and_invalid_question_types() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok)

    assert engine.run_batch([]) == []
    with pytest.raises(ValueError):
        HybridGenerationEngine(model, tok, question=None)
    with pytest.raises(TypeError):
        HybridGenerationEngine(model, tok, question=12345)


def test_t2_f4_02_run_batch_images_length_mismatch_raises_value_error() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok)
    imgs = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]

    with pytest.raises(ValueError, match="images"):
        engine.run_batch(["Q1", "Q2", "Q3"], images=imgs)


def test_t2_f4_03_multi_question_early_eos_and_max_response_tokens_budget_truncation() -> None:
    tok = build_tiny_tokenizer()
    scripts = {
        "Q_FAST_EOS": [SPECIAL_TOKEN_MAP["</think>"], tok.eos_token_id],
        "Q_BUDGET_INF": tok.encode("abcdefghijklmnopqrstuvwxyz", add_special_tokens=False),
    }
    model = build_tiny_hybrid_model(tokenizer=tok, scripted_responses=scripts)
    gen_engine = HybridGenerationEngine(
        model,
        tok,
        ["Q_FAST_EOS", "Q_BUDGET_INF"],
        batch_size=1,
        max_response_tokens=5,
        temperature=0.0,
    )
    gen_engine.run()

    r0, r1 = gen_engine.rows[0], gen_engine.rows[1]
    assert r0.end_reason == "eos"
    assert r0.generation_steps == 2

    assert r1.end_reason == "budget"
    assert r1.extra.get("budget_truncated") is True
    assert r1.generation_steps == 5


def test_t2_f4_04_latent_span_min_steps_guard_and_forced_max_steps_exit() -> None:
    scripts = {
        "Q_EARLY_SIGNAL": {
            "turns": ["Start.<latent></latent>End.</think><answer>\\boxed{EARLY_OK}</answer>"],
            "latent_steps": 4,
            "early_exit_at": 2,
            "never_signal_exit": False,
        },
        "Q_NEVER_SIGNAL": {
            "turns": ["Start.<latent></latent>End.</think><answer>\\boxed{FORCED_OK}</answer>"],
            "latent_steps": 10,
            "never_signal_exit": True,
        },
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    tok = model.backbone.tokenizer
    gen_engine = HybridGenerationEngine(
        model,
        tok,
        ["Q_EARLY_SIGNAL", "Q_NEVER_SIGNAL"],
        batch_size=1,
        min_latent_steps=4,
        max_latent_steps=6,
        temperature=0.0,
    )
    gen_engine.run()

    res0 = gen_engine.row_result(gen_engine.rows[0])
    res1 = gen_engine.row_result(gen_engine.rows[1])

    # Q0 ignored early </latent> at step 2 (< min_latent_steps=4) and exited via anchor at step 4
    assert res0["latent_steps"] == 4
    assert res0["anchor_exits"] == 1
    assert res0["forced_exits"] == 0
    assert extract_answer(res0["output"]) == "EARLY_OK"

    # Q1 never emitted </latent> and was forced to exit at max_latent_steps=6
    assert res1["latent_steps"] == 6
    assert res1["anchor_exits"] == 0
    assert res1["forced_exits"] == 1
    assert extract_answer(res1["output"]) == "FORCED_OK"


def test_t2_f4_05_fabricated_result_rejection_and_tool_cap_termination() -> None:
    tok = build_tiny_tokenizer()
    scripts = {
        "Q_FABRICATE": [
            SPECIAL_TOKEN_MAP["</think>"],
            SPECIAL_TOKEN_MAP["<result>"],
            tok.eos_token_id,
        ],
        "Q_EXCEED_CAP": (
            "Call 1.</think><python>print(1)</python><result></result>"
            "<think>Call 2.</think><python>print(2)</python><result></result>"
            "<think>Done.</think><answer>\\boxed{2}</answer>"
        ),
    }
    model = build_tiny_hybrid_model(tokenizer=tok, scripted_responses=scripts)
    gen_engine = HybridGenerationEngine(
        model,
        tok,
        ["Q_FABRICATE", "Q_EXCEED_CAP"],
        batch_size=1,
        max_python_calls=1,
        terminate_on_tool_cap=True,
        reject_fabricated_result=True,
        temperature=0.0,
    )
    gen_engine.run()

    r0, r1 = gen_engine.rows[0], gen_engine.rows[1]
    assert r0.end_reason == "invalid:fabricated_result"
    assert r1.end_reason == "tool_cap"
    assert r1.python_calls == 1


# ==============================================================================
# Feature F5 Boundaries: Mask Buffer Realloc, Split Tags, Uneven FEED & Degenerate Logits (5 Tests)
# ==============================================================================


def test_t2_f5_01_attn_mask_buf_dynamic_growth_when_exceeding_initial_capacity() -> None:
    model = build_tiny_hybrid_model(
        scripted_responses={"Q_GROW": "abcdefghij</think><answer>\\boxed{1}</answer>"}
    )
    tok = model.backbone.tokenizer
    engine = HybridGenerationEngine(model, tok, "Q_GROW", batch_size=1, temperature=0.0)
    engine._init_generation()

    # Force a tiny buffer capacity of global_len + 2 to trigger dynamic growth
    small_cap = engine.global_len + 2
    engine._attn_mask_buf = engine.attn_mask[:, :engine.global_len].clone()
    engine._ensure_attn_mask_capacity(small_cap)
    engine._attn_mask_buf = engine._attn_mask_buf[:, :small_cap].contiguous()
    engine.attn_mask = engine._attn_mask_buf[:, :engine.global_len]

    for _ in range(8):
        if engine.rows[0].mode == "DONE":
            break
        engine._decode_step(rng=None)

    assert engine._attn_mask_buf.shape[1] >= engine.global_len
    assert engine.attn_mask.shape == (1, engine.global_len)
    assert torch.all(engine.attn_mask == 1)


def test_t2_f5_02_row_tail_gate_handles_split_closing_tags_and_trailing_whitespace() -> None:
    tok = build_tiny_tokenizer()
    # Emit "</py" (272) followed by "thon>\n" (273)
    turn1_ids = (
        tok.encode("Calc.</think><python>print(8 + 9)", add_special_tokens=False)
        + [SPECIAL_TOKEN_MAP["</py"], SPECIAL_TOKEN_MAP["thon>\n"]]
    )
    turn2_ids = tok.encode("<think>Done.</think><answer>\\boxed{17}</answer>", add_special_tokens=False) + [
        tok.eos_token_id
    ]
    model = build_tiny_hybrid_model(tokenizer=tok)

    engine = HybridGenerationEngine(model, tok, "Q_SPLIT", batch_size=1, temperature=0.0)
    assert engine._token_may_close_tag(SPECIAL_TOKEN_MAP["</py"]) is False
    assert engine._token_may_close_tag(SPECIAL_TOKEN_MAP["thon>\n"]) is True

    # Feed turn1_ids through _RowScriptState
    from tests.e2e.helpers import _RowScriptState

    model.backbone.scripted_responses["Q_SPLIT"] = " dummy "
    orig_resolve = model.backbone._resolve_row_script
    model.backbone._resolve_row_script = lambda _: _RowScriptState(
        turns=[
            [("tok", t) for t in turn1_ids],
            [("tok", t) for t in turn2_ids],
        ]
    )
    try:
        engine.run()
    finally:
        model.backbone._resolve_row_script = orig_resolve

    res = engine.row_result(engine.rows[0])
    assert res["python_calls"] == 1
    assert "<result>17</result>" in res["output"]
    assert extract_answer(res["output"]) == "17"


def test_t2_f5_03_multi_token_feed_uneven_observation_lengths_and_empty_result() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    engine = HybridGenerationEngine(model, tok, ["Q0", "Q1", "Q2"], batch_size=1, temperature=0.0)
    engine._init_generation()

    pos_before = engine.positions.clone()
    long_obs = tok.encode("<result>0123456789</result>", add_special_tokens=False)  # 12 tokens
    short_obs = tok.encode("<result></result>", add_special_tokens=False)  # 2 tokens

    engine.rows[0].mode = "FEED"
    engine.rows[0].feed_queue = list(long_obs)
    engine.rows[1].mode = "FEED"
    engine.rows[1].feed_queue = list(short_obs)
    engine.rows[2].mode = "DONE"

    engine._multi_token_feed_step()

    assert engine.rows[0].mode == "DECODE"
    assert engine.rows[1].mode == "DECODE"
    assert engine.rows[2].mode == "DONE"
    assert int(engine.positions[0].item()) == int(pos_before[0].item()) + len(long_obs)
    assert int(engine.positions[1].item()) == int(pos_before[1].item()) + len(short_obs)
    assert int(engine.positions[2].item()) == int(pos_before[2].item())


def test_t2_f5_04_sample_rows_all_tokens_blocked_degenerate_fallback() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, -1.0], [3.0, 0.0, 9.0, 1.0]])
    blocked_all = torch.ones_like(logits, dtype=torch.bool)
    lp = token_log_probs(logits, temperature=0.7, top_p=0.95, top_k=20, blocked_rows=blocked_all)
    assert torch.all(~torch.isfinite(lp))

    sampled_nan = sample_rows(lp, [0, 1], rng=torch.Generator().manual_seed(0), temperature=0.7)
    assert sampled_nan == {0: 0, 1: 0}

    lp_neginf = torch.full_like(logits, float("-inf"))
    sampled_neginf = sample_rows(lp_neginf, [0, 1], rng=torch.Generator().manual_seed(0), temperature=0.7)
    assert sampled_neginf == {0: 0, 1: 0}


def test_t2_f5_05_normalized_embeddings_cache_invalidation_on_resize_or_empty_readout() -> None:
    model = build_tiny_hybrid_model(vocab_size=544, hidden_size=64)
    tok = model.backbone.tokenizer
    w1 = model.backbone.get_input_embeddings().weight
    bank1 = normalized_embeddings(w1)
    assert bank1.shape == (len(tok), 64)

    tok.add_special_tokens({"additional_special_tokens": ["<extra_custom_tok>"]})
    model.resize_to_tokenizer(tok)
    w2 = model.backbone.get_input_embeddings().weight
    bank2 = normalized_embeddings(w2)
    assert bank2.shape == (len(tok), 64)
    assert bank2.shape[0] == bank1.shape[0] + 1

    empty_lat = torch.empty((0, 64), dtype=torch.bfloat16)
    assert nearest_token_ids(empty_lat, bank2).shape == (0,)
    assert latent_readout(empty_lat, w2, tok) == []


# ==============================================================================
# Feature F6 Boundaries: Missing processor_config.json, Mixed Batch, Non-RGB & Errors (5 Tests)
# ==============================================================================


def test_t2_f6_01_missing_processor_config_json_synthesizes_hybrid_multimodal_processor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(
            Path(tmp), include_processor_config=False, include_vision=True
        )
        assert not (ckpt / "processor_config.json").exists()

        model, tok = load_hybrid_agent(ckpt, load_processor=True)
        assert model.processor is not None
        img = Image.new("RGB", (48, 48), color=(90, 120, 150))
        proc_out = model.processor(text="Hello vision", images=img, return_tensors="pt")
        assert "input_ids" in proc_out
        assert "pixel_values" in proc_out
        assert proc_out["pixel_values"].shape[0] == 1
        assert proc_out["pixel_values"].shape[1] == 3


def test_t2_f6_02_mixed_batch_multimodal_and_text_only_questions() -> None:
    model = build_tiny_hybrid_model(with_vision=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)

    img1 = Image.new("RGB", (64, 64), color=(100, 0, 0))
    img3 = Image.new("RGB", (64, 64), color=(0, 100, 0))
    questions = ["What is 1+1?", "Solve 2*3", "Compute 10 - 4"]
    images = [img1, None, img3]

    trajs = engine.run_batch(questions, images=images, batch_size=3)
    assert len(trajs) == 3
    assert [t.final_answer for t in trajs] == ["2", "6", "6"]
    assert model.backbone.last_prefill_pixel_values is not None
    assert model.backbone.last_prefill_pixel_values.shape[0] == 2


def test_t2_f6_03_non_rgb_grayscale_rgba_and_extreme_aspect_ratio_images() -> None:
    model = build_tiny_hybrid_model(with_vision=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)

    img_gray = Image.new("L", (16, 16), color=128)
    img_rgba = Image.new("RGBA", (32, 32), color=(10, 20, 30, 200))
    img_wide = Image.new("RGB", (512, 8), color=(200, 100, 50))

    for img in (img_gray, img_rgba, img_wide):
        traj = engine.run("Analyze the chart trend.", images=img)
        assert traj.final_answer == "B"
        pv = model.backbone.last_prefill_pixel_values
        assert pv is not None
        assert pv.shape == (1, 3, 64, 64)


def test_t2_f6_04_require_hybridlink_true_vs_false_when_weights_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(Path(tmp), include_hybridlink=False)
        with pytest.raises(FileNotFoundError):
            load_hybrid_agent(ckpt, require_hybridlink=True)

        model, tok = load_hybrid_agent(ckpt, require_hybridlink=False)
        assert isinstance(model, HybridAgentModel)
        assert isinstance(model.hybridlink, HybridLink)


def test_t2_f6_05_invalid_hybridlink_adapter_type_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(
            Path(tmp), include_hybridlink=True, adapter_type="invalid_lora"
        )
        with pytest.raises(ValueError, match="unsupported HybridLink config"):
            HybridAgentModel.from_pretrained(ckpt)


# ==============================================================================
# Feature F7 Boundaries: Malformed JSONL, Crash Resume, Limits & Duplicate Tool Queries (5 Tests)
# ==============================================================================


def test_t2_f7_01_run_batch_malformed_jsonl_missing_question_and_blank_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        valid_with_blanks = tmp_p / "blanks.jsonl"
        valid_with_blanks.write_text(
            '\n{"question": "First Q"}\n\n   \n{"id": 10, "question": "Second Q"}\n',
            encoding="utf-8",
        )
        qs = run_batch_cli.read_questions(str(valid_with_blanks))
        assert len(qs) == 2
        assert qs[0]["id"] == 0
        assert qs[1]["id"] == 10

        bad_jsonl = tmp_p / "bad.jsonl"
        bad_jsonl.write_text('{"id": 99, "not_a_question": "oops"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="record without a question"):
            run_batch_cli.read_questions(str(bad_jsonl))


def test_t2_f7_02_run_batch_corrupted_resume_output_file_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        in_jsonl = tmp_p / "in.jsonl"
        out_jsonl = tmp_p / "out.jsonl"

        in_jsonl.write_text(
            "\n".join(json.dumps({"id": i, "question": f"Problem {i}"}) for i in range(3)) + "\n",
            encoding="utf-8",
        )
        # Pre-populate 2 valid lines + 1 truncated corrupted line
        valid_lines = [
            json.dumps({"id": 0, "sample_idx": 0, "question": "Problem 0", "output": "ok", "prediction": "ans_0"}),
            json.dumps({"id": 1, "sample_idx": 0, "question": "Problem 1", "output": "ok", "prediction": "ans_1"}),
        ]
        out_jsonl.write_text("\n".join(valid_lines) + '\n{"id": 2, "sample_idx": 0, "outp', encoding="utf-8")

        done = run_batch_cli.read_done(str(out_jsonl))
        assert done == {(0, 0), (1, 0)}

        # Clean truncated tail and run with --resume so only id=2 is executed
        out_jsonl.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(out_jsonl), "--resume", "--temperature", "0.0"]
        )
        final_recs = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert [r["id"] for r in final_recs] == [0, 1, 2]
        assert final_recs[2]["prediction"] == "ans_2"


def test_t2_f7_03_run_batch_limit_and_uneven_question_batch_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        in_jsonl = tmp_p / "in7.jsonl"
        out_jsonl = tmp_p / "out5.jsonl"

        in_jsonl.write_text(
            "\n".join(json.dumps({"id": i, "question": f"Problem {i}"}) for i in range(7)) + "\n",
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
                "--limit",
                "5",
                "--question_batch_size",
                "2",
                "--num_samples",
                "1",
                "--temperature",
                "0.0",
            ]
        )
        recs = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(recs) == 5
        assert [r["id"] for r in recs] == [0, 1, 2, 3, 4]


def test_t2_f7_04_sglang_tensorrt_runner_serve_cmd_mode_offline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_4bit = create_temp_checkpoint_dir(Path(tmp), quantized_4bit=True)
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
        assert proc.returncode == 0
        assert "sglang.launch_server" in proc.stdout
        assert "--quantization compressed-tensors" in proc.stdout


def test_t2_f7_05_duplicate_search_query_blocked_and_tool_cap_observation() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer

    search_calls = 0

    class _DummySearchTool:
        name = "dummy_search"
        trigger_tag = "search"

        def execute(self, content: str, **kwargs) -> str:
            nonlocal search_calls
            search_calls += 1
            return f"Page 1: Result for {content}"

    tools = ToolExecutor()
    tools.register_tool(_DummySearchTool())  # type: ignore[arg-type]

    engine = HybridGenerationEngine(
        model,
        tok,
        "Q_SEARCH",
        batch_size=1,
        tool_executor=tools,
        max_tool_calls=1,
        max_search_calls=2,
        terminate_on_tool_cap=False,
    )
    row = engine.rows[0]

    # 1st search call: executes tool
    row.pending_tool = ("search", "quantum gravity")
    obs1 = engine._execute_tool(row)
    assert obs1 == "<result>Page 1: Result for quantum gravity</result>"
    assert search_calls == 1
    assert row.search_calls == 1

    # Increase max_tool_calls to 2 to test duplicate query guard
    engine.max_tool_calls = 2
    row.pending_tool = ("search", "quantum gravity")
    obs2 = engine._execute_tool(row)
    assert obs2 == "<result>You have searched this query. Please refer to previous results.</result>"
    assert search_calls == 1  # Backend not called a second time

    # Now exceed max_tool_calls (set max_tool_calls=1 while row.search_calls == 1)
    engine.max_tool_calls = 1
    row.pending_tool = ("search", "new query")
    obs3 = engine._execute_tool(row)
    assert "The maximum tool call limit is exceeded." in obs3


# ==============================================================================
# Feature F8 Boundaries: AIME Strict No-Tools, Gated/Offline Fallbacks & Malformed Answers (5 Tests)
# ==============================================================================


def test_t2_f8_01_aime_2026_strict_no_tools_rejects_python_and_search_tags() -> None:
    scripts = {
        "AIME_TRY_TOOL": (
            "Try tool.</think><python>print(100 + 23)</python><result></result>"
            "<think>Fallback.</think><answer>\\boxed{123}</answer>"
        )
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)
    aime_items = [{"id": "aime_strict_1", "problem": "AIME_TRY_TOOL", "answer": "123"}]

    res = benchmarks.evaluate_benchmark("aime_2026", engine, dataset_items=aime_items, batch_size=1)
    assert res.total == 1
    assert res.total_tool_calls == 0
    assert res.predictions[0]["python_calls"] == 0
    assert res.predictions[0]["search_calls"] == 0


def test_t2_f8_02_gated_gpqa_and_offline_dataset_fallback_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        local_gpqa = tmp_p / "gpqa_diamond.jsonl"
        local_gpqa.write_text(
            json.dumps(
                {
                    "id": "gpqa_local_0",
                    "Question": "What is the spin of a photon?",
                    "Correct Answer": "1",
                    "Incorrect Answer 1": "0",
                    "Incorrect Answer 2": "1/2",
                    "Incorrect Answer 3": "2",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Simulate HF Hub raising gated repo error on Idavidrein/gpqa, falling back to local dataset_dir
        mock_hf = types.ModuleType("datasets")
        mock_hf.load_dataset = mock.MagicMock(side_effect=PermissionError("403 Client Error: GatedRepoError"))

        with (
            mock.patch.dict(sys.modules, {"datasets": mock_hf}),
            mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}),
        ):
            items_local = benchmarks.load_benchmark_dataset("gpqa_diamond", dataset_dir=tmp_p)
            assert len(items_local) == 1
            assert items_local[0]["id"] == "gpqa_local_0"
            assert len(items_local[0]["options"]) == 4

            # Without dataset_dir, falls back to synthetic items without crashing
            items_syn = benchmarks.load_benchmark_dataset("gpqa_diamond", dataset_dir=tmp_p / "nonexistent")
            assert len(items_syn) >= 1


def test_t2_f8_03_malformed_boxed_answers_and_mcq_fallback_extraction() -> None:
    # 1. Nested braces \boxed{\frac{3}{8}} and \boxed{\text{D}}
    ans_frac = extract_answer("<answer>Therefore \\boxed{\\frac{3}{8}}</answer>")
    assert ans_frac == "\\frac{3}{8}"
    assert normalize_math_answer(ans_frac) == "3/8"

    ans_text = extract_answer("<answer>\\boxed{\\text{D}}</answer>")
    assert ans_text == "D"

    # 2. Multiple \boxed{wrong} ... \boxed{99} selects the last \boxed{}
    ans_multi = extract_answer("<answer>First \\boxed{12}, wait no, \\boxed{99}</answer>")
    assert ans_multi == "99"

    # 3. Unclosed \boxed{42 on AIME 2026 and leading zeros "007" == "7", "000" == "0"
    ok_unclosed, pred_unclosed = benchmarks.graders.grade_aime_2026("<answer>\\boxed{42", "42")
    assert ok_unclosed is True
    assert pred_unclosed == "042"

    ok_007, _ = benchmarks.graders.grade_aime_2026("<answer>\\boxed{007}</answer>", "7")
    ok_000, _ = benchmarks.graders.grade_aime_2026("<answer>\\boxed{000}</answer>", "0")
    assert ok_007 is True
    assert ok_000 is True

    # 4. Unboxed MCQ fallback ("The answer is (J).") on 10-choice MMLU Pro
    choice_j = benchmarks.graders.extract_mcq_choice("After checking all options, the answer is (J).", num_choices=10)
    assert choice_j == "J"


def test_t2_f8_04_livecodebench_v6_malformed_code_blocks_and_infinite_loop_timeout() -> None:
    # 1. Multiple ```python blocks in <think> and <answer> -> extracts last block from <answer>
    multi_block_output = (
        "<think>Draft:\n```python\nprint('wrong')\n```\n</think>"
        "<answer>First try:\n```python\nprint('still_wrong')\n```\n"
        "Final solution:\n```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```</answer>"
    )
    extracted = benchmarks.graders.extract_code_block(multi_block_output)
    assert "still_wrong" not in extracted
    assert "* 2" in extracted

    # 2. Unclosed ```python fence
    unclosed_output = "<answer>```python\nprint(42)\n"
    assert benchmarks.graders.extract_code_block(unclosed_output) == "print(42)"

    # 3. Compressed private_test_cases decoding + infinite loop timeout enforcement
    priv_cases = [{"input": "5\n", "output": "10\n", "testtype": "stdin"}]
    raw_item = {
        "question_id": "lcb_inf_loop",
        "question_content": "Double the input.",
        "starter_code": "",
        "public_test_cases": "[]",
        "private_test_cases": compress_lcb_private_tests(priv_cases),
    }
    norm_item = benchmarks.datasets.normalize_livecodebench_v6_item(raw_item)
    assert len(norm_item["test_cases"]) == 1
    assert norm_item["test_cases"][0]["output"] == "10\n"

    # Candidate with infinite loop terminates within timeout
    t0 = time.time()
    passed, _, meta = benchmarks.graders.grade_livecodebench_v6(
        "<answer>```python\nwhile True:\n    pass\n```</answer>",
        norm_item,
        timeout_per_test=0.5,
    )
    elapsed = time.time() - t0
    assert passed is False
    assert meta["reason"] == "timeout"
    assert elapsed < 2.5


def test_t2_f8_05_mmlu_pro_na_options_filtering_and_gpqa_deterministic_shuffle() -> None:
    raw_mmlu = {
        "question_id": "mmlu_na_test",
        "question": "Which law applies?",
        "options": ["Option 1", "Option 2", "N/A", "Option 3", "n/a", "Option 4"],
        "answer": "B",
    }
    norm_mmlu = benchmarks.datasets.normalize_mmlu_pro_item(raw_mmlu)
    assert norm_mmlu["options"] == ["Option 1", "Option 2", "Option 3", "Option 4"]
    prompt_mmlu = benchmarks.prompts.build_benchmark_prompt("mmlu_pro", norm_mmlu)
    assert "N/A" not in prompt_mmlu
    assert "(D) Option 4" in prompt_mmlu

    raw_gpqa = {
        "id": "gpqa_shuffle_1",
        "Question": "What is the ground state energy?",
        "Correct Answer": "E_0 = hw/2",
        "Incorrect Answer 1": "E_0 = 0",
        "Incorrect Answer 2": "E_0 = hw",
        "Incorrect Answer 3": "E_0 = 2hw",
    }
    g1 = benchmarks.datasets.normalize_gpqa_diamond_item(raw_gpqa, seed=42)
    g2 = benchmarks.datasets.normalize_gpqa_diamond_item(raw_gpqa, seed=42)
    assert g1["options"] == g2["options"]
    assert g1["answer"] == g2["answer"]
    gold_idx = ord(g1["answer"]) - 65
    assert g1["options"][gold_idx] == "E_0 = hw/2"


class Tier2BoundariesTestSuite(unittest.TestCase):
    """Unittest wrapper dynamically binding all 40 `test_t2_f*` functions for unittest discovery."""


for _fn_name, _fn_obj in list(globals().items()):
    if _fn_name.startswith("test_t2_f") and callable(_fn_obj):
        setattr(Tier2BoundariesTestSuite, _fn_name, staticmethod(_fn_obj))


if __name__ == "__main__":
    unittest.main()
