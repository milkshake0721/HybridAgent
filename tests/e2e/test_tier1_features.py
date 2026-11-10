"""Tier 1 Feature End-to-End Tests (40 Tests: 5 per Feature F1..F8).

Verifies the happy-path contracts of all 8 core features (F1-F8) across Milestones M1, M2, and M3.
Compatible with both `pytest` and `unittest` (`python3 tests/run_e2e_tests.py --tier 1`).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from PIL import Image
import torch
import torch.nn.functional as F

from tests.e2e.helpers import (
    BYTE_TOKEN_OFFSET,
    PROJECT_ROOT,
    SPECIAL_TOKEN_MAP,
    Gemma4TextExperts,
    SpyTensorRTEngine,
    build_synthetic_benchmark_items,
    build_tiny_hybrid_model,
    build_tiny_tokenizer,
    create_temp_checkpoint_dir,
    ensure_contract_bridges,
    install_runtime_compat,
    reference_4bit_moe,
    reference_bf16_moe,
    run_cli_command,
)

install_runtime_compat()
ensure_contract_bridges()

from hybrid_agent import (
    HybridAgentModel,
    HybridEngine,
    HybridGenerationEngine,
    HybridLink,
    HybridTrajectory,
    SGLangHybridEngine,
    compile_hybridlink_tensorrt,
    enable_tensorrt_acceleration,
    get_sglang_server_command,
    load_hybrid_agent,
    patch_bf16_experts_fastpath,
    prepare_sglang_model_dir,
)
from hybrid_agent.backends import (
    _TensorRTHybridLinkWrapper,
    patch_compiled_ffn_bf16,
)
import hybrid_agent.benchmarks as benchmarks
from hybrid_agent.quantization import (
    Gemma4TextExperts4Bit,
    fused_triton_4bit_moe_forward,
    sort_moe_token_expert_pairs,
)
from hybrid_agent.readout import normalized_embeddings
from hybrid_agent.sampling import sample_rows, token_log_probs
import hybrid_inference.run_batch as run_batch_cli
import sglang_tensorrt_runner


# ==============================================================================
# Feature F1: Dynamic-Batch TensorRT HybridLink Engine (5 Tests)
# ==============================================================================


def test_t1_f1_01_wrapper_executes_trt_engine_for_single_and_batched_2d_inputs() -> None:
    torch.manual_seed(101)
    hidden_size = 64
    base_link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()
    ref_link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()
    ref_link.load_state_dict(base_link.state_dict())

    spy = SpyTensorRTEngine(ref_link)
    base_link.forward = mock.MagicMock(side_effect=AssertionError("Eager fallback to base_link must not occur"))

    wrapper = _TensorRTHybridLinkWrapper(
        base_link,
        spy,
        hidden_size=hidden_size,
        max_batch_size=256,
        min_batch_size=1,
        opt_batch_size=32,
        backend_name="torch_tensorrt (spy)",
    )

    batch_sizes = (1, 4, 16, 32, 64, 256)
    for b in batch_sizes:
        h = torch.randn(b, hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            out = wrapper(h)
            expected = ref_link(h)
        assert out.shape == (b, hidden_size)
        assert out.dtype == torch.bfloat16
        assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)

    assert spy.call_count == len(batch_sizes)
    assert spy.call_shapes == [(b, hidden_size) for b in batch_sizes]
    base_link.forward.assert_not_called()


def test_t1_f1_02_wrapper_flattens_3d_sequence_inputs_and_restores_shape() -> None:
    torch.manual_seed(102)
    hidden_size = 64
    base_link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()
    ref_link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()
    ref_link.load_state_dict(base_link.state_dict())

    spy = SpyTensorRTEngine(ref_link)
    base_link.forward = mock.MagicMock(side_effect=AssertionError("Eager fallback must not occur for 3D inputs"))

    wrapper = _TensorRTHybridLinkWrapper(base_link, spy, hidden_size=hidden_size, max_batch_size=256)

    for shape in ((1, 10, hidden_size), (8, 16, hidden_size)):
        spy.reset_stats()
        h = torch.randn(*shape, dtype=torch.bfloat16)
        with torch.no_grad():
            out = wrapper(h)
            expected = ref_link(h)
        assert out.shape == shape
        assert spy.call_shapes == [(shape[0] * shape[1], hidden_size)]
        assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_t1_f1_03_compile_hybridlink_tensorrt_uses_dynamic_min_opt_max_profile() -> None:
    torch.manual_seed(103)
    hidden_size = 64
    link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()

    mock_trt = types.ModuleType("torch_tensorrt")
    mock_trt.__version__ = "2.11.0"
    mock_trt.Input = mock.MagicMock(return_value="DYNAMIC_PROFILE_INPUT")
    mock_trt.compile = mock.MagicMock(side_effect=lambda mod, **kw: SpyTensorRTEngine(mod))

    with mock.patch.dict(sys.modules, {"torch_tensorrt": mock_trt}):
        wrapper, backend_str = compile_hybridlink_tensorrt(
            link,
            hidden_size=hidden_size,
            dtype=torch.bfloat16,
            device="cpu",
            min_batch_size=1,
            opt_batch_size=32,
            max_batch_size=256,
        )

    assert isinstance(wrapper, _TensorRTHybridLinkWrapper)
    assert "torch_tensorrt" in backend_str
    mock_trt.Input.assert_called_once_with(
        min_shape=(1, hidden_size),
        opt_shape=(32, hidden_size),
        max_shape=(256, hidden_size),
        dtype=torch.bfloat16,
    )
    assert wrapper.min_batch_size == 1
    assert wrapper.opt_batch_size == 32
    assert wrapper.max_batch_size == 256


def test_t1_f1_04_enable_tensorrt_acceleration_configures_dynamic_batch_hybridlink() -> None:
    torch.manual_seed(104)
    model = build_tiny_hybrid_model(hidden_size=64, num_layers=2)
    info = enable_tensorrt_acceleration(
        model,
        max_batch_size=128,
        opt_batch_size=32,
        min_batch_size=1,
    )
    assert isinstance(model.hybridlink, _TensorRTHybridLinkWrapper)
    assert model.hybridlink.max_batch_size == 128
    assert model.hybridlink.opt_batch_size == 32
    assert model.hybridlink.min_batch_size == 1
    assert info["max_batch_size"] == 128
    assert info["hybridlink_backend"]
    assert getattr(model, "_acceleration_info", None) == info

    h_batch = torch.randn(32, 64, dtype=torch.bfloat16)
    out = model.run_hybridlink(h_batch, output_dtype=torch.bfloat16)
    assert out.shape == (32, 64)
    assert out.dtype == torch.bfloat16


def test_t1_f1_05_wrapper_delegates_submodule_attributes_to_base_link() -> None:
    hidden_size = 64
    base_link = HybridLink(hidden_size).to(dtype=torch.bfloat16).eval()
    wrapper = _TensorRTHybridLinkWrapper(base_link, SpyTensorRTEngine(base_link), hidden_size=hidden_size)

    assert wrapper.proj1 is base_link.proj1
    assert wrapper.proj2 is base_link.proj2
    assert wrapper.pre_ln is base_link.pre_ln
    assert wrapper.post_ln is base_link.post_ln
    assert wrapper.act is base_link.act

    ws = wrapper.state_dict()
    for k, v in base_link.state_dict().items():
        assert any(wk.endswith(k) and torch.equal(wv, v) for wk, wv in ws.items())


# ==============================================================================
# Feature F2: High-Concurrency Zero-Sync BF16 MoE & Compiled FFN (5 Tests)
# ==============================================================================


def test_t1_f2_01_bf16_experts_fast_forward_single_token_decode_matches_reference() -> None:
    torch.manual_seed(201)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patched = patch_bf16_experts_fastpath(holder)
    assert patched == 1

    h = torch.randn(1, 64, dtype=torch.bfloat16)
    probs = F.softmax(torch.randn(1, 8), dim=-1)
    top_k_weights, top_k_index = torch.topk(probs, k=2, dim=-1)
    top_k_weights = (top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    with torch.no_grad():
        out_fast = experts(h, top_k_index, top_k_weights)
        out_ref = reference_bf16_moe(h, top_k_index, top_k_weights, experts.gate_up_proj, experts.down_proj)
    assert out_fast.shape == (1, 64)
    assert torch.allclose(out_fast, out_ref, atol=1e-2, rtol=1e-2)


def test_t1_f2_02_bf16_experts_fast_forward_batched_decode_zero_nonzero_sync() -> None:
    torch.manual_seed(202)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    def _forbid_nonzero(*args, **kwargs):
        raise RuntimeError("CPU-GPU sync via Tensor.nonzero() is forbidden in fast MoE path")

    with mock.patch.object(torch.Tensor, "nonzero", _forbid_nonzero):
        for T in (2, 8, 16, 32, 64):
            h = torch.randn(T, 64, dtype=torch.bfloat16)
            probs = F.softmax(torch.randn(T, 8), dim=-1)
            top_k_w, top_k_idx = torch.topk(probs, k=2, dim=-1)
            top_k_w = (top_k_w / top_k_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
            with torch.no_grad():
                out_fast = experts(h, top_k_idx, top_k_w)
                out_ref = reference_bf16_moe(h, top_k_idx, top_k_w, experts.gate_up_proj, experts.down_proj)
            assert out_fast.shape == (T, 64)
            assert torch.allclose(out_fast, out_ref, atol=1e-2, rtol=1e-2)


def test_t1_f2_03_bf16_experts_fast_forward_3d_prefill_tensor_preservation() -> None:
    torch.manual_seed(203)
    experts = Gemma4TextExperts(num_experts=8, hidden_dim=64, intermediate_dim=32, dtype=torch.bfloat16)
    holder = torch.nn.Module()
    holder.experts = experts
    patch_bf16_experts_fastpath(holder)

    h_3d = torch.randn(4, 12, 64, dtype=torch.bfloat16)
    probs = F.softmax(torch.randn(48, 8), dim=-1)
    top_k_w, top_k_idx = torch.topk(probs, k=2, dim=-1)
    top_k_w = (top_k_w / top_k_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    with torch.no_grad():
        out_3d = experts(h_3d, top_k_idx, top_k_w)
        ref_3d = reference_bf16_moe(h_3d, top_k_idx, top_k_w, experts.gate_up_proj, experts.down_proj)
    assert out_3d.shape == (4, 12, 64)
    assert torch.allclose(out_3d, ref_3d, atol=1e-2, rtol=1e-2)


def test_t1_f2_04_patch_bf16_experts_fastpath_patches_all_bf16_moe_layers_only() -> None:
    bf16_model = build_tiny_hybrid_model(num_layers=3, quantized_4bit=False)
    int4_model = build_tiny_hybrid_model(num_layers=3, quantized_4bit=True)
    orig_int4_fwd = int4_model.backbone.model.language_model.layers[0].experts.forward

    patched_bf16 = patch_bf16_experts_fastpath(bf16_model.backbone)
    patched_int4 = patch_bf16_experts_fastpath(int4_model.backbone)

    assert patched_bf16 == 3
    assert patched_int4 == 0
    assert int4_model.backbone.model.language_model.layers[0].experts.forward == orig_int4_fwd


def test_t1_f2_05_compiled_ffn_bf16_step_matches_eager_layer_computation() -> None:
    torch.manual_seed(205)
    model_eager = build_tiny_hybrid_model(hidden_size=64, num_layers=2, quantized_4bit=False, dtype=torch.float32)
    model_comp = build_tiny_hybrid_model(hidden_size=64, num_layers=2, quantized_4bit=False, dtype=torch.float32)
    model_comp.load_state_dict(model_eager.state_dict())

    patched_count = patch_compiled_ffn_bf16(model_comp.backbone)
    assert patched_count == 2

    for B in (1, 16):
        x = torch.randn(B, 3, 64, dtype=torch.float32)
        with torch.no_grad():
            h_e = x
            h_c = x
            for l_e, l_c in zip(
                model_eager.backbone.model.language_model.layers,
                model_comp.backbone.model.language_model.layers,
            ):
                h_e = l_e(h_e)
                h_c = l_c(h_c)
        assert torch.allclose(h_e, h_c, atol=1e-4, rtol=1e-4)


# ==============================================================================
# Feature F3: High-Concurrency Triton W4A16 MoE & SGLang Concurrency Fix (5 Tests)
# ==============================================================================


def test_t1_f3_01_4bit_moe_forward_single_and_batched_numerical_parity() -> None:
    torch.manual_seed(301)
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

    for T in (1, 4, 16, 32):
        h = torch.randn(T, 64, dtype=torch.bfloat16)
        probs = F.softmax(torch.randn(T, 8), dim=-1)
        top_k_w, top_k_idx = torch.topk(probs, k=2, dim=-1)
        top_k_w = (top_k_w / top_k_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

        with torch.no_grad():
            out_mod = experts_4bit(h, top_k_idx, top_k_w)
            out_fn = fused_triton_4bit_moe_forward(
                h,
                top_k_idx,
                top_k_w,
                experts_4bit.gate_up_packed,
                experts_4bit.gate_up_scale,
                experts_4bit.down_packed,
                experts_4bit.down_scale,
            )
            out_ref = reference_4bit_moe(
                h,
                top_k_idx,
                top_k_w,
                experts_4bit.gate_up_packed,
                experts_4bit.gate_up_scale,
                experts_4bit.down_packed,
                experts_4bit.down_scale,
            )
        assert out_mod.shape == (T, 64)
        assert torch.allclose(out_mod, out_fn, atol=1e-2, rtol=1e-2)
        assert torch.allclose(out_mod, out_ref, atol=2e-2, rtol=2e-2)


def test_t1_f3_02_4bit_moe_high_concurrency_expert_sorting_permutation_invariance() -> None:
    torch.manual_seed(302)
    cfg = types.SimpleNamespace(
        num_experts=16,
        hidden_size=64,
        moe_intermediate_size=32,
        hidden_activation="gelu_pytorch_tanh",
    )
    experts_4bit = Gemma4TextExperts4Bit(cfg, dtype=torch.bfloat16)
    experts_4bit.gate_up_packed.random_(0, 2**31 - 1)
    experts_4bit.gate_up_scale.uniform_(-0.03, 0.03)
    experts_4bit.down_packed.random_(0, 2**31 - 1)
    experts_4bit.down_scale.uniform_(-0.03, 0.03)

    T = 32
    h = torch.randn(T, 64, dtype=torch.bfloat16)
    top_k_idx = torch.randint(0, 16, (T, 4))
    top_k_w = F.softmax(torch.randn(T, 4), dim=-1).to(torch.bfloat16)

    sorted_exp, sort_perm, sorted_tok = sort_moe_token_expert_pairs(top_k_idx)
    assert torch.all(sorted_exp[1:] >= sorted_exp[:-1])
    assert torch.equal(top_k_idx.view(-1)[sort_perm], sorted_exp)
    assert torch.equal(sorted_tok, sort_perm // 4)

    perm = torch.randperm(T)
    with torch.no_grad():
        out_orig = experts_4bit(h, top_k_idx, top_k_w)
        out_perm = experts_4bit(h[perm], top_k_idx[perm], top_k_w[perm])
    assert torch.allclose(out_orig[perm], out_perm, atol=1e-2, rtol=1e-2)


def test_t1_f3_03_sglang_hybrid_engine_run_preserves_self_latent_mode_across_calls() -> None:
    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 128
    engine.min_latent_steps = 4
    engine.max_latent_steps = 16
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = threading.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None
    engine.hybridlink = HybridLink(64).to(dtype=torch.bfloat16).eval()
    engine.tools = types.SimpleNamespace(execute=lambda tag, c: "42")

    call_step: dict[str, int] = {}

    def _fake_sgl_gen(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
        qid = "Q1" if "Q1" in prompt else "Q2"
        st = call_step.get(qid, 0)
        call_step[qid] = st + 1
        if st == 0:
            assert return_hidden_states is True
            return {
                "text": "Plan.<latent>",
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": 262144},
                    "hidden_states": [torch.randn(1, 64).tolist()],
                },
            }
        return {
            "text": f"Done.</think><answer>\\boxed{{{qid}_ok}}</answer>",
            "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
        }

    engine._sgl_generate = _fake_sgl_gen
    t1 = engine.run("Solve Q1")
    assert engine.latent_mode == "hybrid"
    assert t1.latent_steps == 4
    assert t1.final_answer == "Q1_ok"

    t2 = engine.run("Solve Q2")
    assert engine.latent_mode == "hybrid"
    assert t2.latent_steps == 4
    assert t2.final_answer == "Q2_ok"


def test_t1_f3_04_sglang_hybrid_engine_run_batch_executes_concurrent_requests_in_order() -> None:
    engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine.max_new_tokens = 128
    engine.min_latent_steps = 4
    engine.max_latent_steps = 16
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.latent_mode = "hybrid"
    engine.latent_open_id = 262144
    engine.latent_close_id = 262145
    engine._hybridlink_lock = threading.Lock()
    engine._fallback_hybrid_engine = None
    engine._sgl_engine = None
    engine.tokenizer = None
    engine.hybridlink = HybridLink(64).to(dtype=torch.bfloat16).eval()
    engine.tools = types.SimpleNamespace(execute=lambda tag, c: "ok")

    lock = threading.Lock()
    step_by_q: dict[int, int] = {}

    def _fake_sgl_gen(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
        idx = int(prompt.split("REQ_")[1].split("_")[0])
        with lock:
            st = step_by_q.get(idx, 0)
            step_by_q[idx] = st + 1
        time.sleep(0.002 * (8 - idx))
        if st == 0:
            return {
                "text": "<latent>",
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": 262144},
                    "hidden_states": [torch.randn(1, 64).tolist()],
                },
            }
        return {
            "text": f"<answer>\\boxed{{ans_{idx}}}</answer>",
            "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
        }

    engine._sgl_generate = _fake_sgl_gen
    questions = [f"REQ_{i}_END" for i in range(8)]
    trajs = engine.run_batch(questions, max_concurrency=4)

    assert len(trajs) == 8
    assert engine.latent_mode == "hybrid"
    for i, tr in enumerate(trajs):
        assert tr.final_answer == f"ans_{i}"
        assert tr["final_answer"] == f"ans_{i}"
        assert tr.latent_steps == 4


def test_t1_f3_05_sglang_model_dir_prep_and_server_command_generation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bf16_dir = create_temp_checkpoint_dir(Path(tmp), name="bf16", quantized_4bit=False)
        int4_dir = create_temp_checkpoint_dir(Path(tmp), name="int4", quantized_4bit=True)

        prep_bf16 = prepare_sglang_model_dir(bf16_dir)
        assert not (prep_bf16 / "adapter_config.json").exists()
        assert (prep_bf16 / "hybridlink_config.json").is_file()
        assert (prep_bf16 / "hybridlink.pt").exists()

        cmd_bf16 = get_sglang_server_command(prep_bf16, port=30001)
        cmd_int4 = get_sglang_server_command(int4_dir, port=30002)

        assert "--quantization" not in cmd_bf16
        assert "--quantization" in cmd_int4
        assert "compressed-tensors" in cmd_int4


# ==============================================================================
# Feature F4: Multi-Question Batched HybridGenerationEngine & HybridEngine.run_batch (5 Tests)
# ==============================================================================


def test_t1_f4_01_multi_question_prefill_left_padding_and_per_row_rope_positions() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    questions = [
        "Short?",
        "Medium length question here?",
        "A much longer third question with many more tokens to test left padding?",
    ]
    engine = HybridGenerationEngine(model, tok, questions, batch_size=1)
    ok = engine._init_generation()
    assert ok is True
    assert engine.B == 3
    assert len(engine.prompt_lens) == 3
    assert engine.prompt_lens[0] < engine.prompt_lens[1] < engine.prompt_lens[2]
    assert engine.positions.tolist() == engine.prompt_lens

    # Verify left-padding: leading 0s followed by 1s in init_attn_mask
    L_max = engine.global_len
    for r_idx, p_len in enumerate(engine.prompt_lens):
        pad_len = L_max - p_len
        mask_row = engine.attn_mask[r_idx].tolist()
        assert mask_row[:pad_len] == [0] * pad_len
        assert mask_row[pad_len:] == [1] * p_len
        assert len(engine.rows[r_idx].kv_ledger) == p_len


def test_t1_f4_02_hybrid_engine_run_batch_returns_trajectories_in_input_order() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)
    questions = ["What is 1+1?", "Solve 2*3", "Compute 10 - 4", "Value of 7?"]
    trajs = engine.run_batch(questions, batch_size=2)

    assert len(trajs) == 4
    assert [t.final_answer for t in trajs] == ["2", "6", "6", "7"]
    for t in trajs:
        assert isinstance(t, HybridTrajectory)
        assert t.latent_steps == 4


def test_t1_f4_03_multi_question_divergent_decode_latent_tool_lockstep() -> None:
    scripts = {
        "Q_LATENT": "Latent step.<latent:4></latent>Done.</think><answer>\\boxed{A0}</answer>",
        "Q_PYTHON": "Run tool.</think><python>print(3 * 7)</python><result></result><think>Got 21.</think><answer>\\boxed{21}</answer>",
        "Q_DIRECT": "Direct.</think><answer>\\boxed{A2}</answer>",
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0, min_latent_steps=2, max_latent_steps=10)

    trajs = engine.run_batch(["Q_LATENT", "Q_PYTHON", "Q_DIRECT"], batch_size=3)
    assert len(trajs) == 3
    assert trajs[0].final_answer == "A0"
    assert trajs[0].latent_steps == 4
    assert trajs[0].latent_spans == 1

    assert trajs[1].final_answer == "21"
    assert trajs[1].raw_result["python_calls"] == 1
    assert "<result>21</result>" in trajs[1].output

    assert trajs[2].final_answer == "A2"
    assert trajs[2].latent_steps == 0


def test_t1_f4_04_multi_question_with_multiple_samples_per_question() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    gen_engine = HybridGenerationEngine(
        model,
        tok,
        ["What is 1+1?", "Solve 2*3"],
        batch_size=2,
        temperature=0.0,
    )
    gen_engine.run()

    assert gen_engine.B == 4
    assert len(gen_engine.rows) == 4
    results = [gen_engine.row_result(r) for r in gen_engine.rows]
    assert [(r["question_idx"], r["sample_idx"]) for r in results] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert results[0]["question"] == "What is 1+1?"
    assert results[2]["question"] == "Solve 2*3"


def test_t1_f4_05_multi_question_latent_readout_rendering() -> None:
    scripts = {
        "Q_READ_1": "Step.<latent:3>ABC</latent>Done.</think><answer>\\boxed{R1}</answer>",
        "Q_READ_2": "Step.<latent:4>WXYZ</latent>Done.</think><answer>\\boxed{R2}</answer>",
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0, record_latent_readout=True)

    trajs = engine.run_batch(["Q_READ_1", "Q_READ_2"], batch_size=2, record_latent_readout=True)
    assert len(trajs) == 2
    for tr, expected_steps in zip(trajs, (3, 4)):
        readouts = tr.raw_result.get("latent_readouts", [])
        assert len(readouts) == 1
        assert readouts[0]["steps"] == expected_steps
        assert readouts[0]["pseudo_text"] != ""
        assert "<latent></latent>" not in tr.output_rendered
        assert f"<latent>{readouts[0]['pseudo_text']}</latent>" in tr.output_rendered


# ==============================================================================
# Feature F5: Engine Loop Hot-Path Optimizations (5 Tests)
# ==============================================================================


def test_t1_f5_01_preallocated_attn_mask_advances_without_per_step_cat_reallocation() -> None:
    model = build_tiny_hybrid_model(
        scripted_responses={"*": "Longer step sequence here.</think><answer>\\boxed{10}</answer>"}
    )
    tok = model.backbone.tokenizer
    engine = HybridGenerationEngine(model, tok, ["Q1", "Q2"], batch_size=1, temperature=0.0)
    engine._init_generation()

    initial_ptr = engine._attn_mask_buf.data_ptr()
    initial_global_len = engine.global_len
    ptrs: list[int] = [initial_ptr]

    for _ in range(10):
        active = [r for r in engine.rows if r.mode != "DONE"]
        if not active:
            break
        engine._decode_step(rng=None)
        ptrs.append(engine._attn_mask_buf.data_ptr())
        assert engine.attn_mask.shape == (2, engine.global_len)
        assert engine.attn_mask.data_ptr() == initial_ptr

    assert len(set(ptrs)) == 1
    assert engine.global_len > initial_global_len


def test_t1_f5_02_multi_token_feed_prefills_tool_observation_in_single_forward() -> None:
    scripts = {
        "Q_TOOL_FEED": "Calc.</think><python>print(123456789)</python><result></result><think>Done.</think><answer>\\boxed{123456789}</answer>"
    }
    model = build_tiny_hybrid_model(scripted_responses=scripts)
    tok = model.backbone.tokenizer
    engine = HybridGenerationEngine(model, tok, "Q_TOOL_FEED", batch_size=1, temperature=0.0)
    engine.run()

    assert model.backbone.multi_token_feed_call_count == 1
    row = engine.rows[0]
    assert row.python_calls == 1
    assert "<result>123456789</result>" in row.full_output
    assert row.positions if hasattr(row, "positions") else int(engine.positions[0].item()) == len(row.kv_ledger)


def test_t1_f5_03_lazy_bf16_readout_bank_skips_allocation_when_disabled_and_caches_bf16() -> None:
    model = build_tiny_hybrid_model()
    tok = model.backbone.tokenizer
    engine_no_ro = HybridGenerationEngine(
        model, tok, "What is 1+1?", batch_size=1, record_latent_readout=False, temperature=0.0
    )
    assert engine_no_ro._readout_bank is None
    engine_no_ro.run()
    assert engine_no_ro._readout_bank is None

    w = model.backbone.get_input_embeddings().weight
    bank1 = normalized_embeddings(w)
    bank2 = normalized_embeddings(w)
    assert bank1.dtype == torch.bfloat16
    assert bank1 is bank2
    norms = torch.norm(bank1.float(), dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_t1_f5_04_sample_rows_vectorized_single_transfer_for_greedy_and_stochastic() -> None:
    B, V = 4, 64
    logits = torch.randn(B, V)
    blocked = torch.zeros(B, V, dtype=torch.bool)

    # 1. Greedy (temperature = 0.0)
    lp_greedy = token_log_probs(logits, temperature=0.0, top_p=1.0, top_k=0, blocked_rows=blocked)
    with mock.patch("torch.multinomial", wraps=torch.multinomial) as spy_multi:
        sampled_greedy = sample_rows(lp_greedy, [0, 1, 2, 3], rng=None, temperature=0.0)
        assert spy_multi.call_count == 0
    expected_greedy = {i: int(logits[i].argmax().item()) for i in range(B)}
    assert sampled_greedy == expected_greedy

    # 2. Stochastic (temperature = 0.8)
    lp_stoch = token_log_probs(logits, temperature=0.8, top_p=0.95, top_k=20, blocked_rows=blocked)
    gen = torch.Generator().manual_seed(42)
    with mock.patch("torch.multinomial", wraps=torch.multinomial) as spy_multi_stoch:
        sampled_stoch = sample_rows(lp_stoch, [0, 1, 2, 3], rng=gen, temperature=0.8)
        assert spy_multi_stoch.call_count == 1
    assert set(sampled_stoch.keys()) == {0, 1, 2, 3}


def test_t1_f5_05_row_tail_gate_skips_tokenizer_decode_on_ordinary_tokens() -> None:
    plain_text = "abcdefghijklmno"
    tok = build_tiny_tokenizer()
    plain_ids = tok.encode(plain_text, add_special_tokens=False) + [tok.eos_token_id]
    model = build_tiny_hybrid_model(tokenizer=tok, scripted_responses={"Q_PLAIN": plain_ids})

    engine = HybridGenerationEngine(model, tok, "Q_PLAIN", batch_size=1, temperature=0.0)
    engine._init_generation()

    tok.decode_call_count = 0
    for ch in plain_text:
        tid = BYTE_TOKEN_OFFSET + ord(ch)
        # Prime the cache so we measure _row_tail decode invocations specifically
        _ = engine._token_may_close_tag(tid)
    tok.decode_call_count = 0

    for _ in range(len(plain_text)):
        engine._decode_step(rng=None)

    # None of 'a'..'o' can close a tag, so _row_tail / tokenizer.decode must not be called during stepping
    assert tok.decode_call_count == 0


# ==============================================================================
# Feature F6: Multimodal Vision-Language Support in HybridAgent (5 Tests)
# ==============================================================================


def test_t1_f6_01_load_hybrid_agent_preserves_vision_tower_and_loads_processor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(
            Path(tmp), include_vision=True, include_processor_config=True
        )
        model, tok = load_hybrid_agent(ckpt, load_processor=True)
        assert isinstance(model, HybridAgentModel)
        assert model.has_vision is True
        assert model.vision_tower is not None
        assert model.embed_vision is not None
        assert model.processor is not None
        assert len(tok) >= 544


def test_t1_f6_02_hybrid_engine_run_accepts_single_pil_image_and_passes_pixel_values() -> None:
    model = build_tiny_hybrid_model(with_vision=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)
    img = Image.new("RGB", (64, 64), color=(120, 80, 200))

    traj = engine.run("Analyze the chart trend in this image.", images=img)
    assert traj.final_answer == "B"
    pv = model.backbone.last_prefill_pixel_values
    assert pv is not None
    assert pv.ndim == 4
    assert pv.shape[0] == 1
    assert pv.shape[1] == 3


def test_t1_f6_03_hybrid_engine_run_accepts_multiple_interleaved_pil_images() -> None:
    model = build_tiny_hybrid_model(with_vision=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)
    img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img2 = Image.new("RGB", (64, 64), color=(0, 255, 0))

    traj = engine.run("Compare <image 1> and <image 2> chart trend.", images=[img1, img2])
    assert traj.final_answer == "B"
    pv = model.backbone.last_prefill_pixel_values
    assert pv is not None
    assert pv.shape[0] == 2
    assert pv.shape[1] == 3


def test_t1_f6_04_hybrid_engine_run_batch_processes_multimodal_question_batch() -> None:
    model = build_tiny_hybrid_model(with_vision=True)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)
    imgs = [
        Image.new("RGB", (64, 64), color=(10, 20, 30)),
        Image.new("RGB", (64, 64), color=(40, 50, 60)),
    ]
    trajs = engine.run_batch(["What is 1+1?", "Solve 2*3"], images=imgs, batch_size=2)
    assert len(trajs) == 2
    assert [t.final_answer for t in trajs] == ["2", "6"]
    assert model.backbone.last_prefill_pixel_values is not None
    assert model.backbone.last_prefill_pixel_values.shape[0] == 2


def test_t1_f6_05_multimodal_trajectory_supports_latent_reasoning_and_python_tool() -> None:
    scripts = {
        "MM_TOOL_Q": "Inspect image.<latent:4></latent>Calc.</think><python>print(15 + 27)</python><result></result><think>Done.</think><answer>\\boxed{42}</answer>"
    }
    model = build_tiny_hybrid_model(with_vision=True, scripted_responses=scripts)
    tok = model.backbone.tokenizer
    engine = HybridEngine(model, tok, temperature=0.0)
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))

    traj = engine.run("MM_TOOL_Q", images=img)
    assert traj.final_answer == "42"
    assert traj.latent_steps == 4
    assert traj.raw_result["python_calls"] == 1
    assert model.backbone.last_prefill_pixel_values is not None


# ==============================================================================
# Feature F7: High-Concurrency Batch Inference CLI & Runner Upgrades (5 Tests)
# ==============================================================================


def test_t1_f7_01_run_batch_cli_defaults_to_tensorrt_and_supports_question_batch_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        in_jsonl = tmp_p / "questions.jsonl"
        out_jsonl = tmp_p / "results.jsonl"

        questions = [
            {"id": 0, "question": "What is 1+1?"},
            {"id": 1, "question": "Solve 2*3"},
            {"id": 2, "question": "Compute 10 - 4"},
            {"id": 3, "question": "Value of 7?"},
        ]
        in_jsonl.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")

        args = run_batch_cli.parse_args(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(out_jsonl), "--question_batch_size", "2", "--num_samples", "2"]
        )
        assert args.backend == "tensorrt"
        assert args.question_batch_size == 2

        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(out_jsonl), "--question_batch_size", "2", "--num_samples", "2", "--temperature", "0.0"]
        )
        lines = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 8
        assert {(r["id"], r["sample_idx"]) for r in lines} == {(q_id, s_id) for q_id in range(4) for s_id in range(2)}


def test_t1_f7_02_run_batch_cli_no_tools_flag_disables_python_and_search() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(
            tmp_p,
            scripted_responses={
                "Try tool": "Attempt tool.</think><python>print(99)</python><result></result><think>Fallback.</think><answer>\\boxed{99}</answer>"
            },
        )
        in_jsonl = tmp_p / "q.jsonl"
        out_jsonl = tmp_p / "out.jsonl"
        in_jsonl.write_text(json.dumps({"id": 0, "question": "Try tool"}) + "\n", encoding="utf-8")

        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(out_jsonl), "--no_tools", "--temperature", "0.0"]
        )
        rec = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
        assert rec["num_tool_calls"] == 0
        assert rec["num_python_calls"] == 0
        assert rec["num_search_calls"] == 0


def test_t1_f7_03_sglang_tensorrt_runner_cli_supports_batch_size_and_multiple_questions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = create_temp_checkpoint_dir(Path(tmp))
        proc = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "sglang_tensorrt_runner.py"),
                "--backend",
                "tensorrt",
                "--model",
                str(ckpt),
                "--questions",
                "What is 1+1?",
                "Solve 2*3",
                "--batch-size",
                "2",
            ]
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        assert "Final Answer: '2'" in proc.stdout
        assert "Final Answer: '6'" in proc.stdout


def test_t1_f7_04_eval_gemma4_26b_hybrid_runs_with_batched_tensorrt_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        out_json = tmp_p / "eval_results.json"

        proc = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_gemma4_26b_hybrid.py"),
                str(ckpt),
                str(out_json),
                "tensorrt",
            ]
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        assert out_json.is_file()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["accuracy"] == 1.0
        assert len(data["results"]) == 10
        assert all(item["correct"] is True for item in data["results"])


def test_t1_f7_05_run_batch_sharding_and_resume_with_question_batch_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p)
        in_jsonl = tmp_p / "questions.jsonl"
        s0_out = tmp_p / "shard0.jsonl"
        s1_out = tmp_p / "shard1.jsonl"

        questions = [{"id": i, "question": f"Problem {i}"} for i in range(4)]
        in_jsonl.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")

        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(s0_out), "--num_shards", "2", "--shard_id", "0", "--question_batch_size", "2", "--temperature", "0.0"]
        )
        # Re-run shard 0 with --resume; should not duplicate records
        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(s0_out), "--num_shards", "2", "--shard_id", "0", "--question_batch_size", "2", "--resume", "--temperature", "0.0"]
        )
        run_batch_cli.main(
            ["--model", str(ckpt), "--input", str(in_jsonl), "--output", str(s1_out), "--num_shards", "2", "--shard_id", "1", "--question_batch_size", "2", "--temperature", "0.0"]
        )

        recs_0 = [json.loads(l) for l in s0_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs_1 = [json.loads(l) for l in s1_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert [r["id"] for r in recs_0] == [0, 2]
        assert [r["id"] for r in recs_1] == [1, 3]


# ==============================================================================
# Feature F8: 5-Benchmark HuggingFace Evaluation Suite on G4 GPU (5 Tests)
# ==============================================================================


def test_t1_f8_01_mmlu_pro_10_choice_formatting_and_grading() -> None:
    model = build_tiny_hybrid_model()
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)
    items = build_synthetic_benchmark_items()["mmlu_pro"]

    res = benchmarks.evaluate_benchmark("mmlu_pro", engine, dataset_items=items, batch_size=4)
    assert res.benchmark == "mmlu_pro"
    assert res.total == 5
    assert res.correct == 5
    assert res.accuracy == 100.0
    assert res.reference_score == 82.6
    assert res.delta_vs_reference > 0


def test_t1_f8_02_aime_2026_no_tools_enforcement_and_integer_grading() -> None:
    model = build_tiny_hybrid_model()
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)
    items = build_synthetic_benchmark_items()["aime_2026"]

    res = benchmarks.evaluate_benchmark("aime_2026", engine, dataset_items=items, batch_size=4)
    assert res.benchmark == "aime_2026"
    assert res.total == 5
    assert res.correct == 5
    assert res.accuracy == 100.0
    assert res.reference_score == 88.3
    assert res.total_tool_calls == 0
    assert all(p["tool_calls"] == 0 for p in res.predictions)


def test_t1_f8_03_gpqa_diamond_4_choice_permutation_and_grading() -> None:
    model = build_tiny_hybrid_model()
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)
    items = build_synthetic_benchmark_items()["gpqa_diamond"]

    res = benchmarks.evaluate_benchmark("gpqa_diamond", engine, dataset_items=items, batch_size=4)
    assert res.benchmark == "gpqa_diamond"
    assert res.total == 5
    assert res.correct == 5
    assert res.accuracy == 100.0
    assert res.reference_score == 82.3


def test_t1_f8_04_livecodebench_v6_code_extraction_and_sandbox_pass_at_1() -> None:
    model = build_tiny_hybrid_model()
    engine = HybridEngine(model, model.backbone.tokenizer, temperature=0.0)
    items = build_synthetic_benchmark_items()["livecodebench_v6"]

    res = benchmarks.evaluate_benchmark("livecodebench_v6", engine, dataset_items=items, batch_size=2)
    assert res.benchmark == "livecodebench_v6"
    assert res.total == 2
    assert res.correct == 2
    assert res.accuracy == 100.0
    assert res.reference_score == 77.1


def test_t1_f8_05_mmmu_pro_multimodal_evaluation_and_unified_summary_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        ckpt = create_temp_checkpoint_dir(tmp_p, include_vision=True)
        out_dir = tmp_p / "bench_out"

        proc = run_cli_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval_hf_benchmarks.py"),
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
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        summary_file = out_dir / "summary.json"
        report_md = out_dir / "report.md"
        assert summary_file.is_file()
        assert report_md.is_file()

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        expected_refs = {
            "mmlu_pro": 82.6,
            "aime_2026": 88.3,
            "gpqa_diamond": 82.3,
            "livecodebench_v6": 77.1,
            "mmmu_pro": 73.8,
        }
        for b_name, ref_val in expected_refs.items():
            assert b_name in summary["benchmarks"]
            b_data = summary["benchmarks"][b_name]
            assert b_data["reference_score"] == ref_val
            assert b_data["accuracy"] == 100.0
            assert str(ref_val) in report_md.read_text(encoding="utf-8")


class Tier1FeaturesTestSuite(unittest.TestCase):
    """Unittest wrapper dynamically binding all 40 `test_t1_f*` functions for unittest discovery."""


for _fn_name, _fn_obj in list(globals().items()):
    if _fn_name.startswith("test_t1_f") and callable(_fn_obj):
        setattr(Tier1FeaturesTestSuite, _fn_name, staticmethod(_fn_obj))


if __name__ == "__main__":
    unittest.main()
