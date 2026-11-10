"""Comprehensive unit tests for Milestone M1:
  - Feature F1: Dynamic-Batch TensorRT `HybridLink` (`_TensorRTHybridLinkWrapper`,
    `compile_hybridlink_tensorrt`, `enable_tensorrt_acceleration`)
  - Feature F2: High-Concurrency Zero-Sync BF16 MoE & Compiled BF16 FFN
    (`_fused_bf16_moe_kernel`, `fused_triton_bf16_moe_forward`, `_zero_sync_bf16_moe_pytorch`,
    `fused_bf16_moe_op`, `_bf16_experts_fast_forward`, `patch_bf16_experts_fastpath`,
    `patch_compiled_ffn_bf16`)
  - Feature F3: High-Concurrency Triton W4A16 MoE (`sort_moe_token_expert_pairs`,
    `dequantize_int4_packed`, `_pytorch_zero_sync_4bit_moe_forward`,
    `fused_triton_4bit_moe_forward`, `Gemma4TextExperts4Bit.forward`) &
    `SGLangHybridEngine` Concurrency Fix (`run` non-mutating `latent_mode` & `run_batch`)
"""
from __future__ import annotations

from pathlib import Path
import random
import sys
import threading
import time
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hybrid_agent  # noqa: F401
except ImportError:
    _pkg = types.ModuleType("hybrid_agent")
    _pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "hybrid_agent")]
    _pkg.__package__ = "hybrid_agent"
    sys.modules["hybrid_agent"] = _pkg

import hybrid_agent.backends as backends
import hybrid_agent.quantization as quantization
from hybrid_agent.backends import (
     SGLangHybridEngine,
    _TensorRTHybridLinkWrapper,
    _bf16_experts_fast_forward,
    _ffn_moe_bf16_step,
    _zero_sync_bf16_moe_pytorch,
    compile_hybridlink_tensorrt,
    enable_tensorrt_acceleration,
    fused_bf16_moe_op,
    fused_triton_bf16_moe_forward,
    patch_bf16_experts_fastpath,
    patch_compiled_ffn_4bit,
    patch_compiled_ffn_bf16,
)
from hybrid_agent.engine import HybridTrajectory
from hybrid_agent.hybridlink import HybridLink
from hybrid_agent.quantization import (
    GROUP_SIZE,
    SORT_EXPERT_THRESHOLD,
    Gemma4TextExperts4Bit,
    _pytorch_zero_sync_4bit_moe_forward,
    dequantize_int4_packed,
    fused_triton_4bit_moe_forward,
    sort_moe_token_expert_pairs,
)


def _reference_bf16_moe(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn=None,
) -> torch.Tensor:
    """Reference per-expert BF16 MoE forward pass for ground-truth numerical comparison."""
    if act_fn is None:
        act_fn = lambda x: F.gelu(x, approximate="tanh")
    orig_shape = hidden_states.shape
    x_2d = hidden_states.reshape(-1, orig_shape[-1])
    T, H = x_2d.shape
    K = top_k_index.shape[-1]
    idx_2d = top_k_index.reshape(T, K)
    w_2d = top_k_weights.reshape(T, K)
    out = torch.zeros((T, H), dtype=torch.float32, device=x_2d.device)
    for t in range(T):
        xt = x_2d[t : t + 1].to(torch.float32)
        for k in range(K):
            e = int(idx_2d[t, k].item())
            w = float(w_2d[t, k].item())
            gu = F.linear(xt, gate_up_proj[e].to(torch.float32))
            gate, up = gu.chunk(2, dim=-1)
            inter = act_fn(gate) * up
            dn = F.linear(inter, down_proj[e].to(torch.float32))
            out[t : t + 1] += dn * w
    return out.to(hidden_states.dtype).view(orig_shape)


def _reference_4bit_moe(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gu_packed: torch.Tensor,
    gu_scale: torch.Tensor,
    dn_packed: torch.Tensor,
    dn_scale: torch.Tensor,
    act_fn=None,
) -> torch.Tensor:
    """Reference per-expert W4A16 MoE forward pass for ground-truth numerical comparison."""
    if act_fn is None:
        act_fn = lambda x: F.gelu(x, approximate="tanh")
    orig_shape = hidden_states.shape
    x_2d = hidden_states.reshape(-1, orig_shape[-1])
    T, H = x_2d.shape
    K = top_k_index.shape[-1]
    idx_2d = top_k_index.reshape(T, K)
    w_2d = top_k_weights.reshape(T, K)
    out = torch.zeros((T, H), dtype=torch.float32, device=x_2d.device)
    for t in range(T):
        xt = x_2d[t : t + 1]
        for k in range(K):
            e = int(idx_2d[t, k].item())
            w = float(w_2d[t, k].item())
            w_gu = dequantize_int4_packed(gu_packed[e], gu_scale[e]).to(xt.dtype)
            gu = F.linear(xt, w_gu)
            gate, up = gu.chunk(2, dim=-1)
            inter = act_fn(gate) * up
            w_dn = dequantize_int4_packed(dn_packed[e], dn_scale[e]).to(inter.dtype)
            dn = F.linear(inter, w_dn)
            out[t : t + 1] += dn.to(torch.float32) * w
    return out.to(hidden_states.dtype).view(orig_shape)


class _SyntheticGemma4BF16Experts(nn.Module):
    """Synthetic BF16 Gemma4TextExperts module with gate_up_proj and down_proj."""

    def __init__(self, num_experts: int = 16, hidden_dim: int = 64, intermediate_dim: int = 32, dtype=torch.float32):
        super().__init__()
        self.__class__.__name__ = "Gemma4TextExperts"
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=dtype) * 0.05
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype) * 0.05
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        return _reference_bf16_moe(
            hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj, self.act_fn
        )


class _SyntheticRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.with_scale = True
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _SyntheticSelfAttn(nn.Module):
    def __init__(self, dim: int, dtype=torch.float32):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return self.proj(hidden_states), None


class _SyntheticSharedMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, dtype=torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class _SyntheticRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int = 4, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.config = types.SimpleNamespace(top_k_experts=top_k)
        self.eps = eps
        self.scalar_root_size = float(dim ** -0.5)
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.proj = nn.Linear(dim, num_experts, bias=False, dtype=dtype)
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts, dtype=dtype))

    def forward(self, h_flat: torch.Tensor):
        D = h_flat.shape[-1]
        r_norm = F.rms_norm(h_flat, (D,), None, self.eps) * self.scale * self.scalar_root_size
        scores = self.proj(r_norm)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


class _SyntheticDecoderLayer(nn.Module):
    def __init__(self, dim: int = 64, inter_dim: int = 32, num_experts: int = 16, top_k: int = 4, is_4bit: bool = False, dtype=torch.float32):
        super().__init__()
        self.input_layernorm = _SyntheticRMSNorm(dim, dtype=dtype)
        self.self_attn = _SyntheticSelfAttn(dim, dtype=dtype)
        self.post_attention_layernorm = _SyntheticRMSNorm(dim, dtype=dtype)
        self.pre_feedforward_layernorm = _SyntheticRMSNorm(dim, dtype=dtype)
        self.mlp = _SyntheticSharedMLP(dim, inter_dim, dtype=dtype)
        self.post_feedforward_layernorm_1 = _SyntheticRMSNorm(dim, dtype=dtype)
        self.router = _SyntheticRouter(dim, num_experts, top_k=top_k, dtype=dtype)
        self.pre_feedforward_layernorm_2 = _SyntheticRMSNorm(dim, dtype=dtype)
        if is_4bit:
            cfg = types.SimpleNamespace(
                num_experts=num_experts,
                hidden_size=dim,
                moe_intermediate_size=inter_dim,
                hidden_activation="gelu_pytorch_tanh",
            )
            self.experts = Gemma4TextExperts4Bit(cfg, dtype=dtype)
            self.experts.gate_up_packed.random_(0, 2**31 - 1)
            self.experts.gate_up_scale.uniform_(-0.02, 0.02)
            self.experts.down_packed.random_(0, 2**31 - 1)
            self.experts.down_scale.uniform_(-0.02, 0.02)
        else:
            self.experts = _SyntheticGemma4BF16Experts(num_experts, dim, inter_dim, dtype=dtype)
        self.post_feedforward_layernorm_2 = _SyntheticRMSNorm(dim, dtype=dtype)
        self.post_feedforward_layernorm = _SyntheticRMSNorm(dim, dtype=dtype)
        self.layer_scalar = nn.Parameter(torch.tensor(1.0, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, _ = self.self_attn(h, **kwargs)
        h = self.post_attention_layernorm(h)
        residual = residual + h

        h_ff = self.pre_feedforward_layernorm(residual)
        h_1 = self.post_feedforward_layernorm_1(self.mlp(h_ff))

        h_flat = residual.reshape(-1, residual.shape[-1])
        top_k_weights, top_k_index = self.router(h_flat)
        h_2_in = self.pre_feedforward_layernorm_2(h_flat)
        h_2 = self.experts(h_2_in, top_k_index, top_k_weights).reshape(residual.shape)
        h_2 = self.post_feedforward_layernorm_2(h_2)

        h_comb = self.post_feedforward_layernorm(h_1 + h_2)
        return (residual + h_comb) * self.layer_scalar


class _SyntheticHybridModel(nn.Module):
    def __init__(self, dim: int = 64, num_layers: int = 2, is_4bit: bool = False, dtype=torch.float32):
        super().__init__()
        self.embed = nn.Embedding(128, dim, dtype=dtype)
        layers = nn.ModuleList([_SyntheticDecoderLayer(dim=dim, is_4bit=is_4bit, dtype=dtype) for _ in range(num_layers)])
        lang_model = nn.Module()
        lang_model.layers = layers
        inner_model = nn.Module()
        inner_model.language_model = lang_model
        self.backbone = nn.Module()
        self.backbone.model = inner_model
        self.backbone.get_input_embeddings = lambda: self.embed
        self.hybridlink = HybridLink(dim).to(dtype=dtype)


class TestFeatureF1DynamicTensorRTHybridLink(unittest.TestCase):
    """Tests for Feature F1: _TensorRTHybridLinkWrapper, compile_hybridlink_tensorrt, enable_tensorrt_acceleration."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.hidden_size = 64
        self.base_link = HybridLink(self.hidden_size).eval()

    def test_wrapper_routes_all_batch_sizes_without_eager_fallback(self) -> None:
        """Verify B = 1, 2, 16, 32, 256 (2D and 3D) route directly through trt_engine_2d without calling base_link."""
        call_shapes: list[tuple[int, ...]] = []
        ref_link = HybridLink(self.hidden_size).eval()
        ref_link.load_state_dict(self.base_link.state_dict())

        def spy_trt_engine(x_2d: torch.Tensor) -> torch.Tensor:
            self.assertEqual(x_2d.dim(), 2)
            call_shapes.append(tuple(x_2d.shape))
            return ref_link(x_2d)

        # Make base_link.forward raise if ever called as fallback
        self.base_link.forward = mock.MagicMock(side_effect=AssertionError("Fallback to base_link must not happen!"))

        wrapper = _TensorRTHybridLinkWrapper(
            self.base_link,
            spy_trt_engine,
            hidden_size=self.hidden_size,
            max_batch_size=256,
            min_batch_size=1,
            opt_batch_size=32,
            backend_name="torch_tensorrt (test)",
        )

        for b in (1, 2, 16, 32, 256):
            call_shapes.clear()
            x = torch.randn(b, self.hidden_size)
            with torch.no_grad():
                out = wrapper(x)
                expected = ref_link(x)
            self.assertEqual(call_shapes, [(b, self.hidden_size)])
            self.assertEqual(out.shape, (b, self.hidden_size))
            self.assertTrue(torch.allclose(out, expected, atol=1e-6))

        # 3D inputs [B, S, H] within max_batch_size
        call_shapes.clear()
        x_3d = torch.randn(8, 16, self.hidden_size)  # 128 rows
        with torch.no_grad():
            out_3d = wrapper(x_3d)
            expected_3d = ref_link(x_3d)
        self.assertEqual(call_shapes, [(128, self.hidden_size)])
        self.assertEqual(out_3d.shape, (8, 16, self.hidden_size))
        self.assertTrue(torch.allclose(out_3d, expected_3d, atol=1e-6))
        self.base_link.forward.assert_not_called()

    def test_wrapper_chunks_large_batches_exceeding_max_batch_size(self) -> None:
        """Verify N > max_batch_size (e.g. B = 300, 600 and 3D [16, 20, H]) chunks into max_batch_size slices."""
        call_shapes: list[tuple[int, ...]] = []
        ref_link = HybridLink(self.hidden_size).eval()
        ref_link.load_state_dict(self.base_link.state_dict())

        def spy_trt_engine(x_2d: torch.Tensor) -> torch.Tensor:
            call_shapes.append(tuple(x_2d.shape))
            return ref_link(x_2d)

        self.base_link.forward = mock.MagicMock(side_effect=AssertionError("Fallback to base_link must not happen!"))

        wrapper = _TensorRTHybridLinkWrapper(
            self.base_link,
            spy_trt_engine,
            hidden_size=self.hidden_size,
            max_batch_size=256,
        )

        # B = 300 -> 256 + 44
        call_shapes.clear()
        x_300 = torch.randn(300, self.hidden_size)
        with torch.no_grad():
            out_300 = wrapper(x_300)
            exp_300 = ref_link(x_300)
        self.assertEqual(call_shapes, [(256, self.hidden_size), (44, self.hidden_size)])
        self.assertTrue(torch.allclose(out_300, exp_300, atol=1e-5))

        # B = 600 -> 256 + 256 + 88
        call_shapes.clear()
        x_600 = torch.randn(600, self.hidden_size)
        with torch.no_grad():
            out_600 = wrapper(x_600)
            exp_600 = ref_link(x_600)
        self.assertEqual(call_shapes, [(256, self.hidden_size), (256, self.hidden_size), (88, self.hidden_size)])
        self.assertTrue(torch.allclose(out_600, exp_600, atol=1e-5))

        # 3D [16, 20, H] = 320 rows -> 256 + 64
        call_shapes.clear()
        x_3d_large = torch.randn(16, 20, self.hidden_size)
        with torch.no_grad():
            out_3d_large = wrapper(x_3d_large)
            exp_3d_large = ref_link(x_3d_large)
        self.assertEqual(call_shapes, [(256, self.hidden_size), (64, self.hidden_size)])
        self.assertEqual(out_3d_large.shape, (16, 20, self.hidden_size))
        self.assertTrue(torch.allclose(out_3d_large, exp_3d_large, atol=1e-5))

    def test_wrapper_attribute_delegation_and_tuple_unpacking(self) -> None:
        """Verify __iter__ 2-tuple unpacking, __getattr__ delegation to base_link, and backend_name attributes."""
        wrapper = _TensorRTHybridLinkWrapper(
            self.base_link,
            self.base_link,
            hidden_size=self.hidden_size,
            max_batch_size=128,
            min_batch_size=1,
            opt_batch_size=16,
            backend_name="torch_tensorrt (2.11)",
        )
        mod, b_name = wrapper
        self.assertIs(mod, wrapper)
        self.assertEqual(b_name, "torch_tensorrt (2.11)")
        self.assertEqual(wrapper.backend_name, "torch_tensorrt (2.11)")
        self.assertEqual(wrapper._trt_backend_name, "torch_tensorrt (2.11)")
        self.assertIs(wrapper.proj1, self.base_link.proj1)
        self.assertIs(wrapper.proj2, self.base_link.proj2)
        self.assertIs(wrapper.pre_ln, self.base_link.pre_ln)
        self.assertIs(wrapper.post_ln, self.base_link.post_ln)

    def test_compile_hybridlink_tensorrt_dynamic_profile_and_fallbacks(self) -> None:
        """Verify compile_hybridlink_tensorrt passes min_shape/opt_shape/max_shape to torch_tensorrt.Input
        and falls back cleanly to torch.compile(..., dynamic=True)."""
        mock_trt = types.ModuleType("torch_tensorrt")
        mock_trt.__version__ = "2.11.0"
        mock_trt.Input = mock.MagicMock(return_value="MOCK_TRT_INPUT")
        mock_trt.compile = mock.MagicMock(side_effect=lambda mod, **kw: mod)

        with mock.patch.dict(sys.modules, {"torch_tensorrt": mock_trt}):
            res = compile_hybridlink_tensorrt(
                self.base_link,
                hidden_size=self.hidden_size,
                dtype=torch.float32,
                device="cpu",
                min_batch_size=1,
                opt_batch_size=32,
                max_batch_size=256,
            )
        self.assertIsInstance(res, _TensorRTHybridLinkWrapper)
        mock_trt.Input.assert_called_once_with(
            min_shape=(1, self.hidden_size),
            opt_shape=(32, self.hidden_size),
            max_shape=(256, self.hidden_size),
            dtype=torch.float32,
        )
        mock_trt.compile.assert_called_once()
        self.assertEqual(res.min_batch_size, 1)
        self.assertEqual(res.opt_batch_size, 32)
        self.assertEqual(res.max_batch_size, 256)
        self.assertIn("torch_tensorrt", res.backend_name)

        # Verify torch.compile(..., dynamic=True) fallback when torch_tensorrt fails
        with (
            mock.patch.dict(sys.modules, {"torch_tensorrt": None}),
            mock.patch("torch.compile", side_effect=lambda mod, **kw: mod) as mock_tc,
        ):
            res_fb, backend_fb = compile_hybridlink_tensorrt(
                self.base_link,
                hidden_size=self.hidden_size,
                dtype=torch.float32,
                device="cpu",
                min_batch_size=2,
                opt_batch_size=16,
                max_batch_size=128,
            )
        self.assertIsInstance(res_fb, _TensorRTHybridLinkWrapper)
        self.assertEqual(res_fb.max_batch_size, 128)
        self.assertEqual(res_fb.opt_batch_size, 16)
        self.assertEqual(res_fb.min_batch_size, 2)
        self.assertIn("dynamic=True", backend_fb)
        mock_tc.assert_called_with(
            self.base_link,
            backend="tensorrt",
            mode="reduce-overhead",
            dynamic=True,
        )

    def test_enable_tensorrt_acceleration_bf16_and_4bit_models(self) -> None:
        """Verify enable_tensorrt_acceleration populates _tensorrt_config and _acceleration_info
        and patches BF16 and 4-bit layers."""
        bf16_model = _SyntheticHybridModel(dim=64, num_layers=2, is_4bit=False)
        info_bf16 = enable_tensorrt_acceleration(
            bf16_model,
            compile_backbone_layers=True,
            use_fp8_or_bf16="bf16",
            max_batch_size=128,
            opt_batch_size=16,
            min_batch_size=1,
        )
        self.assertEqual(info_bf16["bf16_experts_zero_sync_layers"], 2)
        self.assertEqual(info_bf16["bf16_ffn_compiled_layers"], 2)
        self.assertEqual(info_bf16["int4_ffn_compiled_layers"], 0)
        self.assertTrue(info_bf16["experts_triton_compiled"])
        self.assertEqual(info_bf16["max_batch_size"], 128)
        self.assertEqual(info_bf16["opt_batch_size"], 16)
        self.assertEqual(bf16_model._tensorrt_config["max_batch_size"], 128)
        self.assertEqual(bf16_model._tensorrt_config["opt_batch_size"], 16)
        self.assertEqual(bf16_model._tensorrt_config["use_fp8_or_bf16"], "bf16")
        self.assertIsInstance(bf16_model.hybridlink, _TensorRTHybridLinkWrapper)
        self.assertEqual(bf16_model.hybridlink.max_batch_size, 128)

        # Also test 4-bit model
        int4_model = _SyntheticHybridModel(dim=64, num_layers=2, is_4bit=True)
        info_int4 = enable_tensorrt_acceleration(
            int4_model,
            compile_backbone_layers=True,
            use_fp8_or_bf16="fp8",
            max_batch_size=256,
            opt_batch_size=32,
        )
        self.assertEqual(info_int4["bf16_experts_zero_sync_layers"], 0)
        self.assertEqual(info_int4["bf16_ffn_compiled_layers"], 0)
        self.assertEqual(info_int4["int4_ffn_compiled_layers"], 2)
        self.assertEqual(info_int4["use_fp8_or_bf16"], "fp8")


class TestFeatureF2ZeroSyncBF16MoEAndCompiledFFN(unittest.TestCase):
    """Tests for Feature F2: Zero-Sync BF16 MoE & Compiled BF16 FFN."""

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.num_experts = 16
        self.hidden_dim = 64
        self.inter_dim = 32
        self.top_k = 4
        self.experts = _SyntheticGemma4BF16Experts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.inter_dim,
            dtype=torch.float32,
        )

    def test_bf16_experts_fast_forward_zero_nonzero_and_exact_parity(self) -> None:
        """Verify _bf16_experts_fast_forward across n_tokens = 1, 2, 8, 9, 16, 32, 64, 128, 256
        matches reference per-expert MoE with ZERO calls to torch.Tensor.nonzero."""
        backbone = nn.Module()
        backbone.experts = self.experts
        patched = patch_bf16_experts_fastpath(backbone)
        self.assertEqual(patched, 1)

        def forbidden_nonzero(*args, **kwargs):
            raise RuntimeError("CPU-GPU sync via .nonzero() is strictly forbidden!")

        with mock.patch.object(torch.Tensor, "nonzero", forbidden_nonzero):
            for n_tokens in (1, 2, 8, 9, 16, 32, 64, 128, 256):
                h = torch.randn(n_tokens, self.hidden_dim, dtype=torch.float32)
                logits = torch.randn(n_tokens, self.num_experts)
                probs = F.softmax(logits, dim=-1)
                top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
                top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

                with torch.no_grad():
                    out_fast = self.experts(h, top_k_index, top_k_weights)
                    out_ref = _reference_bf16_moe(
                        h, top_k_index, top_k_weights, self.experts.gate_up_proj, self.experts.down_proj
                    )
                self.assertEqual(out_fast.shape, (n_tokens, self.hidden_dim))
                self.assertTrue(
                    torch.allclose(out_fast, out_ref, atol=1e-5, rtol=1e-4),
                    f"Mismatch at n_tokens={n_tokens}: max diff={(out_fast - out_ref).abs().max().item()}",
                )

            # Test 3D input [B, S, H]
            h_3d = torch.randn(6, 5, self.hidden_dim, dtype=torch.float32)  # 30 tokens (> 8 and > 16)
            logits_3d = torch.randn(30, self.num_experts)
            top_k_w_3d, top_k_idx_3d = torch.topk(F.softmax(logits_3d, dim=-1), k=self.top_k, dim=-1)
            with torch.no_grad():
                out_3d = self.experts(h_3d, top_k_idx_3d, top_k_w_3d)
                ref_3d = _reference_bf16_moe(
                    h_3d, top_k_idx_3d, top_k_w_3d, self.experts.gate_up_proj, self.experts.down_proj
                )
            self.assertEqual(out_3d.shape, (6, 5, self.hidden_dim))
            self.assertTrue(torch.allclose(out_3d, ref_3d, atol=1e-5, rtol=1e-4))

    def test_fused_triton_bf16_moe_forward_kernel_dispatch_and_sorting(self) -> None:
        """Verify fused_triton_bf16_moe_forward dispatches USE_ORDER=False for T <= 8
        and USE_ORDER=True with sorted expert order for T > 8."""
        launched_calls: list[dict] = []

        class _MockGridLauncher:
            def __init__(self, grid):
                self.grid = grid

            def __call__(self, x_ptr, idx_ptr, order_ptr, w_ptr, out_ptr, **kwargs):
                launched_calls.append({"grid": self.grid, "order": order_ptr.clone(), "idx": idx_ptr.clone(), **kwargs})
                out_ptr.zero_()

        mock_kernel = mock.MagicMock()
        mock_kernel.__getitem__.side_effect = lambda grid: _MockGridLauncher(grid)

        with mock.patch.object(backends, "_fused_bf16_moe_kernel", mock_kernel):
            # Small T = 4 (<= 8): USE_ORDER must be False
            launched_calls.clear()
            h_small = torch.randn(4, self.hidden_dim)
            idx_small = torch.randint(0, self.num_experts, (4, self.top_k))
            w_small = torch.full((4, self.top_k), 0.25)
            out_s = fused_triton_bf16_moe_forward(
                h_small, idx_small, w_small, self.experts.gate_up_proj, self.experts.down_proj
            )
            self.assertEqual(out_s.shape, (4, self.hidden_dim))
            self.assertEqual(len(launched_calls), 2)
            self.assertFalse(launched_calls[0]["USE_ORDER"])
            self.assertFalse(launched_calls[0]["X_PER_EXPERT"])
            self.assertFalse(launched_calls[1]["USE_ORDER"])
            self.assertTrue(launched_calls[1]["X_PER_EXPERT"])

            # High-concurrency T = 16 (> 8): USE_ORDER must be True and expert IDs sorted monotonically
            launched_calls.clear()
            h_large = torch.randn(16, self.hidden_dim)
            idx_large = torch.randint(0, self.num_experts, (16, self.top_k))
            w_large = torch.full((16, self.top_k), 0.25)
            out_l = fused_triton_bf16_moe_forward(
                h_large, idx_large, w_large, self.experts.gate_up_proj, self.experts.down_proj
            )
            self.assertEqual(out_l.shape, (16, self.hidden_dim))
            self.assertEqual(len(launched_calls), 2)
            self.assertTrue(launched_calls[0]["USE_ORDER"])
            self.assertTrue(launched_calls[1]["USE_ORDER"])
            order = launched_calls[0]["order"]
            flat_idx = launched_calls[0]["idx"]
            sorted_experts = flat_idx[order]
            self.assertTrue(torch.all(sorted_experts[1:] >= sorted_experts[:-1]))

    def test_patch_compiled_ffn_bf16_layer_parity(self) -> None:
        """Verify patch_compiled_ffn_bf16 fuses all BF16 layers and matches unpatched layer output for B = 1, 12, 32."""
        torch.manual_seed(999)
        model_ref = _SyntheticHybridModel(dim=self.hidden_dim, num_layers=2, is_4bit=False)
        model_patched = _SyntheticHybridModel(dim=self.hidden_dim, num_layers=2, is_4bit=False)
        model_patched.load_state_dict(model_ref.state_dict())

        num_patched = patch_compiled_ffn_bf16(model_patched.backbone)
        self.assertEqual(num_patched, 2)

        for b in (1, 12, 32):
            x = torch.randn(b, 3, self.hidden_dim)
            with torch.no_grad():
                h_ref = x
                h_pat = x
                for lyr_ref, lyr_pat in zip(
                    model_ref.backbone.model.language_model.layers,
                    model_patched.backbone.model.language_model.layers,
                ):
                    h_ref = lyr_ref(h_ref)
                    h_pat = lyr_pat(h_pat)
            self.assertTrue(
                torch.allclose(h_ref, h_pat, atol=1e-5, rtol=1e-4),
                f"BF16 compiled FFN mismatch for B={b}: max diff={(h_ref - h_pat).abs().max().item()}",
            )


class TestFeatureF3Part1W4A16MoEOptimization(unittest.TestCase):
    """Tests for Feature F3 Part 1: Expert-sorted W4A16 MoE kernel and zero-sync PyTorch fallback."""

    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.num_experts = 16
        self.hidden_dim = 64
        self.inter_dim = 32
        self.top_k = 4
        cfg = types.SimpleNamespace(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            moe_intermediate_size=self.inter_dim,
            hidden_activation="gelu_pytorch_tanh",
        )
        self.experts_4bit = Gemma4TextExperts4Bit(cfg, dtype=torch.bfloat16)
        self.experts_4bit.gate_up_packed.random_(0, 2**31 - 1)
        self.experts_4bit.gate_up_scale.uniform_(-0.05, 0.05)
        self.experts_4bit.down_packed.random_(0, 2**31 - 1)
        self.experts_4bit.down_scale.uniform_(-0.05, 0.05)

    def test_sort_moe_token_expert_pairs_monotonicity_and_invariants(self) -> None:
        """Verify sort_moe_token_expert_pairs produces monotonically non-decreasing expert_ids and valid permutations."""
        for T in (1, 8, 9, 16, 64):
            top_k_index = torch.randint(0, self.num_experts, (T, 8))
            sorted_expert_ids, sort_perm, sorted_token_ids = sort_moe_token_expert_pairs(top_k_index)
            self.assertEqual(sorted_expert_ids.shape, (T * 8,))
            self.assertEqual(sort_perm.shape, (T * 8,))
            self.assertEqual(sorted_token_ids.shape, (T * 8,))
            self.assertTrue(torch.all(sorted_expert_ids[1:] >= sorted_expert_ids[:-1]))
            flat = top_k_index.view(-1)
            self.assertTrue(torch.equal(flat[sort_perm], sorted_expert_ids))
            self.assertTrue(torch.equal(sorted_token_ids, sort_perm // 8))

    def test_dequantize_int4_packed_optional_shifts(self) -> None:
        """Verify dequantize_int4_packed works both with shifts=None and explicit shifts."""
        packed = self.experts_4bit.gate_up_packed[0]
        scale = self.experts_4bit.gate_up_scale[0]
        w_explicit = dequantize_int4_packed(packed, scale, self.experts_4bit._shifts)
        w_implicit = dequantize_int4_packed(packed, scale)
        self.assertTrue(torch.equal(w_explicit, w_implicit))
        self.assertEqual(w_implicit.shape, (2 * self.inter_dim, self.hidden_dim))

    def test_4bit_moe_zero_nonzero_and_sorted_vs_unsorted_equivalence(self) -> None:
        """Verify _pytorch_zero_sync_4bit_moe_forward and Gemma4TextExperts4Bit.forward for T = 1, 4, 8, 9, 16, 64
        match reference per-expert MoE and make ZERO calls to torch.Tensor.nonzero."""
        def forbidden_nonzero(*args, **kwargs):
            raise RuntimeError("CPU-GPU sync via .nonzero() is strictly forbidden!")

        with mock.patch.object(torch.Tensor, "nonzero", forbidden_nonzero):
            for T in (1, 4, 8, 9, 16, 64):
                h = torch.randn(T, self.hidden_dim, dtype=torch.bfloat16)
                logits = torch.randn(T, self.num_experts)
                top_k_w, top_k_idx = torch.topk(F.softmax(logits, dim=-1), k=self.top_k, dim=-1)
                top_k_w = (top_k_w / top_k_w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

                with torch.no_grad():
                    out_mod = self.experts_4bit(h, top_k_idx, top_k_w)
                    out_unsorted = _pytorch_zero_sync_4bit_moe_forward(
                        h,
                        top_k_idx,
                        top_k_w,
                        self.experts_4bit.gate_up_packed,
                        self.experts_4bit.gate_up_scale,
                        self.experts_4bit.down_packed,
                        self.experts_4bit.down_scale,
                        sort_threshold=1000,
                    )
                    out_sorted = _pytorch_zero_sync_4bit_moe_forward(
                        h,
                        top_k_idx,
                        top_k_w,
                        self.experts_4bit.gate_up_packed,
                        self.experts_4bit.gate_up_scale,
                        self.experts_4bit.down_packed,
                        self.experts_4bit.down_scale,
                        sort_threshold=0,
                        chunk_tk=11,
                    )
                    out_ref = _reference_4bit_moe(
                        h,
                        top_k_idx,
                        top_k_w,
                        self.experts_4bit.gate_up_packed,
                        self.experts_4bit.gate_up_scale,
                        self.experts_4bit.down_packed,
                        self.experts_4bit.down_scale,
                    )

                self.assertEqual(out_mod.shape, (T, self.hidden_dim))
                self.assertTrue(torch.allclose(out_unsorted, out_sorted, atol=1e-2, rtol=1e-2))
                self.assertTrue(torch.allclose(out_mod, out_ref, atol=2e-2, rtol=2e-2))

    def test_fused_triton_4bit_moe_forward_mocked_kernel_launch(self) -> None:
        """Verify fused_triton_4bit_moe_forward launches _fused_w4a16_moe_kernel with SORTED=False
        for T <= 8 and SORTED=True with sorted expert indices for T > 8."""
        launched: list[dict] = []

        class _MockW4A16Launcher:
            def __init__(self, grid):
                self.grid = grid

            def __call__(self, x_ptr, idx_ptr, w_packed, w_scale, out_ptr, **kwargs):
                launched.append({"grid": self.grid, "idx": idx_ptr.clone(), **kwargs})
                out_ptr.zero_()

        mock_kernel = mock.MagicMock()
        mock_kernel.__getitem__.side_effect = lambda grid: _MockW4A16Launcher(grid)

        with mock.patch.object(quantization, "_fused_w4a16_moe_kernel", mock_kernel):
            for T, expected_sorted in ((1, False), (8, False), (9, True), (16, True)):
                launched.clear()
                h = torch.randn(T, self.hidden_dim, dtype=torch.bfloat16)
                idx = torch.randint(0, self.num_experts, (T, self.top_k))
                w = torch.full((T, self.top_k), 0.25, dtype=torch.bfloat16)
                out = fused_triton_4bit_moe_forward(
                    h,
                    idx,
                    w,
                    self.experts_4bit.gate_up_packed,
                    self.experts_4bit.gate_up_scale,
                    self.experts_4bit.down_packed,
                    self.experts_4bit.down_scale,
                )
                self.assertEqual(out.shape, (T, self.hidden_dim))
                self.assertEqual(len(launched), 2)
                self.assertEqual(launched[0]["SORTED"], expected_sorted)
                self.assertFalse(launched[0]["X_PER_EXPERT"])
                self.assertEqual(launched[1]["SORTED"], expected_sorted)
                self.assertTrue(launched[1]["X_PER_EXPERT"])
                if expected_sorted:
                    exec_idx = launched[0]["idx"]
                    self.assertTrue(torch.all(exec_idx[1:] >= exec_idx[:-1]))
                    perm = launched[0]["perm_ptr"]
                    self.assertTrue(torch.equal(idx.view(-1)[perm], exec_idx))


class TestFeatureF3Part2SGLangHybridEngineConcurrency(unittest.TestCase):
    """Tests for Feature F3 Part 2: SGLangHybridEngine non-mutating latent_mode and concurrent run_batch."""

    def _make_stub_sglang_engine(self) -> SGLangHybridEngine:
        engine = SGLangHybridEngine.__new__(SGLangHybridEngine)
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
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
        engine.hybridlink = HybridLink(64).eval()
        engine.tools = types.SimpleNamespace(
            execute=mock.MagicMock(side_effect=lambda tool, q: f"{tool}_res({q.strip()})")
        )
        return engine

    def test_sequential_runs_do_not_mutate_engine_latent_mode(self) -> None:
        """Verify calling engine.run('Q1') followed by engine.run('Q2') preserves engine.latent_mode == 'hybrid'
        and executes <latent> steps on BOTH requests."""
        engine = self._make_stub_sglang_engine()
        call_counts: dict[str, int] = {}

        def fake_sgl_generate(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
            q_tag = "Q1" if "Q1" in prompt else "Q2"
            step = call_counts.get(q_tag, 0)
            call_counts[q_tag] = step + 1
            if step == 0:
                self.assertTrue(return_hidden_states)
                self.assertIn("stop_token_ids", sampling_params)
                return {
                    "text": "Thinking...<latent>",
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": 262144},
                        "hidden_states": [torch.randn(1, 64).tolist()],
                    },
                }
            return {
                "text": f"<answer>\\boxed{{{q_tag}_ans}}</answer>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
            }

        engine._sgl_generate = fake_sgl_generate

        r1 = engine.run("Solve Q1")
        self.assertEqual(engine.latent_mode, "hybrid")
        self.assertEqual(r1.latent_steps, 3)
        self.assertEqual(r1.latent_spans, 1)
        self.assertEqual(r1.final_answer, "Q1_ans")
        self.assertEqual(r1["final_answer"], "Q1_ans")

        r2 = engine.run("Solve Q2")
        self.assertEqual(engine.latent_mode, "hybrid")
        self.assertEqual(r2.latent_steps, 3)
        self.assertEqual(r2.latent_spans, 1)
        self.assertEqual(r2.final_answer, "Q2_ans")
        self.assertEqual(r2["final_answer"], "Q2_ans")

    def test_python_and_search_tools_reenable_latent_in_next_round(self) -> None:
        """Verify <python> and <search> tool calls execute and re-enable <latent> in subsequent rounds."""
        engine = self._make_stub_sglang_engine()
        step_idx = 0

        def fake_sgl_generate(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
            nonlocal step_idx
            step_idx += 1
            if step_idx == 1:
                return {
                    "text": "<latent>",
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": 262144},
                        "hidden_states": [torch.randn(1, 64).tolist()],
                    },
                }
            if step_idx == 2:
                return {
                    "text": "<python>print(6 * 7)</python>",
                    "meta_info": {"finish_reason": {"type": "stop", "matched": "</python>"}},
                }
            if step_idx == 3:
                # After <python> tool round, <latent> should be re-enabled!
                self.assertTrue(return_hidden_states)
                return {
                    "text": "<latent>",
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": 262144},
                        "hidden_states": [torch.randn(1, 64).tolist()],
                    },
                }
            if step_idx == 4:
                return {
                    "text": "<search>gemma 4 architecture</search>",
                    "meta_info": {"finish_reason": {"type": "stop", "matched": "</search>"}},
                }
            return {
                "text": "<answer>\\boxed{42}</answer>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
            }

        engine._sgl_generate = fake_sgl_generate
        res = engine.run("Compute 6*7 and search")
        self.assertEqual(res.final_answer, "42")
        self.assertEqual(res.latent_steps, 6)
        self.assertEqual(res.latent_spans, 2)
        self.assertEqual(len(res.rounds), 3)
        self.assertIn("<result>python_res(print(6 * 7))</result>", res.output)
        self.assertIn("<result>search_res(gemma 4 architecture)</result>", res.output)

    def test_run_batch_concurrent_execution_preserves_order_and_thread_safety(self) -> None:
        """Verify SGLangHybridEngine.run_batch across 20 concurrent questions with random jitter
        preserves input order, executes <latent> steps safely, and returns dual HybridTrajectory/dict objects."""
        engine = self._make_stub_sglang_engine()
        lock = threading.Lock()
        per_q_step: dict[int, int] = {}

        def fake_sgl_generate(*, prompt: str, sampling_params: dict, return_hidden_states: bool = False, **kw):
            # Extract question index from prompt "Q_<idx>_"
            q_idx = int(prompt.split("Q_")[1].split("_")[0])
            with lock:
                s = per_q_step.get(q_idx, 0)
                per_q_step[q_idx] = s + 1
            time.sleep(random.uniform(0.001, 0.010))
            if s == 0:
                return {
                    "text": "<latent>",
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": 262144},
                        "hidden_states": [torch.randn(1, 64).tolist()],
                    },
                }
            return {
                "text": f"<answer>\\boxed{{{q_idx * 10}}}</answer>",
                "meta_info": {"finish_reason": {"type": "stop", "matched": "</answer>"}},
            }

        engine._sgl_generate = fake_sgl_generate
        questions = [f"Q_{i}_end" for i in range(20)]
        results = engine.run_batch(questions, max_concurrency=8)

        self.assertEqual(len(results), 20)
        self.assertEqual(engine.latent_mode, "hybrid")
        for i, r in enumerate(results):
            self.assertIsInstance(r, HybridTrajectory)
            self.assertIsInstance(r, dict)
            self.assertEqual(r.final_answer, str(i * 10))
            self.assertEqual(r["final_answer"], str(i * 10))
            self.assertEqual(r["answer"], str(i * 10))
            self.assertEqual(r.latent_steps, 3)
            self.assertEqual(r["latent_steps"], 3)

        # Also test empty batch
        self.assertEqual(engine.run_batch([]), [])


if __name__ == "__main__":
    unittest.main()
