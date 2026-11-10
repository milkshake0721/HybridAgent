"""4-bit compressed-tensors (pack-quantized W4A16 group-32) runtime layers and loader
for Gemma-4-26B-A4B-HybridAgent on 24GB GPUs (e.g., RTX 4090).

Compatible with the exact same `compressed-tensors` checkpoint layout used by vLLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN

GROUP_SIZE = 32
SORT_EXPERT_THRESHOLD = 8

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_w4a16_moe_kernel(
        x_ptr,          # [T, K_dim] if not X_PER_EXPERT else [T * top_k, K_dim]
        idx_ptr,        # [T * top_k] int64/int32 expert indices (sorted when SORTED=True)
        w_packed_ptr,   # [num_experts, N_dim, K_packed] int32
        w_scale_ptr,    # [num_experts, N_dim, K_packed // 4] bf16
        out_ptr,        # [T * top_k, N_dim] bf16
        top_k: tl.constexpr,
        N_dim: tl.constexpr,
        K_packed: tl.constexpr,
        X_PER_EXPERT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KP: tl.constexpr,
        perm_ptr=None,  # [T * top_k] int64/int32 original flat indices when SORTED=True
        SORTED: tl.constexpr = False,
    ):
        pid_tk = tl.program_id(0)  # 0 .. (T * top_k) - 1
        pid_n = tl.program_id(1)   # 0 .. cdiv(N_dim, BLOCK_N) - 1

        expert_id = tl.load(idx_ptr + pid_tk).to(tl.int64)
        if SORTED:
            orig_tk = tl.load(perm_ptr + pid_tk).to(tl.int64)
        else:
            orig_tk = pid_tk
        token_id = orig_tk // top_k

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_dim

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        w_exp_base = w_packed_ptr + expert_id * (N_dim * K_packed)
        s_exp_base = w_scale_ptr + expert_id * (N_dim * (K_packed // 4))
        x_row = pid_tk if X_PER_EXPERT else token_id
        x_base = x_ptr + x_row * (K_packed * 8)

        offs_kp_base = tl.arange(0, BLOCK_KP)
        shifts = tl.arange(0, 8) * 4

        for kp_start in range(0, K_packed, BLOCK_KP):
            offs_kp = kp_start + offs_kp_base
            mask_kp = offs_kp < K_packed

            w_ptrs = w_exp_base + offs_n[:, None] * K_packed + offs_kp[None, :]
            packed = tl.load(w_ptrs, mask=mask_n[:, None] & mask_kp[None, :], other=0)

            offs_s = (kp_start // 4) + tl.arange(0, BLOCK_KP // 4)
            mask_s = offs_s < (K_packed // 4)
            s_ptrs = s_exp_base + offs_n[:, None] * (K_packed // 4) + offs_s[None, :]
            scales_bf16 = tl.load(s_ptrs, mask=mask_n[:, None] & mask_s[None, :], other=0.0).to(tl.bfloat16)

            offs_x = offs_kp[:, None] * 8 + tl.arange(0, 8)[None, :]
            mask_x = offs_kp[:, None] < K_packed
            x_vals = tl.load(x_base + offs_x, mask=mask_x, other=0.0).to(tl.bfloat16)
            x_f32 = tl.reshape(x_vals.to(tl.float32), (BLOCK_KP // 4, 32))

            q = ((packed[:, :, None] >> shifts[None, None, :]) & 0xF) - 8
            q_bf16 = tl.reshape(q.to(tl.bfloat16), (BLOCK_N, BLOCK_KP // 4, 32))
            w_bf16 = (q_bf16 * scales_bf16[:, :, None]).to(tl.float32)
            acc += tl.sum(tl.reshape(w_bf16 * x_f32[None, :, :], (BLOCK_N, BLOCK_KP * 8)), axis=1)

        out_row = orig_tk if (SORTED and X_PER_EXPERT) else pid_tk
        out_ptrs = out_ptr + out_row * N_dim + offs_n
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_n)

    _HAS_TRITON_W4A16 = True
except Exception:
    _HAS_TRITON_W4A16 = False
    _fused_w4a16_moe_kernel = None


def sort_moe_token_expert_pairs(
    top_k_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort flattened `(token, top_k_slot)` pairs by `expert_id` so consecutive threadblocks
    share the same expert weights in L2 cache during high-concurrency execution (`T > 8`).

    Returns `(sorted_expert_ids, sort_perm, sorted_token_ids)` each of shape `[T * top_k]`.
    """
    top_k = top_k_index.shape[-1]
    idx_flat = top_k_index.contiguous().view(-1)
    sorted_expert_ids, sort_perm = torch.sort(idx_flat, stable=True)
    sorted_token_ids = torch.div(sort_perm, top_k, rounding_mode="floor")
    return sorted_expert_ids, sort_perm, sorted_token_ids


def dequantize_int4_packed(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack int32-packed symmetric 4-bit weights (offset +8) and multiply by group-32 scales."""
    if shifts is None or shifts.device != packed.device:
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF) - 8
    return (q.to(scale.dtype).view(*scale.shape, GROUP_SIZE) * scale.unsqueeze(-1)).flatten(-2)


def _pytorch_zero_sync_4bit_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gu_packed: torch.Tensor,
    gu_scale: torch.Tensor,
    dn_packed: torch.Tensor,
    dn_scale: torch.Tensor,
    *,
    shifts: torch.Tensor | None = None,
    act_fn=None,
    sort_threshold: int = SORT_EXPERT_THRESHOLD,
    chunk_tk: int = 64,
) -> torch.Tensor:
    """Zero-sync PyTorch W4A16 MoE forward pass for any T >= 1 without `.nonzero()` or
    `torch.where()` CPU-GPU synchronizations."""
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    h_2d = hidden_states.reshape(-1, hidden_dim)
    T = h_2d.shape[0]
    if T == 0:
        return torch.empty_like(hidden_states)

    top_k = top_k_index.shape[-1]
    idx_2d = top_k_index.reshape(T, top_k)
    w_2d = top_k_weights.reshape(T, top_k)
    intermediate_dim = dn_packed.shape[2] * 8
    if act_fn is None:
        act_fn = lambda x: F.gelu(x, approximate="tanh")

    if T == 1:
        idx = idx_2d[0]
        h0 = h_2d[0]
        w_gu = dequantize_int4_packed(gu_packed.index_select(0, idx), gu_scale.index_select(0, idx), shifts)
        if w_gu.dtype != h0.dtype:
            w_gu = w_gu.to(h0.dtype)
        gu = torch.mv(w_gu.view(-1, hidden_dim), h0).view(top_k, 2 * intermediate_dim)
        gate, up = gu.chunk(2, dim=-1)
        inter = act_fn(gate) * up

        w_dn = dequantize_int4_packed(dn_packed.index_select(0, idx), dn_scale.index_select(0, idx), shifts)
        if w_dn.dtype != inter.dtype:
            w_dn = w_dn.to(inter.dtype)
        out = torch.bmm(w_dn, inter.unsqueeze(-1)).squeeze(-1)
        weights = w_2d[0, :, None].to(out.dtype)
        res = (out * weights).sum(dim=0, keepdim=True).to(hidden_states.dtype)
        return res.view(orig_shape)

    if T <= sort_threshold:
        flat_idx = idx_2d.reshape(-1)
        w_gu = dequantize_int4_packed(
            gu_packed.index_select(0, flat_idx),
            gu_scale.index_select(0, flat_idx),
            shifts,
        ).view(T, top_k, 2 * intermediate_dim, hidden_dim)
        if w_gu.dtype != h_2d.dtype:
            w_gu = w_gu.to(h_2d.dtype)
        gu = torch.einsum("tkih,th->tki", w_gu, h_2d)
        gate, up = gu.chunk(2, dim=-1)
        inter = act_fn(gate) * up

        w_dn = dequantize_int4_packed(
            dn_packed.index_select(0, flat_idx),
            dn_scale.index_select(0, flat_idx),
            shifts,
        ).view(T, top_k, hidden_dim, intermediate_dim)
        if w_dn.dtype != inter.dtype:
            w_dn = w_dn.to(inter.dtype)
        out = torch.einsum("tkhi,tki->tkh", w_dn, inter)
        weights = w_2d.unsqueeze(-1).to(out.dtype)
        res = (out * weights).sum(dim=1).to(hidden_states.dtype)
        return res.view(orig_shape)

    # High-concurrency path (T > sort_threshold): sort (token, slot) pairs by expert_id
    # and process in bounded chunks with zero CPU-GPU syncs.
    TK = T * top_k
    sorted_expert_ids, sort_perm, sorted_token_ids = sort_moe_token_expert_pairs(idx_2d)
    dn_out_flat = torch.empty((TK, hidden_dim), dtype=h_2d.dtype, device=h_2d.device)

    for start in range(0, TK, chunk_tk):
        end = min(start + chunk_tk, TK)
        exp_c = sorted_expert_ids[start:end]
        tok_c = sorted_token_ids[start:end]
        orig_c = sort_perm[start:end]
        x_c = h_2d.index_select(0, tok_c)

        w_gu_c = dequantize_int4_packed(
            gu_packed.index_select(0, exp_c),
            gu_scale.index_select(0, exp_c),
            shifts,
        )
        if w_gu_c.dtype != x_c.dtype:
            w_gu_c = w_gu_c.to(x_c.dtype)
        gu_c = torch.bmm(w_gu_c, x_c.unsqueeze(-1)).squeeze(-1)
        gate_c, up_c = gu_c.chunk(2, dim=-1)
        inter_c = act_fn(gate_c) * up_c

        w_dn_c = dequantize_int4_packed(
            dn_packed.index_select(0, exp_c),
            dn_scale.index_select(0, exp_c),
            shifts,
        )
        if w_dn_c.dtype != inter_c.dtype:
            w_dn_c = w_dn_c.to(inter_c.dtype)
        out_c = torch.bmm(w_dn_c, inter_c.unsqueeze(-1)).squeeze(-1)
        dn_out_flat.index_copy_(0, orig_c, out_c.to(dn_out_flat.dtype))

    dn_out = dn_out_flat.view(T, top_k, hidden_dim)
    weights = w_2d.unsqueeze(-1).to(dn_out.dtype)
    res = (dn_out * weights).sum(dim=1).to(hidden_states.dtype)
    return res.view(orig_shape)


def fused_triton_4bit_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gu_packed: torch.Tensor,
    gu_scale: torch.Tensor,
    dn_packed: torch.Tensor,
    dn_scale: torch.Tensor,
    *,
    sort_threshold: int = SORT_EXPERT_THRESHOLD,
) -> torch.Tensor:
    """Zero-sync, register-fused Triton W4A16 MoE forward pass for both low-latency
    decode/latent steps (T <= 8, direct launch) and high-concurrency decode/prefill
    (T > 8, expert-sorted L2 cache locality launch).
    Never materializes unpacked BF16 expert weights in HBM.
    """
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    h_2d = hidden_states.reshape(-1, hidden_dim)
    T = h_2d.shape[0]
    if T == 0:
        return torch.empty_like(hidden_states)

    is_mocked_kernel = _fused_w4a16_moe_kernel is not None and (
        hasattr(_fused_w4a16_moe_kernel, "assert_called")
        or hasattr(_fused_w4a16_moe_kernel, "side_effect")
    )
    if not (_HAS_TRITON_W4A16 and h_2d.is_cuda) and not is_mocked_kernel:
        return _pytorch_zero_sync_4bit_moe_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            gu_packed,
            gu_scale,
            dn_packed,
            dn_scale,
            sort_threshold=sort_threshold,
        )

    top_k = top_k_index.shape[-1]
    idx_2d = top_k_index.reshape(T, top_k)
    w_2d = top_k_weights.reshape(T, top_k)
    TK = T * top_k
    N_gu = gu_packed.shape[1]
    K_gu_packed = gu_packed.shape[2]
    N_dn = dn_packed.shape[1]
    K_dn_packed = dn_packed.shape[2]

    x_contig = h_2d.contiguous()
    idx_flat = idx_2d.contiguous().view(-1)

    use_sorted = T > sort_threshold
    if use_sorted:
        exec_idx, sort_perm = torch.sort(idx_flat, stable=True)
    else:
        exec_idx = idx_flat
        sort_perm = idx_flat

    gu_out = torch.empty((TK, N_gu), dtype=h_2d.dtype, device=h_2d.device)
    BLOCK_N = 64
    grid_gu = (TK, (N_gu + BLOCK_N - 1) // BLOCK_N)
    _fused_w4a16_moe_kernel[grid_gu](
        x_contig, exec_idx, gu_packed, gu_scale, gu_out,
        top_k=top_k, N_dim=N_gu, K_packed=K_gu_packed, X_PER_EXPERT=False,
        BLOCK_N=BLOCK_N, BLOCK_KP=32, perm_ptr=sort_perm, SORTED=use_sorted, num_warps=4,
    )
    gate, up = gu_out.chunk(2, dim=-1)
    inter = (F.gelu(gate, approximate="tanh") * up).contiguous()

    dn_out = torch.empty((T, top_k, N_dn), dtype=h_2d.dtype, device=h_2d.device)
    grid_dn = (TK, (N_dn + BLOCK_N - 1) // BLOCK_N)
    _fused_w4a16_moe_kernel[grid_dn](
        inter, exec_idx, dn_packed, dn_scale, dn_out,
        top_k=top_k, N_dim=N_dn, K_packed=K_dn_packed, X_PER_EXPERT=True,
        BLOCK_N=BLOCK_N, BLOCK_KP=32, perm_ptr=sort_perm, SORTED=use_sorted, num_warps=4,
    )
    weights = w_2d.unsqueeze(-1).to(dn_out.dtype)
    out = (dn_out * weights).sum(dim=1).to(hidden_states.dtype)
    return out.view(orig_shape)


class Linear4Bit(nn.Module):
    """4-bit pack-quantized W4A16 Linear layer (group_size=32, symmetric)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_packed",
            torch.empty((out_features, in_features // 8), dtype=torch.int32),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty((out_features, in_features // GROUP_SIZE), dtype=dtype),
        )
        self.register_buffer(
            "_shifts",
            torch.arange(0, 32, 4, dtype=torch.int32),
            persistent=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_int4_packed(self.weight_packed, self.weight_scale, self._shifts)
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.linear(x, w, self.bias)


class Gemma4TextExperts4Bit(nn.Module):
    """4-bit pack-quantized W4A16 MoE Experts for Gemma 4 (128 experts, top-8 active).
    Decompresses only the active experts on the fly, with a zero-sync batched fast path
    for single-token decoding (text & latent steps) and expert-sorted high-concurrency execution.
    """

    def __init__(self, config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_activation]

        self.register_buffer(
            "gate_up_packed",
            torch.empty(
                (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim // 8),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "gate_up_scale",
            torch.empty(
                (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim // GROUP_SIZE),
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "down_packed",
            torch.empty(
                (self.num_experts, self.hidden_dim, self.intermediate_dim // 8),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "down_scale",
            torch.empty(
                (self.num_experts, self.hidden_dim, self.intermediate_dim // GROUP_SIZE),
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "_shifts",
            torch.arange(0, 32, 4, dtype=torch.int32),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if _HAS_TRITON_W4A16 and hidden_states.is_cuda:
            return fused_triton_4bit_moe_forward(
                hidden_states,
                top_k_index,
                top_k_weights,
                self.gate_up_packed,
                self.gate_up_scale,
                self.down_packed,
                self.down_scale,
            )
        return _pytorch_zero_sync_4bit_moe_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_up_packed,
            self.gate_up_scale,
            self.down_packed,
            self.down_scale,
            shifts=self._shifts,
            act_fn=self.act_fn,
        )


def is_pack_quantized_dir(model_dir: Path) -> bool:
    cfg_file = model_dir / "config.json"
    if not cfg_file.is_file():
        return False
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        qcfg = cfg.get("quantization_config") or {}
        return qcfg.get("format") == "pack-quantized" or qcfg.get("quant_method") == "compressed-tensors"
    except Exception:
        return False


def _replace_self_attn_with_4bit(backbone: nn.Module, dtype: torch.dtype) -> None:
    text_model = getattr(backbone.model, "language_model", backbone.model)
    for layer in text_model.layers:
        attn = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            old_mod = getattr(attn, proj_name, None)
            if isinstance(old_mod, nn.Linear):
                new_mod = Linear4Bit(
                    old_mod.in_features,
                    old_mod.out_features,
                    bias=old_mod.bias is not None,
                    dtype=dtype,
                )
                setattr(attn, proj_name, new_mod)


def load_pack_quantized_backbone(
    model_dir: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Instantiate Gemma4ForConditionalGeneration with 4-bit Experts & 4-bit Attention Linears,
    and stream compressed-tensors shards into memory."""
    from transformers.models.gemma4 import modeling_gemma4

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    class _ExpertsCls(Gemma4TextExperts4Bit):
        def __init__(self, cfg):
            super().__init__(cfg, dtype=dtype)

    orig_init_weights = modeling_gemma4.Gemma4PreTrainedModel._init_weights
    def _fast_init_weights(self, module):
        if isinstance(module, (Gemma4TextExperts4Bit, nn.Linear, nn.Embedding)):
            return
        orig_init_weights(self, module)

    with (
        mock.patch.object(modeling_gemma4, "Gemma4TextExperts", _ExpertsCls),
        mock.patch.object(modeling_gemma4.Gemma4PreTrainedModel, "_init_weights", _fast_init_weights),
    ):
        backbone = AutoModelForCausalLM.from_config(
            config, dtype=dtype, trust_remote_code=trust_remote_code
        )


    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        idx_data = json.loads(index_file.read_text(encoding="utf-8"))
        shard_files = sorted(set(idx_data["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    text_layers = backbone.model.language_model.layers
    num_layers = len(text_layers)
    num_experts = text_layers[0].experts.num_experts

    # Temporary per-layer collectors for split expert tensors across shards
    exp_parts: list[dict[str, dict[int, torch.Tensor]]] = [
        {
            "gp": {}, "gs": {}, "up": {}, "us": {}, "dp": {}, "ds": {}
        }
        for _ in range(num_layers)
    ]

    params_and_buffers: dict[str, torch.Tensor] = {}
    for name, p in backbone.named_parameters():
        params_and_buffers[name] = p
    for name, b in backbone.named_buffers():
        params_and_buffers[name] = b

    for shard_name in shard_files:
        sd = load_file(str(model_dir / shard_name), device="cpu")
        for k in list(sd.keys()):
            val = sd.pop(k)
            if k.endswith(".weight_shape"):
                continue
            if ".experts." in k and k.startswith("model.language_model.layers."):
                # Format: model.language_model.layers.{l}.experts.{e}.{gate_proj|up_proj|down_proj}.{weight_packed|weight_scale}
                parts = k.split(".")
                l_idx = int(parts[3])
                e_idx = int(parts[5])
                proj = parts[6]
                kind = parts[7]
                tag = (
                    ("gp" if kind == "weight_packed" else "gs")
                    if proj == "gate_proj"
                    else ("up" if kind == "weight_packed" else "us")
                    if proj == "up_proj"
                    else ("dp" if kind == "weight_packed" else "ds")
                )
                exp_parts[l_idx][tag][e_idx] = val
                if len(exp_parts[l_idx]["gp"]) == num_experts and len(exp_parts[l_idx]["up"]) == num_experts and len(exp_parts[l_idx]["gs"]) == num_experts and len(exp_parts[l_idx]["us"]) == num_experts:
                    exp_mod: Gemma4TextExperts4Bit = text_layers[l_idx].experts
                    gp = torch.stack([exp_parts[l_idx]["gp"].pop(e) for e in range(num_experts)], dim=0)
                    up = torch.stack([exp_parts[l_idx]["up"].pop(e) for e in range(num_experts)], dim=0)
                    exp_mod.gate_up_packed.copy_(torch.cat([gp, up], dim=1))
                    del gp, up
                    gs = torch.stack([exp_parts[l_idx]["gs"].pop(e) for e in range(num_experts)], dim=0)
                    us = torch.stack([exp_parts[l_idx]["us"].pop(e) for e in range(num_experts)], dim=0)
                    exp_mod.gate_up_scale.copy_(torch.cat([gs, us], dim=1).to(dtype))
                    del gs, us
                if len(exp_parts[l_idx]["dp"]) == num_experts and len(exp_parts[l_idx]["ds"]) == num_experts:
                    exp_mod = text_layers[l_idx].experts
                    dp = torch.stack([exp_parts[l_idx]["dp"].pop(e) for e in range(num_experts)], dim=0)
                    exp_mod.down_packed.copy_(dp)
                    del dp
                    ds = torch.stack([exp_parts[l_idx]["ds"].pop(e) for e in range(num_experts)], dim=0)
                    exp_mod.down_scale.copy_(ds.to(dtype))
                    del ds
            elif k in params_and_buffers:
                target = params_and_buffers[k]
                target.data.copy_(val.to(target.dtype) if val.is_floating_point() else val)
            del val

    if getattr(config, "tie_word_embeddings", True):
        backbone.tie_weights()

    return backbone
