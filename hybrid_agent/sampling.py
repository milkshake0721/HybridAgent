"""Batched decoding primitives for the hybrid generation engine: fan a prefilled KV cache
out to B rows, run one lockstep forward with per-row attention masks and positions, and
sample text tokens under the mode-dependent token masks.
"""
from __future__ import annotations

import torch


def _cache_layers(past_kv) -> list:
    layers = getattr(past_kv, "layers", None)
    if layers is None:
        raise TypeError(f"unsupported cache object {type(past_kv).__name__!r}: no .layers")
    return list(layers)


def _layer_tensor_attrs(layer) -> list[tuple[str, torch.Tensor]]:
    """Every tensor attribute of a cache layer that carries a batch dimension."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, value in vars(layer).items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            out.append((name, value))
    return out


def expand_cache_batch(past_kv, batch_size: int) -> None:
    """Expand a batch-1 (or batch-M) prefill cache to `batch_size` rows in place."""
    if past_kv is None:
        return
    for layer in _cache_layers(past_kv):
        for name, tensor in _layer_tensor_attrs(layer):
            cur_b = tensor.shape[0]
            if cur_b == batch_size:
                continue
            if cur_b < 1 or batch_size % cur_b != 0:
                raise ValueError(
                    f"cannot expand cache attr {name!r} with batch {cur_b} to {batch_size}"
                )
            repeats = batch_size // cur_b
            setattr(layer, name, tensor.repeat_interleave(repeats, dim=0).contiguous())
        if hasattr(layer, "batch_size") and isinstance(getattr(layer, "batch_size"), int):
            layer.batch_size = batch_size


def forward_step(
    lm,
    *,
    input_embeds: torch.Tensor,        # [B, K, D] (K=1 for decode, K>=1 for multi-token FEED)
    past_key_values,
    attention_mask: torch.Tensor,      # [B, L_total] incl. the K new slots
    position_ids: torch.Tensor,        # [B, K] per-row true RoPE positions
    cache_position: torch.Tensor,      # [K] global cache slot index
):
    """One lockstep decode or multi-token FEED forward; returns (cache, next logits [B, V], last hidden [B, D])."""
    kwargs = dict(
        inputs_embeds=input_embeds,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    if input_embeds.shape[1] > 1:
        try:
            outputs = lm(**kwargs, logits_to_keep=1)
        except TypeError:
            outputs = lm(**kwargs)
    else:
        outputs = lm(**kwargs)
    return (
        outputs.past_key_values,
        outputs.logits[:, -1, :],
        outputs.hidden_states[-1][:, -1, :],
    )


def token_log_probs(
    logits: torch.Tensor,              # [B, V]
    temperature: float,
    top_p: float,
    top_k: int,
    blocked_rows: torch.Tensor,        # [B, V] bool, True = blocked
) -> torch.Tensor:
    """Log-probs of the sampling distribution: block -> temperature -> top-k -> softmax ->
    top-p -> renormalize. Excluded tokens get -inf."""
    if temperature <= 0.0:
        masked = logits.masked_fill(blocked_rows, float("-inf"))
        selected = masked.argmax(dim=-1, keepdim=True)
        out = torch.full(logits.shape, float("-inf"), dtype=torch.float32, device=logits.device)
        return out.scatter(1, selected, 0.0)

    x = logits.float().masked_fill(blocked_rows, float("-inf"))
    scaled = x / max(temperature, 1e-5)
    if top_k and 0 < top_k < scaled.shape[-1]:
        kth = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumulative - sorted_probs) <= top_p
        keep[..., 0] = True
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return torch.log(probs.clamp_min(1e-45)).masked_fill(probs <= 0, float("-inf"))


def sample_rows(
    log_probs: torch.Tensor,           # [B, V]
    rows: list[int],
    rng: torch.Generator | None = None,
    temperature: float | None = None,
) -> dict[int, int]:
    """Sample one token per listed row in a single vectorized operation and single host transfer."""
    if not rows:
        return {}
    sorted_rows = sorted(rows)
    row_idx = torch.as_tensor(sorted_rows, dtype=torch.long, device=log_probs.device)
    sub_log_probs = log_probs.index_select(0, row_idx)  # [N, V]

    is_greedy = temperature is not None and temperature <= 0.0
    if not is_greedy and temperature is None:
        # Detect deterministic 1-hot log-probs produced by token_log_probs(temperature <= 0.0)
        finite_counts = torch.isfinite(sub_log_probs).sum(dim=-1)
        if bool((finite_counts <= 1).all()):
            is_greedy = True

    if is_greedy:
        sampled = sub_log_probs.argmax(dim=-1).tolist()
    else:
        probs = sub_log_probs.exp()
        row_sums = probs.sum(dim=-1, keepdim=True)
        invalid = ~(row_sums > 0.0)
        if bool(invalid.any()):
            fallback = torch.zeros_like(probs)
            best = sub_log_probs.argmax(dim=-1, keepdim=True)
            fallback.scatter_(1, best, 1.0)
            probs = torch.where(invalid, fallback, probs)
        sampled_tensor = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        sampled = sampled_tensor.tolist()

    if isinstance(sampled, int):
        sampled = [sampled]
    return {r: int(tok) for r, tok in zip(sorted_rows, sampled)}
