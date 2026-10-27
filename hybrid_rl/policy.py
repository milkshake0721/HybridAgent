"""Policy loading and saving for HybridRL: the backbone is trained in bf16 (fp32 masters
live in the optimizer), HybridLink stays frozen, dropout is disabled and gradient
checkpointing enabled so rollout and replay compute the same function.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hybrid_agent.model import HybridAgentModel, add_anchor_tokens, ensure_chat_template, resolve_model_dir

_DROPOUT_NAMES = (
    "attention_dropout", "hidden_dropout", "hidden_dropout_prob", "resid_pdrop", "embd_pdrop", "dropout",
)


def disable_dropout(module: torch.nn.Module) -> int:
    """Zero every module- and config-level dropout; returns how many were changed."""
    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout) and child.p != 0.0:
            child.p = 0.0
            count += 1
        for name in _DROPOUT_NAMES:  # some attention classes copy the config value at init
            raw = getattr(child, name, None)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            if float(raw) != 0.0:
                setattr(child, name, 0.0)
                count += 1
    root_config = getattr(module, "config", None)
    configs = [root_config, getattr(root_config, "text_config", None)]
    get_text_config = getattr(root_config, "get_text_config", None)
    if callable(get_text_config):
        try:
            configs.append(get_text_config())
        except TypeError:
            pass
    seen: set[int] = set()
    for config in configs:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        for name in _DROPOUT_NAMES:
            if not hasattr(config, name):
                continue
            try:
                value = float(getattr(config, name) or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if value != 0.0:
                setattr(config, name, 0.0)
                count += 1
    return count


def load_policy(spec: str, device, *, freeze_experts: bool = False) -> tuple[HybridAgentModel, "AutoTokenizer"]:
    local = resolve_model_dir(spec)
    tokenizer = AutoTokenizer.from_pretrained(str(local))
    ensure_chat_template(tokenizer)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = HybridAgentModel.from_pretrained(local, dtype=torch.bfloat16, require_hybridlink=True)
    vocab = int(model.backbone.get_input_embeddings().weight.shape[0])
    if vocab != len(tokenizer):
        raise RuntimeError(
            f"tokenizer/model vocabulary mismatch ({len(tokenizer)} vs {vocab}); "
            "start from a complete HybridSFT checkpoint"
        )
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for mod in (
        model.backbone.get_decoder(),
        model.backbone.get_input_embeddings(),
        model.backbone.get_output_embeddings(),
    ):
        if mod is not None:
            for name, parameter in mod.named_parameters():
                if freeze_experts and ".experts." in f".{name}":
                    continue
                parameter.requires_grad_(True)
    for parameter in model.hybridlink.parameters():
        parameter.requires_grad_(False)
    disable_dropout(model.backbone)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.backbone.train()
    model.hybridlink.eval()
    return model, tokenizer


def save_checkpoint(model: HybridAgentModel, tokenizer, out_dir: str | Path, cfg=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, tokenizer)
    if cfg is not None:
        (out / "rl_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
