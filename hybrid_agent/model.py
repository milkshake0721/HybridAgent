"""HybridAgentModel: a causal-LM backbone paired with HybridLink.

Provides the anchor-token registration, HybridLink application, checkpoint save/load
and a loader that accepts either a local directory or a Hugging Face repo id.
Shared by HybridSFT, HybridRL and inference.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import ANCHOR_TOKENS, THINK_CLOSE, THINK_OPEN
from .hybridlink import HYBRIDLINK_CONFIG, HybridLink

HYBRIDLINK_WEIGHTS = "adapter.pt"
HYBRIDLINK_CONFIG_FILE = "adapter_config.json"

DEFAULT_GEMMA4_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|turn>system\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'user' %}<|turn>user\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'assistant' %}<|turn>model\n{{ message['content'] }}<turn|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n<think>\n{% endif %}"
)


def ensure_chat_template(tokenizer) -> None:
    """Assign a default chat template when a base model (e.g. Gemma 4) has none."""
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = DEFAULT_GEMMA4_CHAT_TEMPLATE


@contextlib.contextmanager
def _no_peft_autodetect():
    """HybridLink's `adapter_config.json` is not a PEFT config, but when `peft` is installed
    transformers treats any checkpoint containing that file as a PEFT adapter and fails to
    load it. Disable that auto-detection while loading the backbone."""
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "maybe_load_adapters"):
        yield
        return
    with mock.patch.object(
        modeling_utils, "maybe_load_adapters",
        lambda path, download_kwargs, **adapter_kwargs: (None, path, adapter_kwargs),
    ):
        yield


def resolve_model_dir(path_or_repo: str | os.PathLike, revision: str | None = None) -> Path:
    """Local directory as is; otherwise download the Hugging Face repo snapshot."""
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path_or_repo), revision=revision))


def _is_single_known_token(tokenizer, token: str) -> bool:
    """True when `token` is both a known single vocabulary entry and encodes to that single id."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1 or (unk_id is not None and ids[0] == unk_id):
        return False
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        tok_id = convert(token)
        if tok_id is None or (unk_id is not None and tok_id == unk_id) or tok_id != ids[0]:
            return False
    return True


def add_anchor_tokens(tokenizer) -> int:
    """Register `<latent>` / `</latent>` as special tokens (plus `<think>` / `</think>` for
    tokenizers like Gemma 4 where they are not single tokens); returns how many were added."""
    known = set(getattr(tokenizer, "all_special_tokens", None) or [])
    missing = [tok for tok in ANCHOR_TOKENS if tok not in known]
    for tok in (THINK_OPEN, THINK_CLOSE):
        if tok not in known and tok not in missing and not _is_single_known_token(tokenizer, tok):
            missing.append(tok)
    if missing:
        existing = list(getattr(tokenizer, "additional_special_tokens", None) or [])
        merged = existing + [t for t in missing if t not in existing]
        tokenizer.add_special_tokens({"additional_special_tokens": merged})
    return len(missing)


class HybridAgentModel(nn.Module):
    def __init__(self, backbone: nn.Module, hybridlink: HybridLink | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.get_input_embeddings().weight.shape[-1]
        self.hybridlink = hybridlink if hybridlink is not None else HybridLink(hidden_size)
        self.hybridlink.to(dtype=next(backbone.parameters()).dtype)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | os.PathLike,
        *,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        require_hybridlink: bool = False,
        revision: str | None = None,
    ) -> "HybridAgentModel":
        """Load the backbone and, when present, the HybridLink weights of a checkpoint."""
        local = resolve_model_dir(path_or_repo, revision)
        with _no_peft_autodetect():
            backbone = AutoModelForCausalLM.from_pretrained(
                str(local), dtype=dtype, trust_remote_code=trust_remote_code
            )
        model = cls(backbone)
        weights = local / HYBRIDLINK_WEIGHTS
        if weights.is_file():
            config_file = local / HYBRIDLINK_CONFIG_FILE
            if config_file.is_file():
                tag = json.loads(config_file.read_text(encoding="utf-8")).get("adapter_type")
                if tag != HYBRIDLINK_CONFIG["adapter_type"]:
                    raise ValueError(f"unsupported HybridLink config {tag!r} in {config_file}")
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
            model.hybridlink.load_state_dict(state, strict=True)
        elif require_hybridlink:
            raise FileNotFoundError(f"{HYBRIDLINK_WEIGHTS} not found in {local}")
        return model

    def run_hybridlink(
        self, hidden_states: torch.Tensor, output_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """R(h): next-step latent inputs from last-layer hidden states."""
        link_dtype = next(self.hybridlink.parameters()).dtype
        x = hidden_states if hidden_states.dtype == link_dtype else hidden_states.to(link_dtype)
        out = self.hybridlink(x)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def resize_to_tokenizer(self, tokenizer) -> None:
        """Match the embedding table to the tokenizer (needed once for a fresh base model)."""
        if self.backbone.get_input_embeddings().num_embeddings != len(tokenizer):
            self.backbone.resize_token_embeddings(len(tokenizer))

    def save_pretrained(self, output_dir: str | os.PathLike, tokenizer=None, state_dict=None) -> None:
        """Write the backbone (HF format), tokenizer and HybridLink weights into one directory."""
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        link_sd = {k[len("hybridlink."):]: v for k, v in state_dict.items() if k.startswith("hybridlink.")}
        self.backbone.save_pretrained(output_dir, state_dict=backbone_sd)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(link_sd, os.path.join(output_dir, HYBRIDLINK_WEIGHTS))
        with open(os.path.join(output_dir, HYBRIDLINK_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(HYBRIDLINK_CONFIG, f, indent=2)


def load_hybrid_agent(
    path_or_repo: str | os.PathLike,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
    require_hybridlink: bool = True,
    revision: str | None = None,
):
    """Return `(model, tokenizer)` ready for generation from a local dir or HF repo id."""
    local = resolve_model_dir(path_or_repo, revision)
    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
    ensure_chat_template(tokenizer)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = HybridAgentModel.from_pretrained(
        local, dtype=dtype, trust_remote_code=trust_remote_code,
        require_hybridlink=require_hybridlink, revision=revision,
    )
    model.resize_to_tokenizer(tokenizer)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer
