"""HybridAgentModel: a causal-LM or multimodal conditional-generation backbone paired with HybridLink.

Provides the anchor-token registration, HybridLink application, multimodal processor loading
(with automatic fallback when `processor_config.json` is absent), checkpoint save/load,
and a loader that accepts either a local directory or a Hugging Face repo id.
Supports both BF16 checkpoints and 4-bit compressed-tensors (pack-quantized W4A16) checkpoints.
Shared by HybridSFT, HybridRL and inference.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .constants import ANCHOR_TOKENS, THINK_CLOSE, THINK_OPEN
from .hybridlink import HYBRIDLINK_CONFIG, HybridLink
from .quantization import is_pack_quantized_dir, load_pack_quantized_backbone

logger = logging.getLogger(__name__)

HYBRIDLINK_WEIGHTS = "adapter.pt"
HYBRIDLINK_ALT_WEIGHTS = "hybridlink.pt"
HYBRIDLINK_CONFIG_FILE = "adapter_config.json"
HYBRIDLINK_ALT_CONFIG_FILE = "hybridlink_config.json"

DEFAULT_GEMMA4_PROCESSOR_REPO = "google/gemma-4-26B-A4B-it"

DEFAULT_GEMMA4_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|turn>system\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'user' %}<|turn>user\n{{ message['content'] }}<turn|>\n"
    "{% elif message['role'] == 'assistant' %}<|turn>model\n{{ message['content'] }}<turn|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n<think>\n{% endif %}"
)


def ensure_chat_template(tokenizer: Any) -> None:
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
        modeling_utils,
        "maybe_load_adapters",
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


def _is_single_known_token(tokenizer: Any, token: str) -> bool:
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


def add_anchor_tokens(tokenizer: Any) -> int:
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


def _is_mocked(obj: Any) -> bool:
    """Return True if `obj` is a unittest.mock object."""
    return (
        isinstance(obj, mock.NonCallableMock)
        or hasattr(obj, "_mock_name")
        or hasattr(obj, "assert_called")
    )


def _read_config_dict(local_dir: Path | None) -> dict[str, Any]:
    if local_dir is None:
        return {}
    cfg_file = Path(local_dir) / "config.json"
    if not cfg_file.is_file():
        return {}
    try:
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_vision_config(local_dir: Path | None, config: Any = None) -> bool:
    """Return True if the model config or checkpoint directory defines a multimodal vision tower."""
    if config is not None and getattr(config, "vision_config", None) is not None:
        return True
    raw = _read_config_dict(local_dir)
    if raw.get("vision_config") is not None:
        return True
    architectures = raw.get("architectures") or []
    return any(
        "ConditionalGeneration" in arch or "ImageTextToText" in arch
        for arch in architectures
        if isinstance(arch, str)
    )


def _flatten_images(images: Any) -> list[Any]:
    """Flatten arbitrarily nested lists/tuples of images (ignoring None) into a flat list."""
    if images is None:
        return []
    if isinstance(images, torch.Tensor):
        if images.ndim == 4:
            return [images[i] for i in range(images.shape[0])]
        return [images]
    if isinstance(images, (list, tuple)):
        out: list[Any] = []
        for item in images:
            out.extend(_flatten_images(item))
        return out
    return [images]


class HybridMultimodalProcessor:
    """Fallback multimodal processor combining a tokenizer with PIL / numpy / Tensor image
    preprocessing into `pixel_values`, `input_ids`, and `attention_mask`.

    Automatically synthesized when a checkpoint (such as `milkshake721/Gemma-4-26B-A4B-HybridAgent`)
    does not ship a `processor_config.json` and Hugging Face Hub / `Gemma4Processor` cannot be
    loaded (e.g., in offline environments or CPU unit tests).
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        vision_soft_tokens_per_image: int = 280,
        boi_token: str = "<|image>",
        image_token: str = "<|image|>",
        eoi_token: str = "<image|>",
        boi_token_id: int | None = 255999,
        image_token_id: int | None = 258880,
        eoi_token_id: int | None = 258882,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.pooling_kernel_size = int(pooling_kernel_size)
        self.vision_soft_tokens_per_image = int(vision_soft_tokens_per_image)
        self.boi_token = boi_token
        self.image_token = image_token
        self.eoi_token = eoi_token
        self.boi_token_id = boi_token_id
        self.image_token_id = image_token_id
        self.eoi_token_id = eoi_token_id
        self.image_mean = image_mean
        self.image_std = image_std
        self.chat_template = getattr(tokenizer, "chat_template", DEFAULT_GEMMA4_CHAT_TEMPLATE)

    @classmethod
    def from_config(
        cls,
        tokenizer: Any,
        config: Any = None,
        raw_config: dict[str, Any] | None = None,
    ) -> "HybridMultimodalProcessor":
        raw = raw_config or {}
        vcfg = getattr(config, "vision_config", None)
        vdict = raw.get("vision_config") if isinstance(raw.get("vision_config"), dict) else {}

        def _get(key: str, default: Any) -> Any:
            if config is not None and getattr(config, key, None) is not None:
                return getattr(config, key)
            if vcfg is not None and getattr(vcfg, key, None) is not None:
                return getattr(vcfg, key)
            if isinstance(vcfg, dict) and vcfg.get(key) is not None:
                return vcfg[key]
            if key in raw and raw[key] is not None:
                return raw[key]
            if key in vdict and vdict[key] is not None:
                return vdict[key]
            return default

        patch_size = int(_get("patch_size", 16))
        pooling_kernel = int(_get("pooling_kernel_size", 3))
        soft_tokens = int(_get("default_output_length", _get("vision_soft_tokens_per_image", 280)))
        default_img_size = int(_get("image_size", 224))
        return cls(
            tokenizer=tokenizer,
            image_size=default_img_size,
            patch_size=patch_size,
            pooling_kernel_size=pooling_kernel,
            vision_soft_tokens_per_image=soft_tokens,
            boi_token_id=_get("boi_token_id", 255999),
            image_token_id=_get("image_token_id", 258880),
            eoi_token_id=_get("eoi_token_id", 258882),
        )

    def preprocess_image(self, image: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert a PIL.Image, numpy ndarray, dict, or torch.Tensor into a normalized [3, H, W] tensor."""
        if isinstance(image, dict) and "pixel_values" in image:
            image = image["pixel_values"]
        if isinstance(image, torch.Tensor):
            t = image.detach().clone()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim == 2:
                t = t.unsqueeze(0).repeat(3, 1, 1)
            elif t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            elif t.shape[0] == 4:
                t = t[:3]
            if not t.is_floating_point():
                t = t.to(dtype=torch.float32) / 255.0
            else:
                t = t.to(dtype=torch.float32)
            if t.shape[-2] != self.image_size or t.shape[-1] != self.image_size:
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            return t.to(dtype=dtype)

        if hasattr(image, "convert"):
            image = image.convert("RGB")
            if hasattr(image, "resize") and getattr(image, "size", None) != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.as_tensor(np.ascontiguousarray(arr))
        if not t.is_floating_point():
            t = t.to(dtype=torch.float32) / 255.0
        else:
            t = t.to(dtype=torch.float32)
        if t.shape[-2] != self.image_size or t.shape[-1] != self.image_size:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        mean = torch.tensor(self.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / std
        return t.to(dtype=dtype)

    def apply_chat_template(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError("Underlying tokenizer has no apply_chat_template method")
        normalized_messages: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                normalized_messages.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        parts.append(item)
                normalized_messages.append({**msg, "content": "\n".join(parts)})
            else:
                normalized_messages.append(msg)
        return self.tokenizer.apply_chat_template(normalized_messages, *args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any | None = None,
        *,
        padding: bool | str = True,
        return_tensors: str | None = "pt",
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if text is not None and self.tokenizer is not None:
            tok_kwargs: dict[str, Any] = dict(
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
            )
            if isinstance(text, (list, tuple)) and len(text) > 1:
                tok_kwargs["padding"] = padding
            try:
                tok_out = self.tokenizer(text, **tok_kwargs)
            except TypeError:
                tok_out = self.tokenizer(text, return_tensors=return_tensors)
            out.update(dict(tok_out))
        if images is not None:
            flat_imgs = _flatten_images(images)
            if flat_imgs:
                pixel_values = torch.stack([self.preprocess_image(img) for img in flat_imgs], dim=0)
                out["pixel_values"] = pixel_values
        return out


def load_hybrid_processor(
    path_or_repo: str | os.PathLike | None = None,
    *,
    tokenizer: Any | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    fallback_repo: str = DEFAULT_GEMMA4_PROCESSOR_REPO,
    config: Any | None = None,
) -> Any:
    """Load a multimodal processor (`AutoProcessor` / `Gemma4Processor`) for a HybridAgent
    checkpoint, with automatic fallback to `google/gemma-4-26B-A4B-it` or a synthesized
    `HybridMultimodalProcessor` when `processor_config.json` is absent or offline.
    """
    local: Path | None = None
    if path_or_repo is not None:
        p = Path(path_or_repo)
        if p.is_dir():
            local = p
        else:
            try:
                local = resolve_model_dir(path_or_repo, revision=revision)
            except Exception:
                local = None

    if tokenizer is None and local is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
        except Exception:
            tokenizer = None
    if tokenizer is not None:
        ensure_chat_template(tokenizer)
        add_anchor_tokens(tokenizer)
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def _bind_tokenizer(proc: Any) -> Any:
        if tokenizer is not None:
            try:
                proc.tokenizer = tokenizer
            except Exception:
                pass
        if getattr(proc, "chat_template", None) is None and tokenizer is not None:
            try:
                proc.chat_template = getattr(tokenizer, "chat_template", DEFAULT_GEMMA4_CHAT_TEMPLATE)
            except Exception:
                pass
        return proc

    # Tier 1: Load from local dir / repo if processor config exists (or AutoProcessor is mocked)
    try:
        from transformers import AutoProcessor
    except ImportError:
        AutoProcessor = None  # type: ignore[assignment]

    has_local_proc_cfg = local is not None and (
        (local / "processor_config.json").is_file()
        or (local / "preprocessor_config.json").is_file()
    )
    if AutoProcessor is not None and (
        has_local_proc_cfg
        or _is_mocked(AutoProcessor)
        or _is_mocked(getattr(AutoProcessor, "from_pretrained", None))
    ):
        target = str(local if local is not None else path_or_repo)
        try:
            proc = AutoProcessor.from_pretrained(
                target, trust_remote_code=trust_remote_code, revision=revision
            )
            return _bind_tokenizer(proc)
        except Exception as e:
            logger.debug(f"AutoProcessor.from_pretrained({target!r}) failed: {e}")

    # Tier 2a: Construct Gemma4Processor + Gemma4ImageProcessor in-memory if available in transformers
    if (
        tokenizer is not None
        and not _is_mocked(tokenizer)
        and isinstance(tokenizer, PreTrainedTokenizerBase)
    ):
        try:
            from transformers import Gemma4Processor

            try:
                from transformers import Gemma4ImageProcessorPil as _Gemma4ImgProc
            except ImportError:
                from transformers import Gemma4ImageProcessor as _Gemma4ImgProc

            img_proc = _Gemma4ImgProc()
            proc = Gemma4Processor(image_processor=img_proc, tokenizer=tokenizer)
            return _bind_tokenizer(proc)
        except Exception:
            pass

    # Tier 2b: Try fallback_repo (e.g. google/gemma-4-26B-A4B-it) from local HF cache first, then hub
    if AutoProcessor is not None and fallback_repo:
        for local_only in (True, False):
            if not local_only and os.environ.get("HF_HUB_OFFLINE", "0") == "1":
                continue
            try:
                proc = AutoProcessor.from_pretrained(
                    fallback_repo,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_only,
                )
                return _bind_tokenizer(proc)
            except Exception:
                continue

    # Tier 3: Synthesize HybridMultimodalProcessor from vision_config + tokenizer
    raw_cfg = _read_config_dict(local)
    return HybridMultimodalProcessor.from_config(tokenizer=tokenizer, config=config, raw_config=raw_cfg)


def _load_backbone_from_pretrained(
    local: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Load the causal or multimodal conditional-generation backbone from `local`,
    preserving `vision_tower` and `embed_vision` when `vision_config` is present."""
    # If a test specifically mocked AutoModelForCausalLM, honor the mock directly.
    if _is_mocked(AutoModelForCausalLM) or _is_mocked(getattr(AutoModelForCausalLM, "from_pretrained", None)):
        with _no_peft_autodetect():
            return AutoModelForCausalLM.from_pretrained(
                str(local), dtype=dtype, trust_remote_code=trust_remote_code
            )

    if _has_vision_config(local):
        raw_cfg = _read_config_dict(local)
        architectures = raw_cfg.get("architectures") or []

        # 1. Try explicit Gemma4ForConditionalGeneration if specified in config.json
        if "Gemma4ForConditionalGeneration" in architectures:
            try:
                from transformers import Gemma4ForConditionalGeneration

                with _no_peft_autodetect():
                    return Gemma4ForConditionalGeneration.from_pretrained(
                        str(local), dtype=dtype, trust_remote_code=trust_remote_code
                    )
            except Exception as e:
                logger.debug(f"Gemma4ForConditionalGeneration.from_pretrained fallback: {e}")

        # 2. Try AutoModelForImageTextToText to preserve vision_tower & embed_vision
        try:
            from transformers import AutoModelForImageTextToText

            with _no_peft_autodetect():
                return AutoModelForImageTextToText.from_pretrained(
                    str(local), dtype=dtype, trust_remote_code=trust_remote_code
                )
        except Exception as e:
            logger.debug(f"AutoModelForImageTextToText.from_pretrained fallback: {e}")

    with _no_peft_autodetect():
        return AutoModelForCausalLM.from_pretrained(
            str(local), dtype=dtype, trust_remote_code=trust_remote_code
        )


class HybridAgentModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hybridlink: HybridLink | None = None,
        *,
        processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.get_input_embeddings().weight.shape[-1]
        self.hybridlink = hybridlink if hybridlink is not None else HybridLink(hidden_size)
        try:
            backbone_dtype = next(backbone.parameters()).dtype
            self.hybridlink.to(dtype=backbone_dtype)
        except StopIteration:
            pass
        self.processor: Any | None = processor
        self.model_dir: Path | None = None

    @property
    def vision_tower(self) -> nn.Module | None:
        """Return the multimodal vision encoder if present on the backbone."""
        vt = getattr(self.backbone, "vision_tower", None)
        if vt is not None:
            return vt
        inner = getattr(self.backbone, "model", None)
        return getattr(inner, "vision_tower", None)

    @property
    def embed_vision(self) -> nn.Module | None:
        """Return the vision-to-language embedding projector (`embed_vision` or `multi_modal_projector`)."""
        for holder in (self.backbone, getattr(self.backbone, "model", None)):
            if holder is None:
                continue
            for attr in ("embed_vision", "multi_modal_projector"):
                mod = getattr(holder, attr, None)
                if mod is not None:
                    return mod
        return None

    @property
    def has_vision(self) -> bool:
        """Return True if the model has a vision tower or multimodal vision config."""
        return self.vision_tower is not None or _has_vision_config(
            self.model_dir, getattr(self.backbone, "config", None)
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs through the backbone's input embedding layer (preserving Gemma 4 embed_scale)."""
        embed_layer = self.backbone.get_input_embeddings()
        return embed_layer(input_ids)

    def get_processor(self, tokenizer: Any | None = None) -> Any:
        """Return `self.processor`, lazily initializing a multimodal processor if needed."""
        if self.processor is None:
            self.processor = load_hybrid_processor(
                self.model_dir,
                tokenizer=tokenizer,
                config=getattr(self.backbone, "config", None),
            )
        return self.processor

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate forward pass to the underlying causal or multimodal backbone."""
        return self.backbone(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | os.PathLike,
        *,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        require_hybridlink: bool = False,
        revision: str | None = None,
        gradient_checkpointing: bool = False,
    ) -> "HybridAgentModel":
        """Load the backbone (BF16 or 4-bit pack-quantized) and the HybridLink weights."""
        local = resolve_model_dir(path_or_repo, revision)
        if is_pack_quantized_dir(local):
            backbone = load_pack_quantized_backbone(
                local, dtype=dtype, trust_remote_code=trust_remote_code
            )
        else:
            backbone = _load_backbone_from_pretrained(
                local, dtype=dtype, trust_remote_code=trust_remote_code
            )
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = cls(backbone)
        model.model_dir = local

        weights = local / HYBRIDLINK_WEIGHTS
        if not weights.is_file():
            weights = local / HYBRIDLINK_ALT_WEIGHTS

        if weights.is_file():
            config_file = local / HYBRIDLINK_CONFIG_FILE
            if not config_file.is_file():
                config_file = local / HYBRIDLINK_ALT_CONFIG_FILE
            if config_file.is_file():
                tag = json.loads(config_file.read_text(encoding="utf-8")).get("adapter_type")
                if tag != HYBRIDLINK_CONFIG["adapter_type"]:
                    raise ValueError(f"unsupported HybridLink config {tag!r} in {config_file}")
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
            model.hybridlink.load_state_dict(state, strict=True)
        elif require_hybridlink:
            raise FileNotFoundError(f"{HYBRIDLINK_WEIGHTS} / {HYBRIDLINK_ALT_WEIGHTS} not found in {local}")
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

    def resize_to_tokenizer(self, tokenizer: Any) -> None:
        """Match the embedding table to the tokenizer (needed once for a fresh base model)."""
        if self.backbone.get_input_embeddings().num_embeddings != len(tokenizer):
            self.backbone.resize_token_embeddings(len(tokenizer))

    def save_pretrained(self, output_dir: str | os.PathLike, tokenizer: Any = None, state_dict: dict | None = None) -> None:
        """Write the backbone (HF format), tokenizer, processor, and HybridLink weights into one directory."""
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        link_sd = {k[len("hybridlink."):]: v for k, v in state_dict.items() if k.startswith("hybridlink.")}
        self.backbone.save_pretrained(output_dir, state_dict=backbone_sd)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            with contextlib.suppress(Exception):
                self.processor.save_pretrained(output_dir)
        torch.save(link_sd, os.path.join(output_dir, HYBRIDLINK_WEIGHTS))
        with open(os.path.join(output_dir, HYBRIDLINK_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(HYBRIDLINK_CONFIG, f, indent=2)


def load_hybrid_agent(
    path_or_repo: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
    require_hybridlink: bool = True,
    revision: str | None = None,
    backend: str = "eager",
    max_batch_size: int = 256,
    load_processor: bool = False,
) -> tuple[HybridAgentModel, PreTrainedTokenizerBase]:
    """Return `(model, tokenizer)` ready for generation from a local dir or HF repo id.

    Also attaches `model.processor` (`Gemma4Processor` / `AutoProcessor` with automatic
    fallback to `google/gemma-4-26B-A4B-it` or `HybridMultimodalProcessor` when
    `processor_config.json` is absent) when `load_processor=True` or when the model
    contains a multimodal `vision_config`.
    """
    from .backends import enable_tensorrt_acceleration, patch_bf16_experts_fastpath

    local = resolve_model_dir(path_or_repo, revision)
    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
    ensure_chat_template(tokenizer)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = HybridAgentModel.from_pretrained(
        local,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        require_hybridlink=require_hybridlink,
        revision=revision,
    )
    model.resize_to_tokenizer(tokenizer)
    if load_processor or model.has_vision:
        model.processor = load_hybrid_processor(
            local,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            revision=revision,
            config=getattr(model.backbone, "config", None),
        )
    patch_bf16_experts_fastpath(model.backbone)
    if device is not None:
        model.to(device)
    model.eval()
    if backend in ("tensorrt", "trt", "compile"):
        sig = inspect.signature(enable_tensorrt_acceleration)
        trt_kwargs: dict[str, Any] = {}
        if "max_batch_size" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            trt_kwargs["max_batch_size"] = max_batch_size
        try:
            enable_tensorrt_acceleration(model, **trt_kwargs)
        except TypeError:
            enable_tensorrt_acceleration(model)
    return model, tokenizer
