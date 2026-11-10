"""Hybrid generation engine: multi-turn generation that switches between
text mode and latent mode over one persistent KV cache per trajectory, the agent's
working memory. Text mode samples tokens until the entry anchor `<latent>` is emitted;
latent mode feeds HybridLink outputs back as the next inputs until the exit-anchor
readout fires; tool calls execute concurrently at a barrier and their observations are
appended to the working memory as text. Supports both B rollouts of one prompt and
multi-question batched generation across M distinct prompts in lockstep.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from .constants import (
    ENTRY_ANCHOR,
    EXIT_ANCHOR,
    RESULT_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_STOP_TAGS,
    special_token_id,
)
from .prompts import SYSTEM_PROMPT
from .readout import (
    LatentSpanReadout,
    nearest_token_ids,
    normalized_embeddings,
    pseudo_text,
    render_with_readout,
)
from .sampling import expand_cache_batch, forward_step, sample_rows, token_log_probs
from .tools import ToolExecutor

DECODE, LATENT, FEED, IDLE, DONE = "DECODE", "LATENT", "FEED", "IDLE", "DONE"

_MAX_FEED_CHUNK = 512


@dataclass
class RowState:
    """Per-trajectory state of one row of the lockstep batch."""
    idx: int
    mode: str = DECODE
    in_think: bool = False
    latent_steps: int = 0                       # steps of the open latent span
    generated_ids: list[int] = field(default_factory=list)
    text_ids_since_resume: list[int] = field(default_factory=list)
    kv_ledger: list[int | None] = field(default_factory=list)   # token per cache slot, None = latent
    generation_steps: int = 0                   # text tokens + latent steps
    latent_steps_total: int = 0
    latent_spans_opened: int = 0
    anchor_exits: int = 0
    forced_exits: int = 0
    turn_count: int = 1
    python_calls: int = 0
    search_calls: int = 0
    tool_requests: int = 0
    search_query_history: set[str] = field(default_factory=set)
    full_output: str = ""
    _last_decoded_len: int = 0
    _turn_start_char: int = 0
    turn_texts: list[str] = field(default_factory=list)
    turn_latent_steps: list[int] = field(default_factory=lambda: [0])
    pending_tool: tuple[str, str] | None = None
    feed_queue: list[int] = field(default_factory=list)
    readout_ids: list[int] = field(default_factory=list)        # readout of the open span
    _open_span_vecs: list[torch.Tensor] = field(default_factory=list)
    latent_spans: list[LatentSpanReadout] = field(default_factory=list)
    end_reason: str = ""                        # "eos" | "budget" | "tool_cap" | "max_turns"
    extra: dict = field(default_factory=dict)
    question_idx: int = 0
    sample_idx: int = 0
    question: str = ""
    prompt_len: int = 0
    prompt_ends_in_think: bool = False

    @property
    def true_len(self) -> int:
        return len(self.kv_ledger)


class HybridGenerationEngine:
    """Lockstep hybrid generation engine supporting both B rollouts of one prompt
    and multi-question batched generation across M distinct prompts (with B = M * batch_size),
    persistent KV working memory per row, multi-token tool observation prefill,
    and concurrent tool execution."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        question: str | Sequence[str] | Any = None,
        *legacy_pos_args: Any,
        images: Any | Sequence[Any] | None = None,
        processor: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        batch_size: int = 1,
        max_response_tokens: int = 8192,
        max_latent_steps: int = 32,
        min_latent_steps: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_rounds: int = 8,
        max_tool_calls: int | None = 6,
        max_python_calls: int | None = 6,
        max_search_calls: int | None = 6,
        tool_executor: ToolExecutor | None = None,
        record_hidden: bool = False,
        record_token_logprobs: bool = False,
        record_latent_readout: bool = False,
        count_tool_tokens: bool = False,
        terminate_on_tool_cap: bool = False,
        max_turns: int | None = None,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        think_guards: bool = True,
        reject_fabricated_result: bool = True,
        device: torch.device | str | None = None,
        tool_workers: int = 8,
    ) -> None:
        # Resolve legacy 6-positional-arg calling convention:
        # HybridGenerationEngine(model, tokenizer, device, tool_executor, system_prompt, question, ...)
        if len(legacy_pos_args) == 3:
            device_pos = question
            tool_executor_pos, system_prompt_pos, question = legacy_pos_args
            if device is None and device_pos is not None:
                device = device_pos
            if tool_executor is None and tool_executor_pos is not None:
                tool_executor = tool_executor_pos
            if system_prompt_pos is not None:
                system_prompt = system_prompt_pos
        elif len(legacy_pos_args) > 0:
            raise TypeError(
                f"HybridGenerationEngine.__init__ expected 3 or 6 positional arguments, "
                f"got {3 + len(legacy_pos_args)}"
            )

        if question is None:
            raise ValueError("`question` must be provided as a str or Sequence[str].")

        if device is None:
            try:
                device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")

        self.model = model
        self.backbone = model.backbone
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.tool_executor = tool_executor
        self.processor = processor if processor is not None else getattr(model, "processor", None)
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

        if isinstance(question, str):
            self.is_multi_question = False
            self.questions: list[str] = [question]
            self.question: str | list[str] = question
            self.samples_per_question = max(1, int(batch_size))
            self.B = self.samples_per_question
        elif isinstance(question, Sequence):
            self.is_multi_question = True
            self.questions = [str(q) for q in question]
            self.question = (
                self.questions[0]
                if len(self.questions) == 1
                else (list(self.questions) if self.questions else "")
            )
            self.samples_per_question = max(1, int(batch_size))
            self.B = len(self.questions) * self.samples_per_question
        else:
            raise TypeError(f"`question` must be str or Sequence[str], got {type(question).__name__!r}")

        self.images = images
        self.question_images = self._normalize_question_images(
            images,
            len(self.questions),
            is_single_question_input=not self.is_multi_question,
        )
        self.has_images = any(img is not None for img in self.question_images)
        if self.has_images and self.processor is None:
            from .model import load_hybrid_processor

            self.processor = load_hybrid_processor(
                getattr(model, "model_dir", None),
                tokenizer=tokenizer,
                config=getattr(self.backbone, "config", None),
            )
            try:
                model.processor = self.processor
            except Exception:
                pass
        self.last_pixel_values: torch.Tensor | None = None
        self.pixel_values: torch.Tensor | None = None

        # Budget per trajectory: text tokens + latent steps, or every slot after the
        # prompt (tool observations included) when count_tool_tokens is set.
        self.max_response_tokens = int(max_response_tokens)
        self.count_tool_tokens = bool(count_tool_tokens)
        self.max_tool_calls = max_tool_calls
        self.max_search_calls = max_search_calls
        self.max_python_calls = max_python_calls
        self.terminate_on_tool_cap = bool(terminate_on_tool_cap)
        resolved_turns = int(max_turns) if max_turns is not None else int(max_rounds)
        self.max_turns = resolved_turns
        self.max_rounds = resolved_turns
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.min_latent_steps = int(min_latent_steps)
        self.max_latent_steps = int(max_latent_steps)
        self.think_guards = bool(think_guards)
        self.reject_fabricated_result = bool(reject_fabricated_result)
        self.record_latent_readout = bool(record_latent_readout)
        self.record_hidden = bool(record_hidden)
        self.record_token_logprobs = bool(record_token_logprobs)
        self.tool_stop_tags = tuple(TOOL_STOP_TAGS)
        self.tool_workers = int(tool_workers)

        self.embed_layer = self.backbone.get_input_embeddings()
        self.latent_open_id = special_token_id(tokenizer, ENTRY_ANCHOR)
        self.latent_close_id = special_token_id(tokenizer, EXIT_ANCHOR)
        self.think_open_id = special_token_id(tokenizer, THINK_OPEN)
        self.think_close_id = special_token_id(tokenizer, THINK_CLOSE)
        self.eos_ids = self._collect_eos_ids()
        pad_id = tokenizer.pad_token_id
        self.pad_id = int(pad_id if pad_id is not None else tokenizer.eos_token_id)

        # Text-mode token masks: `</latent>` is only produced by the exit readout and
        # `<latent>` only inside <think>; think guards keep the think tags well nested.
        in_think = {self.latent_close_id}
        out_think = {self.latent_close_id, self.latent_open_id}
        if self.think_guards:
            in_think.add(self.think_open_id)
            out_think.add(self.think_close_id)
        self.blocked_token_sets = {"in": frozenset(in_think), "out": frozenset(out_think)}
        vocab = self.embed_layer.num_embeddings
        self._blocked_masks = {
            key: self._bool_mask(ids, vocab) for key, ids in self.blocked_token_sets.items()
        }
        # Lazy BF16 readout bank: built only when a latent span actually closes and needs readout
        self._readout_bank: torch.Tensor | None = None

        # Tag-closing character filter for fast _row_tail gating
        watched_tags = list(self.tool_stop_tags) + ([RESULT_OPEN] if self.reject_fabricated_result else [])
        self._tag_end_chars = frozenset(tag[-1] for tag in watched_tags if tag)
        self._tok_has_gt: dict[int, bool] = {}

        # Lockstep state
        self.rows = [
            RowState(
                idx=i,
                question_idx=i // self.samples_per_question,
                sample_idx=i % self.samples_per_question,
                question=self.questions[i // self.samples_per_question] if self.questions else "",
            )
            for i in range(self.B)
        ]
        self.past_kv = None
        self.next_logits = None      # [B, V]
        self.last_hidden = None      # [B, D]
        self.attn_mask = None        # [B, L_total] view into _attn_mask_buf
        self._attn_mask_buf: torch.Tensor | None = None
        self.positions = None        # [B] next RoPE position per row
        self.global_len = 0          # lockstep cache length (all rows equal)
        self.initial_prompt = ""
        self.initial_prompts: list[str] = []
        self.prompt_len = 0
        self.prompt_lens: list[int] = []
        self.prompt_ends_in_think = False

        self.llm_time = 0.0
        self.tool_time = 0.0
        self.total_time: float | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_question_images(
        images: Any,
        num_questions: int,
        *,
        is_single_question_input: bool = False,
    ) -> list[Any]:
        if num_questions <= 0:
            return []
        if images is None:
            return [None] * num_questions
        if is_single_question_input or num_questions == 1:
            if isinstance(images, (list, tuple)) and len(images) == 1:
                return [images[0]]
            return [images]
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] == num_questions:
            return [images[i] for i in range(num_questions)]
        if isinstance(images, (list, tuple)) and len(images) == num_questions:
            return list(images)
        raise ValueError(
            f"Expected `images` to have length {num_questions} to match `questions`, "
            f"got {len(images) if isinstance(images, (list, tuple)) else 1}."
        )

    def _get_readout_bank(self) -> torch.Tensor:
        if self._readout_bank is None:
            self._readout_bank = normalized_embeddings(self.embed_layer.weight)
        return self._readout_bank

    def _token_may_close_tag(self, tok_id: int) -> bool:
        if not self._tag_end_chars:
            return False
        res = self._tok_has_gt.get(tok_id)
        if res is None:
            tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            res = any(ch in tok_str for ch in self._tag_end_chars)
            self._tok_has_gt[tok_id] = res
        return res

    def _ensure_attn_mask_capacity(self, required_len: int) -> None:
        buf = self._attn_mask_buf
        if (
            buf is None
            or buf.shape[0] != self.B
            or buf.device != self.device
            or (
                self.attn_mask is not None
                and self.attn_mask.data_ptr() != buf.data_ptr()
            )
        ):
            cap = max(required_len + 1024, self.global_len + self.max_response_tokens + 512)
            new_buf = torch.zeros((self.B, cap), dtype=torch.long, device=self.device)
            if self.attn_mask is not None and self.global_len > 0:
                copy_len = min(self.global_len, self.attn_mask.shape[1])
                new_buf[:, :copy_len] = self.attn_mask[:, :copy_len]
            self._attn_mask_buf = new_buf
        elif required_len > buf.shape[1]:
            cap = max(required_len + 1024, buf.shape[1] * 2)
            new_buf = torch.zeros((self.B, cap), dtype=torch.long, device=self.device)
            if self.global_len > 0:
                new_buf[:, :self.global_len] = buf[:, :self.global_len]
            self._attn_mask_buf = new_buf

    def _bool_mask(self, ids, vocab) -> torch.Tensor:
        mask = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        if ids:
            mask[torch.tensor(sorted(ids), device=self.device, dtype=torch.long)] = True
        return mask

    def _collect_eos_ids(self) -> set[int]:
        config = getattr(self.backbone, "config", None)
        candidates = [
            self.tokenizer.eos_token_id,
            getattr(config, "eos_token_id", None),
            getattr(getattr(config, "text_config", None), "eos_token_id", None),
            getattr(getattr(self.backbone, "generation_config", None), "eos_token_id", None),
        ]
        eos: set[int] = set()
        for cand in candidates:
            if cand is None:
                continue
            for tok in cand if isinstance(cand, (list, tuple)) else [cand]:
                if tok is not None:
                    eos.add(int(tok))
        try:
            vocab = self.tokenizer.get_vocab()
        except (AttributeError, TypeError):
            vocab = {}
        for tok_str in ("<|endoftext|>", "<turn|>", "<end_of_turn>"):
            if tok_str in vocab:
                eos.add(int(vocab[tok_str]))
        return eos

    # ------------------------------------------------------------------
    # Prefill: single prompt at batch 1 (fanned out to B rows), or M distinct
    # prompts left-padded and prefilled in a single batched forward pass.
    # ------------------------------------------------------------------

    def _build_prompt_for_question(self, q: str, image: Any = None) -> str:
        from .model import ensure_chat_template

        ensure_chat_template(self.tokenizer)
        if image is not None and self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            img_list = list(image) if isinstance(image, (list, tuple)) else [image]
            img_list = [im for im in img_list if im is not None]
            if img_list:
                try:
                    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in img_list]
                    user_content.append({"type": "text", "text": q})
                    mm_messages = [
                        {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                        {"role": "user", "content": user_content},
                    ]
                    prompt = self.processor.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True,
                    )
                    if isinstance(prompt, str):
                        if not prompt.rstrip("\n").endswith(THINK_OPEN):
                            prompt = prompt + f"{THINK_OPEN}\n"
                        return prompt
                except Exception:
                    pass

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": q},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return prompt

    def _build_prompt(self) -> str:
        q0 = self.questions[0] if self.questions else ""
        img0 = self.question_images[0] if self.question_images else None
        prompt = self._build_prompt_for_question(q0, img0)
        self.initial_prompt = prompt
        return prompt

    def _extract_multimodal_kwargs(self, prompts: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """When `question_images` has images, build processor inputs or extract tensor `pixel_values`."""
        has_images = any(img is not None for img in self.question_images)
        if not has_images:
            return None, {}

        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = self.embed_layer.weight.dtype

        raw_imgs: list[Any] = []
        for img_entry in self.question_images:
            if img_entry is None:
                continue
            if isinstance(img_entry, (list, tuple)):
                raw_imgs.extend([im for im in img_entry if im is not None])
            else:
                raw_imgs.append(img_entry)

        if not raw_imgs:
            return None, {}

        # Case 1: Pre-computed tensor pixel_values or dicts passed directly in `images`
        if all(isinstance(im, torch.Tensor) for im in raw_imgs):
            pv = torch.cat(
                [im if im.ndim == 4 else im.unsqueeze(0) for im in raw_imgs], dim=0
            ).to(device=self.device, dtype=backbone_dtype)
            self.last_pixel_values = pv
            self.pixel_values = pv
            return None, {"pixel_values": pv}

        if all(isinstance(im, dict) and "pixel_values" in im for im in raw_imgs):
            pv = torch.cat(
                [
                    im["pixel_values"] if im["pixel_values"].ndim == 4 else im["pixel_values"].unsqueeze(0)
                    for im in raw_imgs
                ],
                dim=0,
            ).to(device=self.device, dtype=backbone_dtype)
            self.last_pixel_values = pv
            self.pixel_values = pv
            return None, {"pixel_values": pv}

        # Case 2: Use self.processor on PIL Images / numpy arrays / mixed image lists
        proc = self.processor or getattr(self.model, "processor", None)
        mm_extra: dict[str, Any] = {}
        tokenized: dict[str, Any] = {}
        if proc is not None and callable(proc):
            try:
                proc_inputs = proc(
                    text=prompts if len(prompts) > 1 else prompts[0],
                    images=raw_imgs,
                    return_tensors="pt",
                    padding=len(prompts) > 1,
                )
                for k, v in dict(proc_inputs).items():
                    if isinstance(v, torch.Tensor):
                        if v.is_floating_point():
                            v = v.to(device=self.device, dtype=backbone_dtype)
                        else:
                            v = v.to(device=self.device)
                    if k in ("input_ids", "attention_mask"):
                        tokenized[k] = v
                    else:
                        mm_extra[k] = v
            except Exception:
                pass

        # Fallback: synthesize pixel_values via HybridMultimodalProcessor if processor didn't return pixel_values
        if "pixel_values" not in mm_extra:
            from .model import HybridMultimodalProcessor

            fallback_proc = HybridMultimodalProcessor.from_config(
                tokenizer=self.tokenizer,
                config=getattr(self.backbone, "config", None),
            )
            pv = torch.stack(
                [fallback_proc.preprocess_image(im, dtype=backbone_dtype) for im in raw_imgs],
                dim=0,
            ).to(device=self.device, dtype=backbone_dtype)
            mm_extra["pixel_values"] = pv

        if "pixel_values" in mm_extra and isinstance(mm_extra["pixel_values"], torch.Tensor):
            self.last_pixel_values = mm_extra["pixel_values"]
            self.pixel_values = mm_extra["pixel_values"]

        if "input_ids" in tokenized:
            return tokenized, mm_extra
        return None, mm_extra

    def _run_backbone_prefill(self, model_inputs: dict[str, Any]) -> Any:
        prefill_kwargs = dict(use_cache=True, output_hidden_states=True, return_dict=True)
        try:
            return self.backbone(**model_inputs, logits_to_keep=1, **prefill_kwargs)
        except TypeError:
            try:
                return self.backbone(**model_inputs, **prefill_kwargs)
            except TypeError:
                inputs_no_pos = {k: v for k, v in model_inputs.items() if k != "position_ids"}
                try:
                    return self.backbone(**inputs_no_pos, logits_to_keep=1, **prefill_kwargs)
                except TypeError:
                    try:
                        return self.backbone(**inputs_no_pos, **prefill_kwargs)
                    except TypeError:
                        text_only = {
                            k: v
                            for k, v in inputs_no_pos.items()
                            if k in ("input_ids", "attention_mask", "inputs_embeds")
                        }
                        return self.backbone(**text_only, **prefill_kwargs)

    @torch.no_grad()
    def _init_generation(self) -> bool:
        if self.B <= 0 or not self.questions:
            return False

        M = len(self.questions)
        has_images = any(img is not None for img in self.question_images)

        # Fast path: single question, text-only (prefill at batch 1, then expand_cache_batch to B)
        if M == 1 and not has_images:
            prompt = self._build_prompt()
            self.initial_prompts = [prompt]
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            prompt_ids = inputs["input_ids"][0].tolist()
            self.prompt_len = len(prompt_ids)
            self.prompt_lens = [self.prompt_len] * self.B
            self.prompt_ends_in_think = prompt.rstrip("\n").endswith(THINK_OPEN)
            if not self._prompt_length_ok(len(prompt_ids)):
                return False

            t0 = time.time()
            outputs = self._run_backbone_prefill(dict(inputs))
            self.llm_time += time.time() - t0
            self.past_kv = outputs.past_key_values
            expand_cache_batch(self.past_kv, self.B)
            self.next_logits = outputs.logits[:, -1, :].repeat(self.B, 1)
            self.last_hidden = outputs.hidden_states[-1][:, -1, :].repeat(self.B, 1)
            self.global_len = len(prompt_ids)
            self.attn_mask = None
            self._ensure_attn_mask_capacity(self.global_len + self.max_response_tokens + 64)
            self._attn_mask_buf[:, :self.global_len] = 1
            self.attn_mask = self._attn_mask_buf[:, :self.global_len]
            self.positions = torch.full((self.B,), self.global_len, dtype=torch.long, device=self.device)
            for row in self.rows:
                row.kv_ledger = list(prompt_ids)
                row.prompt_len = self.prompt_len
                row.prompt_ends_in_think = self.prompt_ends_in_think
                row.in_think = self.prompt_ends_in_think
                row.mode = DECODE
            return True

        # Multi-question (M >= 1) and/or Multimodal prefill path with left-padding
        prompts = [
            self._build_prompt_for_question(q, img)
            for q, img in zip(self.questions, self.question_images)
        ]
        self.initial_prompts = prompts
        self.initial_prompt = prompts[0]
        ends_in_think_per_q = [p.rstrip("\n").endswith(THINK_OPEN) for p in prompts]

        proc_tokenized, mm_extra = self._extract_multimodal_kwargs(prompts)
        prompt_ids_per_q = [
            self.tokenizer(p, add_special_tokens=False)["input_ids"]
            for p in prompts
        ]
        # Ensure each prompt_ids_per_q entry is a flat list[int]
        normalized_ids_per_q: list[list[int]] = []
        for ids in prompt_ids_per_q:
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist() if ids.ndim > 1 else ids.tolist()
            elif isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            normalized_ids_per_q.append([int(x) for x in ids])
        prompt_ids_per_q = normalized_ids_per_q

        if proc_tokenized is not None and "input_ids" in proc_tokenized and M == 1:
            input_ids = proc_tokenized["input_ids"].to(device=self.device, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            L_max = int(input_ids.shape[1])
            if "attention_mask" in proc_tokenized:
                init_attn_mask = proc_tokenized["attention_mask"].to(dtype=torch.long, device=self.device)
                if init_attn_mask.ndim == 1:
                    init_attn_mask = init_attn_mask.unsqueeze(0)
            else:
                init_attn_mask = torch.ones((1, L_max), dtype=torch.long, device=self.device)
            prompt_ids_per_q = [input_ids[0].tolist()]
        else:
            L_max = max((len(ids) for ids in prompt_ids_per_q), default=0)
            if L_max <= 0:
                return False
            padded_ids: list[list[int]] = []
            mask_rows: list[list[int]] = []
            for ids in prompt_ids_per_q:
                pad_len = L_max - len(ids)
                padded_ids.append([self.pad_id] * pad_len + list(ids))
                mask_rows.append([0] * pad_len + [1] * len(ids))
            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            init_attn_mask = torch.tensor(mask_rows, dtype=torch.long, device=self.device)

        prompt_lens_per_q = [len(ids) for ids in prompt_ids_per_q]
        self.prompt_len = prompt_lens_per_q[0]
        self.prompt_ends_in_think = ends_in_think_per_q[0]
        for p_len in prompt_lens_per_q:
            if not self._prompt_length_ok(p_len):
                return False

        position_ids = (init_attn_mask.cumsum(dim=-1) - 1).clamp_min(0)
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": init_attn_mask,
            "position_ids": position_ids,
            **mm_extra,
        }
        t0 = time.time()
        outputs = self._run_backbone_prefill(model_inputs)
        self.llm_time += time.time() - t0

        self.past_kv = outputs.past_key_values
        logits_last = outputs.logits[:, -1, :]
        hidden_last = outputs.hidden_states[-1][:, -1, :]

        if self.samples_per_question > 1:
            expand_cache_batch(self.past_kv, self.B)
            logits_last = logits_last.repeat_interleave(self.samples_per_question, dim=0)
            hidden_last = hidden_last.repeat_interleave(self.samples_per_question, dim=0)
            init_attn_mask = init_attn_mask.repeat_interleave(self.samples_per_question, dim=0)

        self.next_logits = logits_last
        self.last_hidden = hidden_last
        self.global_len = int(init_attn_mask.shape[1])

        self.attn_mask = None
        self._ensure_attn_mask_capacity(self.global_len + self.max_response_tokens + 64)
        self._attn_mask_buf[:, :self.global_len] = init_attn_mask
        self.attn_mask = self._attn_mask_buf[:, :self.global_len]

        pos_list: list[int] = []
        for i, row in enumerate(self.rows):
            q_idx = i // self.samples_per_question
            q_ids = prompt_ids_per_q[q_idx]
            q_len = len(q_ids)
            pos_list.append(q_len)
            row.kv_ledger = list(q_ids)
            row.prompt_len = q_len
            row.prompt_ends_in_think = ends_in_think_per_q[q_idx]
            row.in_think = ends_in_think_per_q[q_idx]
            row.mode = DECODE

        self.prompt_lens = list(pos_list)
        self.positions = torch.tensor(pos_list, dtype=torch.long, device=self.device)
        return True

    def _prompt_length_ok(self, n: int) -> bool:  # HybridRL overrides
        return True

    # ------------------------------------------------------------------
    # Lockstep stepping & Pre-allocated attention mask management
    # ------------------------------------------------------------------

    def _row_budget_left(self, row: RowState) -> int:
        if self.count_tool_tokens:
            base_prompt_len = row.prompt_len if row.prompt_len > 0 else self.prompt_len
            return self.max_response_tokens - (row.true_len - base_prompt_len)
        return self.max_response_tokens - row.generation_steps

    @torch.no_grad()
    def _lockstep_forward(self, inputs_embeds: torch.Tensor, real_rows: list[int]):
        """One batched forward; only `real_rows` slots carry real content, the other
        slots are masked out of attention forever and do not advance their row's position."""
        new_len = self.global_len + 1
        self._ensure_attn_mask_capacity(new_len)
        self._attn_mask_buf[:, self.global_len] = 0
        idx = None
        if real_rows:
            idx = torch.as_tensor(real_rows, dtype=torch.long, device=self.device)
            self._attn_mask_buf[idx, self.global_len] = 1
        self.attn_mask = self._attn_mask_buf[:, :new_len]

        position_ids = self.positions.unsqueeze(1)
        cache_position = torch.tensor([self.global_len], dtype=torch.long, device=self.device)
        t0 = time.time()
        self.past_kv, step_logits, step_hidden = forward_step(
            self.backbone,
            input_embeds=inputs_embeds,
            past_key_values=self.past_kv,
            attention_mask=self.attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        self.llm_time += time.time() - t0
        self.global_len = new_len
        if idx is not None:
            self.positions[idx] += 1
            if len(real_rows) == self.B or self.next_logits is None:
                self.next_logits = step_logits
                self.last_hidden = step_hidden
            else:
                self.next_logits[idx] = step_logits[idx]
                self.last_hidden[idx] = step_hidden[idx]

    @torch.no_grad()
    def _multi_token_feed_step(self) -> None:
        """Advance all active FEED rows by up to `_MAX_FEED_CHUNK` tokens in a single
        multi-token forward pass using right-aligned padding within the chunk so every
        FEED row's final fed token lands at column -1 while preserving non-FEED rows."""
        feed_rows = [r for r in self.rows if r.mode == FEED and r.feed_queue]
        if not feed_rows:
            for r in self.rows:
                if r.mode == FEED:
                    r.in_think = False
                    r.text_ids_since_resume = []
                    r.mode = DONE if r.turn_count > self.max_turns else DECODE
            return

        K = min(max(len(r.feed_queue) for r in feed_rows), _MAX_FEED_CHUNK)
        if K <= 1:
            self._decode_step(rng=None)
            return

        embed_w = self.embed_layer.weight
        embed_scale = getattr(self.embed_layer, "embed_scale", None)
        if embed_scale is not None:
            embed_scale = embed_scale.to(device=embed_w.device, dtype=embed_w.dtype)

        pad_vec = embed_w[self.pad_id]
        if embed_scale is not None:
            pad_vec = pad_vec * embed_scale

        inputs = pad_vec.view(1, 1, -1).expand(self.B, K, -1).clone()
        position_ids = self.positions.unsqueeze(1).expand(self.B, K).clone()

        new_len = self.global_len + K
        self._ensure_attn_mask_capacity(new_len)
        self._attn_mask_buf[:, self.global_len : new_len] = 0

        chunks_per_row: list[tuple[RowState, list[int]]] = []
        for row in feed_rows:
            i = row.idx
            chunk = row.feed_queue[:K]
            del row.feed_queue[:K]
            n_r = len(chunk)
            pad_r = K - n_r
            tok_tensor = torch.as_tensor(chunk, dtype=torch.long, device=self.device)
            vecs = embed_w.index_select(0, tok_tensor)
            if embed_scale is not None:
                vecs = vecs * embed_scale
            inputs[i, pad_r:K] = vecs
            self._attn_mask_buf[i, self.global_len + pad_r : new_len] = 1
            position_ids[i, pad_r:K] = self.positions[i] + torch.arange(
                n_r, dtype=torch.long, device=self.device
            )
            chunks_per_row.append((row, chunk))

        self.attn_mask = self._attn_mask_buf[:, :new_len]
        cache_position = torch.arange(self.global_len, new_len, dtype=torch.long, device=self.device)

        t0 = time.time()
        self.past_kv, step_logits, step_hidden = forward_step(
            self.backbone,
            input_embeds=inputs,
            past_key_values=self.past_kv,
            attention_mask=self.attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        self.llm_time += time.time() - t0
        self.global_len = new_len

        for row, chunk in chunks_per_row:
            i = row.idx
            self.positions[i] += len(chunk)
            if self.next_logits is not None:
                self.next_logits[i] = step_logits[i]
            if self.last_hidden is not None:
                self.last_hidden[i] = step_hidden[i]
            for tok in chunk:
                row.kv_ledger.append(tok)
                self._on_env_token_fed(row)
            if not row.feed_queue:
                row.in_think = False
                row.text_ids_since_resume = []
                if row.turn_count > self.max_turns:
                    row.end_reason = "max_turns"
                    row.mode = DONE
                else:
                    row.mode = DECODE

    # -- per-row hooks HybridRL overrides to record the trajectory ---------

    def _record_text_action(self, row: RowState, token_id: int,
                            log_probs_row: torch.Tensor, blocked_key: str) -> None:
        if self.record_token_logprobs:
            lst = row.extra.setdefault("token_logprobs", [])
            lst.append(float(log_probs_row[token_id].detach().item()))

    def _record_latent_vec(self, row: RowState, injected_row: torch.Tensor) -> None:
        if self.record_hidden:
            lst = row.extra.setdefault("latent_vectors", [])
            lst.append(injected_row.detach().cpu())

    def _on_env_token_fed(self, row: RowState) -> None:
        pass

    def _on_latent_span_opened(self, row: RowState) -> None:
        pass

    def _on_latent_span_closed(self, row: RowState) -> None:
        pass

    def _sequence_cap_ok(self, row: RowState) -> bool:
        return True

    def _env_tokens_fit(self, row: RowState, n_new: int) -> bool:
        return True

    def _row_invalidate(self, row: RowState, reason: str) -> None:
        row.end_reason = row.end_reason or f"invalid:{reason}"
        row.mode = DONE

    # ------------------------------------------------------------------

    def _flush_new_model_text(self, row: RowState) -> None:
        decoded = self.tokenizer.decode(row.generated_ids, skip_special_tokens=False)
        new_text = decoded[row._last_decoded_len:]
        row._last_decoded_len = len(decoded)
        if new_text:
            row.full_output += new_text

    def _row_tail(self, row: RowState) -> str:
        if len(row.text_ids_since_resume) < 2:
            return ""
        return self.tokenizer.decode(row.text_ids_since_resume[-50:], skip_special_tokens=False)

    def _check_tool_tags(self, tail: str) -> str | None:
        for stop in self.tool_stop_tags:
            if stop in tail:
                return stop
        return None

    def _latent_exit(self, row: RowState, readout_id: int) -> str | None:
        """Exit-anchor readout: the span closes when the readout of the
        current last-layer state is `</latent>` (after min_latent_steps), or at
        max_latent_steps."""
        if self.max_latent_steps > 0 and row.latent_steps >= self.max_latent_steps:
            return "max_steps"
        if row.latent_steps >= self.min_latent_steps and readout_id == self.latent_close_id:
            return "anchor"
        return None

    def _close_span_readout(self, row: RowState, exit_reason: str) -> None:
        if not self.record_latent_readout:
            row._open_span_vecs.clear()
            row.readout_ids.clear()
            return
        if row._open_span_vecs and not row.readout_ids:
            span_vecs = torch.stack(row._open_span_vecs, dim=0)
            row.readout_ids = [
                int(t) for t in nearest_token_ids(span_vecs, self._get_readout_bank()).tolist()
            ]
        row._open_span_vecs.clear()
        row.latent_spans.append(LatentSpanReadout(
            steps=len(row.readout_ids), exit=exit_reason,
            token_ids=list(row.readout_ids), pseudo_text=pseudo_text(row.readout_ids, self.tokenizer),
        ))
        row.readout_ids = []

    # ------------------------------------------------------------------
    # One global step over all rows
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_step(self, rng: torch.Generator | None) -> None:
        """Advance every DECODE/LATENT/FEED row by one slot (or multi-token FEED chunk); IDLE/DONE rows get PAD."""
        if any(r.mode == FEED and len(r.feed_queue) > 1 for r in self.rows):
            self._multi_token_feed_step()
            return

        embed_w = self.embed_layer.weight
        embed_scale = getattr(self.embed_layer, "embed_scale", None)
        if embed_scale is not None:
            embed_scale = embed_scale.to(device=embed_w.device, dtype=embed_w.dtype)

        def _tok_embed(tok_id: int) -> torch.Tensor:
            v = embed_w[tok_id]
            return v * embed_scale if embed_scale is not None else v

        inputs = torch.empty((self.B, 1, embed_w.shape[1]), dtype=embed_w.dtype, device=self.device)
        pad_embed = _tok_embed(self.pad_id)

        sample_rows_list: list[int] = []
        blocked_keys: dict[int, str] = {}
        real_rows: list[int] = []
        forced_tokens: dict[int, int] = {}
        latent_vec_rows: dict[int, torch.Tensor] = {}

        # Latent rows: next inputs R(h) from the current last-layer states and
        # the token readout of those states for the exit decision (single .tolist() transfer).
        latent_rows = [r.idx for r in self.rows if r.mode == LATENT]
        latent_pos_map = {r_idx: pos for pos, r_idx in enumerate(latent_rows)}
        injected_all = None
        latent_readout_list: list[int] = []
        if latent_rows:
            idx = torch.as_tensor(latent_rows, dtype=torch.long, device=self.device)
            injected_all = self.model.run_hybridlink(
                self.last_hidden.index_select(0, idx), output_dtype=embed_w.dtype
            )
            latent_readout_list = self.next_logits.index_select(0, idx).argmax(dim=-1).tolist()
            if isinstance(latent_readout_list, int):
                latent_readout_list = [latent_readout_list]

        for row in self.rows:
            i = row.idx
            if row.mode in (IDLE, DONE):
                inputs[i, 0] = pad_embed
                continue

            # budget gate for generating rows (observations are fed in full)
            if row.mode != FEED:
                if self._row_budget_left(row) <= 0:
                    row.extra["budget_truncated"] = True
                    row.end_reason = row.end_reason or "budget"
                    row.mode = DONE
                    inputs[i, 0] = pad_embed
                    continue
                if not self._sequence_cap_ok(row):
                    inputs[i, 0] = pad_embed
                    continue

            if row.mode == LATENT:
                row.generation_steps += 1
                pos_in_batch = latent_pos_map[i]
                injected = injected_all[pos_in_batch]
                exit_reason = self._latent_exit(row, int(latent_readout_list[pos_in_batch]))
                if exit_reason is not None:
                    row.generated_ids.append(self.latent_close_id)
                    row.kv_ledger.append(self.latent_close_id)
                    forced_tokens[i] = self.latent_close_id
                    real_rows.append(i)
                    inputs[i, 0] = _tok_embed(self.latent_close_id)
                    if exit_reason == "max_steps":
                        row.forced_exits += 1
                    else:
                        row.anchor_exits += 1
                    self._close_span_readout(row, exit_reason)
                    self._on_latent_span_closed(row)
                    row.mode = DECODE
                    row.latent_steps = 0
                else:
                    inputs[i, 0] = injected
                    real_rows.append(i)
                    latent_vec_rows[i] = injected
                    row.kv_ledger.append(None)
                    row.latent_steps += 1
                    row.latent_steps_total += 1
                    row.turn_latent_steps[-1] += 1
                    if self.record_latent_readout:
                        row._open_span_vecs.append(injected.detach())
                continue

            if row.mode == FEED:
                tok = row.feed_queue.pop(0)
                inputs[i, 0] = _tok_embed(tok)
                row.kv_ledger.append(tok)
                self._on_env_token_fed(row)
                forced_tokens[i] = tok
                real_rows.append(i)
                if not row.feed_queue:
                    # observation fed: next turn starts with the model's own <think>
                    row.in_think = False
                    row.text_ids_since_resume = []
                    if row.turn_count > self.max_turns:
                        row.end_reason = "max_turns"
                        row.mode = DONE
                    else:
                        row.mode = DECODE
                continue

            # DECODE: sample from this row's current logits
            row.generation_steps += 1
            sample_rows_list.append(i)
            blocked_keys[i] = "in" if row.in_think else "out"

        if sample_rows_list:
            sample_idx = torch.as_tensor(sample_rows_list, dtype=torch.long, device=self.device)
            sub_logits = self.next_logits.index_select(0, sample_idx)
            sub_blocked = torch.stack(
                [self._blocked_masks[blocked_keys[i]] for i in sample_rows_list], dim=0
            )
            if self.repetition_penalty != 1.0:
                sub_logits = sub_logits.clone()
                for pos, i in enumerate(sample_rows_list):
                    seen = {t for t in self.rows[i].kv_ledger if t is not None}
                    if not seen:
                        continue
                    seen_idx = torch.tensor(sorted(seen), dtype=torch.long, device=sub_logits.device)
                    vals = sub_logits[pos, seen_idx]
                    sub_logits[pos, seen_idx] = torch.where(
                        vals > 0, vals / self.repetition_penalty, vals * self.repetition_penalty,
                    )
            sub_log_probs = token_log_probs(
                sub_logits, self.temperature, self.top_p, self.top_k, sub_blocked,
            )
            sampled = sample_rows(
                sub_log_probs,
                list(range(len(sample_rows_list))),
                rng,
                temperature=self.temperature,
            )
            for pos, i in enumerate(sample_rows_list):
                tok = sampled[pos]
                row = self.rows[i]
                row.generated_ids.append(tok)
                row.text_ids_since_resume.append(tok)
                row.kv_ledger.append(tok)
                self._record_text_action(row, tok, sub_log_probs[pos], blocked_keys[i])
                inputs[i, 0] = _tok_embed(tok)
                forced_tokens[i] = tok
                real_rows.append(i)

        for i, vec in latent_vec_rows.items():
            self._record_latent_vec(self.rows[i], vec)

        if not real_rows:
            return

        self._lockstep_forward(inputs, sorted(set(real_rows)))

        # events for rows that sampled a text token this step
        for i in sample_rows_list:
            row = self.rows[i]
            tok = row.generated_ids[-1]

            if tok in self.eos_ids:
                row.end_reason = "eos"
                row.mode = DONE
                continue
            if tok == self.think_open_id:
                row.in_think = True
                continue
            elif tok == self.think_close_id:
                row.in_think = False
                continue
            elif tok == self.latent_open_id:
                row.mode = LATENT
                row.latent_steps = 0
                row.latent_spans_opened += 1
                self._on_latent_span_opened(row)
                continue

            if not self._token_may_close_tag(tok):
                continue

            tail = self._row_tail(row)
            if self.reject_fabricated_result and RESULT_OPEN in tail:
                self._row_invalidate(row, "fabricated_result")
                continue

            stop = self._check_tool_tags(tail)
            if stop:
                tag = stop[2:-1]
                decoded = self.tokenizer.decode(row.generated_ids, skip_special_tokens=False)
                content = ToolExecutor.extract_content(decoded, tag)
                self._on_tool_request(row, tag, content)

    # ------------------------------------------------------------------
    # Tool barrier
    # ------------------------------------------------------------------

    def _on_tool_request(self, row: RowState, tag: str, content: str) -> None:
        """A row emitted a tool call: apply the caps, then pause it until the barrier."""
        if self.terminate_on_tool_cap:
            cap = None
            if self.max_tool_calls is not None and row.tool_requests >= self.max_tool_calls:
                cap = "shared"
            elif tag == "python" and self.max_python_calls is not None and row.python_calls >= self.max_python_calls:
                cap = "python"
            elif tag == "search" and self.max_search_calls is not None and row.search_calls >= self.max_search_calls:
                cap = "search"
            if cap is not None:
                row.end_reason = "tool_cap"
                row.mode = DONE
                return
            row.tool_requests += 1
        row.pending_tool = (tag, content)
        row.mode = IDLE

    def _execute_tool(self, row: RowState) -> str:
        """Run one row's pending tool call; cap notices are returned as observations."""
        if self.tool_executor is None:
            from .tools import build_tool_executor

            self.tool_executor = build_tool_executor(require_search=False)
        tag, content = row.pending_tool
        if (self.max_tool_calls is not None
                and (row.python_calls + row.search_calls) >= self.max_tool_calls):
            return ("<result>The maximum tool call limit is exceeded. "
                    "You are not allowed to call any more tools.</result>")
        if tag == "python":
            if self.max_python_calls is None or row.python_calls < self.max_python_calls:
                row.python_calls += 1
                result = self.tool_executor.execute("python", content)
            else:
                result = ("The maximum python call limit is exceeded. "
                          "You are not allowed to use python.")
        elif tag == "search":
            if self.max_search_calls is None or row.search_calls < self.max_search_calls:
                if content not in row.search_query_history:
                    row.search_query_history.add(content)
                    row.search_calls += 1
                    result = self.tool_executor.execute("search", content)
                else:
                    result = ("You have searched this query. "
                              "Please refer to previous results.")
            else:
                result = ("The maximum search limit is exceeded. "
                          "You are not allowed to search.")
        else:
            result = ""
        return f"<result>{result}</result>" if result else "<result></result>"

    @torch.no_grad()
    def _tool_barrier(self) -> None:
        """Execute all pending tool calls concurrently; queue their observations."""
        pending = [r for r in self.rows if r.mode == IDLE and r.pending_tool]
        if not pending:
            return
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, self.tool_workers)) as pool:
            results = list(pool.map(self._execute_tool, pending))
        self.tool_time += time.time() - t0
        for row, result_text in zip(pending, results):
            row.pending_tool = None
            self._flush_new_model_text(row)
            row.full_output += result_text
            row.turn_texts.append(row.full_output[row._turn_start_char:])
            row._turn_start_char = len(row.full_output)
            token_ids = self.tokenizer(result_text, add_special_tokens=False)["input_ids"]
            if not self._env_tokens_fit(row, len(token_ids)):
                continue
            row.feed_queue = list(token_ids)
            row.turn_count += 1
            row.turn_latent_steps.append(0)
            if token_ids:
                row.mode = FEED
            else:
                row.in_think = False
                row.text_ids_since_resume = []
                if row.turn_count > self.max_turns:
                    row.end_reason = "max_turns"
                    row.mode = DONE
                else:
                    row.mode = DECODE

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, rng: torch.Generator | None = None) -> None:
        start = time.time()
        if not self._init_generation():
            self.total_time = time.time() - start
            return
        if self.max_turns <= 0:
            for row in self.rows:
                row.end_reason = "max_turns"
                row.mode = DONE
        while True:
            active = [r for r in self.rows if r.mode in (DECODE, LATENT, FEED)]
            if not active:
                pending = [r for r in self.rows if r.mode == IDLE]
                if not pending:
                    break
                self._tool_barrier()
                continue
            if any(r.mode == FEED and len(r.feed_queue) > 1 for r in active):
                self._multi_token_feed_step()
                continue
            self._decode_step(rng)
        for row in self.rows:
            self._flush_new_model_text(row)
            if row._turn_start_char < len(row.full_output) or not row.turn_texts:
                row.turn_texts.append(row.full_output[row._turn_start_char:])
                row._turn_start_char = len(row.full_output)
            if row.readout_ids or row._open_span_vecs:
                self._close_span_readout(row, row.end_reason or "budget")
        self.total_time = time.time() - start

    def row_result(self, row: RowState) -> dict[str, Any]:
        out: dict[str, Any] = {
            "output": row.full_output,
            "question": row.question,
            "question_idx": row.question_idx,
            "sample_idx": row.sample_idx,
            "turns": row.turn_count,
            "rounds": [
                {"text": txt, "latent_steps": steps}
                for txt, steps in zip(row.turn_texts, row.turn_latent_steps)
            ],
            "python_calls": row.python_calls,
            "search_calls": row.search_calls,
            "tool_requests": row.tool_requests,
            "generation_steps": row.generation_steps,
            "latent_steps": row.latent_steps_total,
            "latent_spans": row.latent_spans_opened,
            "anchor_exits": row.anchor_exits,
            "forced_exits": row.forced_exits,
            "end_reason": row.end_reason,
        }
        if self.record_latent_readout:
            out["latent_readouts"] = [span.to_dict() for span in row.latent_spans]
            out["output_rendered"] = render_with_readout(row.full_output, row.latent_spans)
        if self.record_token_logprobs and "token_logprobs" in row.extra:
            out["token_logprobs"] = list(row.extra["token_logprobs"])
        if self.record_hidden and "latent_vectors" in row.extra:
            out["latent_vectors"] = list(row.extra["latent_vectors"])
        return out


@dataclass
class HybridRound:
    """Single-turn summary returned by the high-level HybridEngine wrapper."""
    text: str
    latent_steps: int = 0


@dataclass
class HybridTrajectory:
    """High-level trajectory returned by HybridEngine.run() and HybridEngine.run_batch()."""
    rounds: list[HybridRound]
    final_answer: str
    output: str = ""
    output_rendered: str = ""
    latent_steps: int = 0
    latent_spans: int = 0
    raw_result: dict[str, Any] = field(default_factory=dict)

    @property
    def n_latent_steps(self) -> int:
        return self.latent_steps

    @property
    def n_latent_spans(self) -> int:
        return self.latent_spans

    @property
    def turns(self) -> int:
        return len(self.rounds)


def _parse_rounds_from_result(res: dict[str, Any]) -> list[HybridRound]:
    """Build per-turn HybridRound objects from row_result(), falling back to output parsing
    when a caller provides a synthetic result dict without pre-recorded `rounds`."""
    import re

    recorded = res.get("rounds")
    if isinstance(recorded, list) and recorded:
        return [
            HybridRound(text=str(r.get("text", "")), latent_steps=int(r.get("latent_steps", 0)))
            for r in recorded
        ]
    full_out = res.get("output", "")
    tokens = re.split(r"(<result>.*?</result>)", full_out, flags=re.DOTALL)
    raw_chunks: list[tuple[str, str]] = []
    idx = 0
    while idx < len(tokens):
        assistant_part = tokens[idx]
        result_part = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        chunk = assistant_part + result_part
        if chunk.strip() or not raw_chunks:
            raw_chunks.append((chunk, assistant_part))
        idx += 2

    closed_marker = f"{ENTRY_ANCHOR}{EXIT_ANCHOR}"
    non_think_re = re.compile(
        r"<(?:python|search|result|answer)>.*?(?:</(?:python|search|result|answer)>|$)",
        flags=re.DOTALL,
    )
    span_counts: list[int] = []
    for _, assistant_part in raw_chunks:
        think_only = non_think_re.sub("", assistant_part)
        n_closed = think_only.count(closed_marker)
        n_unclosed = think_only.replace(closed_marker, "").count(ENTRY_ANCHOR)
        span_counts.append(n_closed + n_unclosed)

    readouts = list(res.get("latent_readouts") or [])
    total_steps = int(res.get("latent_steps", 0))
    total_spans = sum(span_counts)
    span_cursor = 0
    rounds: list[HybridRound] = []
    for (chunk, _), n_spans in zip(raw_chunks, span_counts):
        if readouts:
            round_steps = sum(
                int(s.get("steps", 0)) for s in readouts[span_cursor : span_cursor + n_spans]
            )
            span_cursor += n_spans
        elif total_spans > 0:
            round_steps = (total_steps * n_spans) // total_spans
        else:
            round_steps = total_steps if not rounds else 0
        rounds.append(HybridRound(text=chunk, latent_steps=round_steps))
    return rounds


class HybridEngine:
    """High-level wrapper around HybridGenerationEngine supporting both single-question
    `run()` and high-concurrency multi-question `run_batch()` execution."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tool_executor: ToolExecutor | None = None,
        *,
        processor: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 4096,
        max_latent_steps: int = 32,
        min_latent_steps: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_rounds: int = 6,
        max_tool_calls: int | None = 6,
        max_python_calls: int | None = 6,
        max_search_calls: int | None = 6,
        record_latent_readout: bool = True,
        default_batch_size: int = 32,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        max_turns: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tool_executor = tool_executor
        self.processor = processor if processor is not None else getattr(model, "processor", None)
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.min_latent_steps = int(min_latent_steps)
        self.max_latent_steps = int(max_latent_steps)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.max_rounds = int(max_turns if max_turns is not None else max_rounds)
        self.max_turns = self.max_rounds
        self.max_tool_calls = max_tool_calls
        self.max_python_calls = max_python_calls
        self.max_search_calls = max_search_calls
        self.record_latent_readout = bool(record_latent_readout)
        self.default_batch_size = max(1, int(default_batch_size))

    @staticmethod
    def _row_to_trajectory(gen_engine: HybridGenerationEngine, row: RowState) -> HybridTrajectory:
        from .answer import extract_answer

        res = gen_engine.row_result(row)
        full_out = res.get("output", "")
        ans = extract_answer(full_out) or ""
        rounds = _parse_rounds_from_result(res)
        return HybridTrajectory(
            rounds=rounds,
            final_answer=ans,
            output=full_out,
            output_rendered=res.get("output_rendered", full_out),
            latent_steps=res.get("latent_steps", 0),
            latent_spans=res.get("latent_spans", 0),
            raw_result=res,
        )

    def _build_gen_engine_kwargs(
        self,
        *,
        system_prompt: str | None = None,
        record_latent_readout: bool | None = None,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        from .tools import build_tool_executor

        try:
            device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

        tools = overrides.get("tool_executor", self.tool_executor)
        if tools is None:
            tools = build_tool_executor(require_search=False)

        sys_prompt = (
            system_prompt
            if system_prompt is not None
            else overrides.get("system_prompt", self.system_prompt)
        )
        max_resp = int(
            overrides.get("max_response_tokens", overrides.get("max_new_tokens", self.max_new_tokens))
        )
        max_turns_val = int(
            overrides.get("max_turns", overrides.get("max_rounds", self.max_turns))
        )
        rec_readout = (
            self.record_latent_readout
            if record_latent_readout is None
            else bool(record_latent_readout)
        )
        if "record_latent_readout" in overrides and record_latent_readout is None:
            rec_readout = bool(overrides["record_latent_readout"])

        return dict(
            device=overrides.get("device", device),
            tool_executor=tools,
            processor=overrides.get("processor", self.processor),
            system_prompt=sys_prompt,
            batch_size=int(overrides.get("samples_per_question", 1)),
            max_response_tokens=max_resp,
            count_tool_tokens=bool(overrides.get("count_tool_tokens", False)),
            max_tool_calls=overrides.get("max_tool_calls", self.max_tool_calls),
            max_search_calls=overrides.get("max_search_calls", self.max_search_calls),
            max_python_calls=overrides.get("max_python_calls", self.max_python_calls),
            terminate_on_tool_cap=bool(overrides.get("terminate_on_tool_cap", False)),
            max_turns=max_turns_val,
            temperature=float(overrides.get("temperature", self.temperature)),
            top_p=float(overrides.get("top_p", self.top_p)),
            top_k=int(overrides.get("top_k", self.top_k)),
            repetition_penalty=float(overrides.get("repetition_penalty", self.repetition_penalty)),
            min_latent_steps=int(overrides.get("min_latent_steps", self.min_latent_steps)),
            max_latent_steps=int(overrides.get("max_latent_steps", self.max_latent_steps)),
            think_guards=bool(overrides.get("think_guards", True)),
            reject_fabricated_result=bool(overrides.get("reject_fabricated_result", True)),
            record_latent_readout=rec_readout,
            record_hidden=bool(overrides.get("record_hidden", False)),
            record_token_logprobs=bool(overrides.get("record_token_logprobs", False)),
            tool_workers=int(overrides.get("tool_workers", 8)),
        )

    def run(
        self,
        question: str,
        system_prompt: str | None = None,
        seed: int | None = 0,
        *,
        images: Any | None = None,
        **overrides: Any,
    ) -> HybridTrajectory:
        engine_kwargs = self._build_gen_engine_kwargs(
            system_prompt=system_prompt,
            record_latent_readout=overrides.get("record_latent_readout"),
            overrides=overrides,
        )
        gen_engine = HybridGenerationEngine(
            self.model,
            self.tokenizer,
            question,
            images=images,
            **engine_kwargs,
        )
        rng = None
        if seed is not None:
            rng = torch.Generator(device=gen_engine.device).manual_seed(int(seed))
        gen_engine.run(rng)
        return self._row_to_trajectory(gen_engine, gen_engine.rows[0])

    def run_batch(
        self,
        questions: Sequence[str],
        *,
        images: Sequence[Any] | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        record_latent_readout: bool | None = None,
        system_prompt: str | None = None,
        seed: int | None = 0,
        **overrides: Any,
    ) -> list[HybridTrajectory]:
        """Executes a list of distinct questions (and optional multimodal images) in high-concurrency
        batches of size `batch_size`, preserving input order in the returned `list[HybridTrajectory]`."""
        if isinstance(questions, str):
            q_list = [questions]
        else:
            q_list = [str(q) for q in questions]
        n_total = len(q_list)
        if n_total == 0:
            return []

        if images is not None:
            if not isinstance(images, (list, tuple)) or len(images) != n_total:
                if n_total == 1:
                    img_list: list[Any] = [images]
                else:
                    raise ValueError(
                        f"Expected `images` length {n_total} to match `questions`, "
                        f"got {len(images) if isinstance(images, (list, tuple)) else 1}."
                    )
            else:
                img_list = list(images)
        else:
            img_list = [None] * n_total

        if batch_size is not None and max_concurrency is not None:
            eff_batch_size = min(int(batch_size), int(max_concurrency))
        elif batch_size is not None:
            eff_batch_size = int(batch_size)
        elif max_concurrency is not None:
            eff_batch_size = int(max_concurrency)
        else:
            eff_batch_size = self.default_batch_size
        eff_batch_size = max(1, eff_batch_size)

        engine_kwargs = self._build_gen_engine_kwargs(
            system_prompt=system_prompt,
            record_latent_readout=record_latent_readout,
            overrides=overrides,
        )

        sort_by_length = bool(overrides.get("sort_by_length", False)) and n_total > eff_batch_size
        if sort_by_length:
            order = sorted(range(n_total), key=lambda idx: len(q_list[idx]))
        else:
            order = list(range(n_total))

        trajectories: list[HybridTrajectory | None] = [None] * n_total
        samples_per_q = int(engine_kwargs.get("batch_size", 1))

        for chunk_idx, start in enumerate(range(0, n_total, eff_batch_size)):
            chunk_indices = order[start : start + eff_batch_size]
            chunk_questions = [q_list[i] for i in chunk_indices]
            chunk_images = [img_list[i] for i in chunk_indices] if images is not None else None

            gen_engine = HybridGenerationEngine(
                self.model,
                self.tokenizer,
                chunk_questions,
                images=chunk_images,
                **engine_kwargs,
            )
            rng = None
            if seed is not None:
                rng = torch.Generator(device=gen_engine.device).manual_seed(int(seed) + chunk_idx)
            gen_engine.run(rng)

            for local_q_idx, orig_idx in enumerate(chunk_indices):
                row_idx = local_q_idx * samples_per_q
                trajectories[orig_idx] = self._row_to_trajectory(gen_engine, gen_engine.rows[row_idx])

        return [t for t in trajectories if t is not None]
