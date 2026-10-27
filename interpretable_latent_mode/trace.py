"""Hybrid traces for Interpretable Latent Mode: run a question through HybridAgent while
recording every latent input, map each latent step to its nearest token embeddings, and
split the trajectory into text, latent, tool-call, observation and answer segments so the
latent thoughts can be read alongside everything the agent did in text.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import torch

from hybrid_agent import SYSTEM_PROMPT, HybridGenerationEngine, extract_answer
from hybrid_agent.engine import RowState
from hybrid_agent.readout import latent_readout

_SEGMENT_RE = re.compile(
    r"(<latent></latent>|<search>.*?</search>|<python>.*?</python>|<result>.*?</result>|<answer>.*?</answer>)",
    re.DOTALL,
)

_EOS_RE = re.compile(r"(?:<\|endoftext\|>|<\|im_end\|>|<turn\|>|<end_of_turn>|<eos>)\s*$")

GENERATION_DEFAULTS = dict(
    max_response_tokens=4096,
    max_tool_calls=5,
    max_search_calls=5,
    max_python_calls=5,
    terminate_on_tool_cap=True,
    max_turns=30,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.1,
    min_latent_steps=4,
    max_latent_steps=200,
)


@dataclass
class LatentStep:
    index: int
    token: str
    token_id: int
    cosine: float
    candidates: list[tuple[str, int, float]] = field(default_factory=list)


@dataclass
class Segment:
    kind: str                      # text | latent | tool_call | tool_result | answer
    text: str = ""
    tool: str | None = None
    steps: list[LatentStep] = field(default_factory=list)
    exit: str = ""
    pseudo_text: str = ""


@dataclass
class HybridTrace:
    question: str
    segments: list[Segment]
    prediction: str = ""
    stats: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    model: str = ""
    sample_idx: int = 0

    @property
    def latent_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.kind == "latent"]

    def to_dict(self) -> dict:
        return asdict(self)


class LatentRecordingEngine(HybridGenerationEngine):
    """The generation engine plus a record of every latent input, grouped by span."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, record_latent_readout=True, **kwargs)
        self.spans: list[list[list[torch.Tensor]]] = [[] for _ in range(self.B)]

    def _on_latent_span_opened(self, row: RowState) -> None:
        self.spans[row.idx].append([])

    def _record_latent_vec(self, row: RowState, injected_row: torch.Tensor) -> None:
        self.spans[row.idx][-1].append(injected_row.detach().reshape(-1).clone())


def parse_segments(output: str, spans: list[Segment] | None = None) -> list[Segment]:
    """Split a decoded output into segments; the k-th `<latent></latent>` marker takes the
    k-th recorded latent span (or an empty one)."""
    segments: list[Segment] = []
    latent_index = 0
    output = _EOS_RE.sub("", output)
    for part in _SEGMENT_RE.split(output):
        if not part:
            continue
        if part == "<latent></latent>":
            if spans is not None and latent_index < len(spans):
                segments.append(spans[latent_index])
            else:
                segments.append(Segment(kind="latent"))
            latent_index += 1
        elif part.startswith("<result>"):
            segments.append(Segment(kind="tool_result", text=part[len("<result>"):-len("</result>")]))
        elif part.startswith("<answer>"):
            segments.append(Segment(kind="answer", text=part[len("<answer>"):-len("</answer>")]))
        elif part.startswith("<search>") or part.startswith("<python>"):
            tool = "search" if part.startswith("<search>") else "python"
            segments.append(Segment(kind="tool_call", tool=tool, text=part[len(tool) + 2:-(len(tool) + 3)]))
        else:
            segments.append(Segment(kind="text", text=part))
    return segments


def _stats(result: dict) -> dict:
    return {
        "latent_spans": result["latent_spans"],
        "latent_steps": result["latent_steps"],
        "text_tokens": result["generation_steps"] - result["latent_steps"],
        "search_calls": result["search_calls"],
        "python_calls": result["python_calls"],
        "turns": result["turns"],
        "end_reason": result["end_reason"],
    }


def inspect(
    model,
    tokenizer,
    question: str,
    *,
    device,
    tool_executor,
    readout_top_k: int = 5,
    num_samples: int = 1,
    seed: int = 0,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "",
    **generation_kwargs,
) -> list[HybridTrace]:
    """Generate `num_samples` answers and return one annotated trace per sample."""
    settings = {**GENERATION_DEFAULTS, **generation_kwargs}
    engine = LatentRecordingEngine(
        model, tokenizer, device, tool_executor, system_prompt, question, batch_size=num_samples, **settings,
    )
    engine.run(torch.Generator(device=device).manual_seed(seed))
    embed_weight = model.backbone.get_input_embeddings().weight
    traces = []
    for row in engine.rows:
        result = engine.row_result(row)
        spans: list[Segment] = []
        for span_vectors, readout in zip(engine.spans[row.idx], row.latent_spans):
            steps: list[LatentStep] = []
            if span_vectors:
                candidates = latent_readout(torch.stack(span_vectors), embed_weight, tokenizer, top_k=readout_top_k)
                for i, cands in enumerate(candidates):
                    token, token_id, cosine = cands[0]
                    steps.append(LatentStep(index=i, token=token, token_id=token_id, cosine=cosine, candidates=cands))
            spans.append(Segment(kind="latent", steps=steps, exit=readout.exit, pseudo_text=readout.pseudo_text))
        traces.append(HybridTrace(
            question=question,
            segments=parse_segments(result["output"], spans),
            prediction=extract_answer(result["output"]),
            stats=_stats(result),
            settings=settings,
            model=model_name,
            sample_idx=row.idx,
        ))
    return traces


def trace_from_record(record: dict) -> HybridTrace:
    """A trace from a record written by `hybrid_inference.run_batch` (span pseudo-text only)."""
    spans = [
        Segment(kind="latent", exit=r.get("exit", ""), pseudo_text=r.get("pseudo_text", ""))
        for r in record.get("latent_readouts", [])
    ]
    output = record.get("output", "")
    return HybridTrace(
        question=record.get("question", ""),
        segments=parse_segments(output, spans),
        prediction=record.get("prediction") or extract_answer(output),
        stats={
            "latent_spans": record.get("latent_spans", len(spans)),
            "latent_steps": record.get("latent_steps", 0),
            "text_tokens": max(record.get("generation_steps", 0) - record.get("latent_steps", 0), 0),
            "search_calls": record.get("num_search_calls", 0),
            "python_calls": record.get("num_python_calls", 0),
            "end_reason": record.get("end_reason", ""),
        },
        sample_idx=int(record.get("sample_idx", 0)),
    )
