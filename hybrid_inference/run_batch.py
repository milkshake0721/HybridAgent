"""Batch inference with HybridAgent: reads questions from a JSONL file, runs high-concurrency
hybrid generation with TensorRT/SGLang/eager backends and Python/search tools (supporting both
multi-question lockstep batching via --question_batch_size and multi-sample rollouts via --num_samples),
and appends one JSON record per sample with the output, extracted answer, latent-span readouts,
and tool-call statistics.

  python -m hybrid_inference.run_batch --model <checkpoint dir or HF id> \
      --backend tensorrt --question_batch_size 32 \
      --input examples/questions.jsonl --output outputs/results.jsonl --num_samples 4
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import time

import torch

from hybrid_agent import (
    HybridGenerationEngine,
    SYSTEM_PROMPT,
    build_tool_executor,
    extract_answer,
    load_hybrid_agent,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HybridAgent batch inference")
    p.add_argument("--model", required=True, help="HybridAgent checkpoint: local dir or HF model id")
    p.add_argument("--input", required=True, help="JSONL with one {\"id\", \"question\"} per line")
    p.add_argument("--output", required=True, help="JSONL output (appended; see --resume)")
    p.add_argument(
        "--backend",
        choices=("tensorrt", "eager", "sglang"),
        default="tensorrt",
        help="Execution backend: 'tensorrt' (default), 'eager', or 'sglang'",
    )
    p.add_argument("--num_samples", type=int, default=1, help="samples per question")
    p.add_argument("--batch_size", type=int, default=None, help="samples generated together (default: num_samples)")
    p.add_argument(
        "--question_batch_size",
        "--question-batch-size",
        "--concurrency",
        dest="question_batch_size",
        type=int,
        default=1,
        help="number of distinct questions batched concurrently in lockstep",
    )
    p.add_argument(
        "--no_tools",
        "--no-tools",
        "--disable_tools",
        "--disable-tools",
        dest="no_tools",
        action="store_true",
        help="disable all <python> and <search> tool calls (e.g. for AIME 2026 No Tools)",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_response_tokens", type=int, default=4096, help="text tokens + latent steps per sample")
    p.add_argument("--count_tool_tokens", action="store_true", help="count tool observations toward the budget")
    p.add_argument("--max_tool_calls", type=int, default=5, help="tool calls per sample")
    p.add_argument("--max_search_calls", type=int, default=5)
    p.add_argument("--max_python_calls", type=int, default=5)
    p.add_argument(
        "--continue_on_tool_cap",
        action="store_true",
        help="keep generating after the tool cap instead of ending the sample",
    )
    p.add_argument("--max_turns", type=int, default=30)
    p.add_argument("--min_latent_steps", type=int, default=4)
    p.add_argument("--max_latent_steps", type=int, default=200)
    p.add_argument("--no_latent_readout", action="store_true", help="skip the pseudo-text readout of latent spans")
    p.add_argument(
        "--record_latent_readout",
        dest="no_latent_readout",
        action="store_false",
        help="explicitly enable the pseudo-text readout of latent spans",
    )
    p.add_argument("--search_max_results", type=int, default=10)
    p.add_argument("--search_result_length", type=int, default=500)
    p.add_argument("--python_timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_id", type=int, default=None, help="process this shard of the questions (default: RANK env)")
    p.add_argument("--num_shards", type=int, default=None, help="total shards (default: WORLD_SIZE env)")
    p.add_argument("--resume", action="store_true", help="skip (id, sample) pairs already in --output")
    p.add_argument("--limit", type=int, default=0, help="stop after this many questions (0 = all)")
    args = p.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = args.num_samples
    if args.shard_id is None:
        args.shard_id = int(os.environ.get("RANK", 0))
    if args.num_shards is None:
        args.num_shards = int(os.environ.get("WORLD_SIZE", 1))
    args.question_batch_size = max(1, int(args.question_batch_size))
    if args.no_tools:
        args.max_tool_calls = 0
        args.max_python_calls = 0
        args.max_search_calls = 0
    return args


def read_questions(path: str) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "question" not in rec:
                raise ValueError(f"record without a question: {line[:80]}")
            rec.setdefault("id", len(questions))
            questions.append(rec)
    return questions


def read_done(path: str) -> set[tuple]:
    done = set()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["id"], rec["sample_idx"]))
                except Exception:
                    continue
    return done


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    q_batch = max(1, int(getattr(args, "question_batch_size", 1)))
    max_bs = max(64, q_batch * max(1, int(args.batch_size)))
    model, tokenizer = load_hybrid_agent(
        args.model,
        device=device,
        backend=args.backend,
        max_batch_size=max_bs,
    )
    tools = build_tool_executor(
        args.python_timeout,
        args.search_max_results,
        args.search_result_length,
        require_search=False,
    )

    questions = read_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    mine = [(i, q) for i, q in enumerate(questions) if i % args.num_shards == args.shard_id]
    done = read_done(args.output) if args.resume else set()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} questions x {args.num_samples} samples "
        f"(backend={args.backend}, question_batch_size={q_batch}) -> {args.output}",
        flush=True,
    )

    with open(args.output, "a", encoding="utf-8") as out:
        pending_items: list[tuple[int, dict, list[int]]] = []
        for q_index, rec in mine:
            pend = [s for s in range(args.num_samples) if (rec["id"], s) not in done]
            if pend:
                pending_items.append((q_index, rec, pend))

        for wave_start in range(0, len(pending_items), q_batch):
            wave = pending_items[wave_start : wave_start + q_batch]
            if (
                len(wave) > 1
                and all(len(w[2]) == args.num_samples for w in wave)
                and args.batch_size >= args.num_samples
            ):
                q_texts = [w[1]["question"] for w in wave]
                engine = HybridGenerationEngine(
                    model,
                    tokenizer,
                    device,
                    tools,
                    SYSTEM_PROMPT,
                    q_texts,
                    batch_size=args.num_samples,
                    max_response_tokens=args.max_response_tokens,
                    count_tool_tokens=args.count_tool_tokens,
                    max_tool_calls=args.max_tool_calls,
                    max_search_calls=args.max_search_calls,
                    max_python_calls=args.max_python_calls,
                    terminate_on_tool_cap=not args.continue_on_tool_cap,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    min_latent_steps=args.min_latent_steps,
                    max_latent_steps=args.max_latent_steps,
                    record_latent_readout=not args.no_latent_readout,
                )
                rng = torch.Generator(device=device)
                rng.manual_seed(args.seed * 1_000_003 + wave[0][0] * 1_000)
                t0 = time.time()
                engine.run(rng)
                wall = time.time() - t0
                for local_q_idx, (_, rec, sample_ids) in enumerate(wave):
                    meta = {k: v for k, v in rec.items() if k not in ("id", "question")}
                    for s_offset, sample_idx in enumerate(sample_ids):
                        row = engine.rows[local_q_idx * args.num_samples + s_offset]
                        result = engine.row_result(row)
                        record = {
                            "id": rec["id"],
                            "sample_idx": sample_idx,
                            "question": rec["question"],
                            **({"meta": meta} if meta else {}),
                            "output": result["output"],
                            "prediction": extract_answer(result["output"]),
                            "latent_readouts": result.get("latent_readouts", []),
                            "output_rendered": result.get("output_rendered", result["output"]),
                            "num_tool_calls": result["python_calls"] + result["search_calls"],
                            "num_search_calls": result["search_calls"],
                            "num_python_calls": result["python_calls"],
                            "generation_steps": result["generation_steps"],
                            "latent_steps": result["latent_steps"],
                            "latent_spans": result["latent_spans"],
                            "end_reason": result["end_reason"],
                            "wall_time_s": round(wall, 2),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                try:
                    os.fsync(out.fileno())
                except OSError:
                    pass
                print(
                    f"[shard {args.shard_id}] wave of {len(wave)} questions x {args.num_samples} samples in {wall:.2f}s "
                    f"(latent steps {sum(engine.row_result(r)['latent_steps'] for r in engine.rows)})",
                    flush=True,
                )
            else:
                for q_index, rec, pending in wave:
                    meta = {k: v for k, v in rec.items() if k not in ("id", "question")}
                    for start in range(0, len(pending), args.batch_size):
                        sample_ids = pending[start : start + args.batch_size]
                        engine = HybridGenerationEngine(
                            model,
                            tokenizer,
                            device,
                            tools,
                            SYSTEM_PROMPT,
                            rec["question"],
                            batch_size=len(sample_ids),
                            max_response_tokens=args.max_response_tokens,
                            count_tool_tokens=args.count_tool_tokens,
                            max_tool_calls=args.max_tool_calls,
                            max_search_calls=args.max_search_calls,
                            max_python_calls=args.max_python_calls,
                            terminate_on_tool_cap=not args.continue_on_tool_cap,
                            max_turns=args.max_turns,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            min_latent_steps=args.min_latent_steps,
                            max_latent_steps=args.max_latent_steps,
                            record_latent_readout=not args.no_latent_readout,
                        )
                        rng = torch.Generator(device=device)
                        rng.manual_seed(args.seed * 1_000_003 + q_index * 1_000 + sample_ids[0])
                        t0 = time.time()
                        engine.run(rng)
                        wall = time.time() - t0
                        for row, sample_idx in zip(engine.rows, sample_ids):
                            result = engine.row_result(row)
                            record = {
                                "id": rec["id"],
                                "sample_idx": sample_idx,
                                "question": rec["question"],
                                **({"meta": meta} if meta else {}),
                                "output": result["output"],
                                "prediction": extract_answer(result["output"]),
                                "latent_readouts": result.get("latent_readouts", []),
                                "output_rendered": result.get("output_rendered", result["output"]),
                                "num_tool_calls": result["python_calls"] + result["search_calls"],
                                "num_search_calls": result["search_calls"],
                                "num_python_calls": result["python_calls"],
                                "generation_steps": result["generation_steps"],
                                "latent_steps": result["latent_steps"],
                                "latent_spans": result["latent_spans"],
                                "end_reason": result["end_reason"],
                                "wall_time_s": round(wall, 2),
                            }
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                        try:
                            os.fsync(out.fileno())
                        except OSError:
                            pass
                        print(
                            f"[shard {args.shard_id}] question {rec['id']}: {len(sample_ids)} samples in {wall:.2f}s "
                            f"(latent steps {sum(engine.row_result(r)['latent_steps'] for r in engine.rows)})",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
