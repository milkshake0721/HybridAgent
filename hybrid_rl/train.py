"""HybridRL trainer. Each step samples a group of hybrid
trajectories per prompt (Hybrid Policy Rollout), scores them with r_correct + r_format +
r_tool, computes group-normalized advantages, replays every trajectory in embedding space
with its recorded latent inputs fixed (Exact Rollout Replay), and updates the backbone
with the clipped objective over the sampled text positions (Hybrid Policy Factorization).
HybridLink stays frozen. Data-parallel over prompts with a ZeRO-1 sharded AdamW.

Launch with torchrun (see scripts/run_hybrid_rl.sh):
  torchrun --nnodes 4 --nproc_per_node 8 ... -m hybrid_rl.train \
      --model <HybridSFT checkpoint> --data <parquet or hf://repo/file> --task math --output_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

# the same attention kernels serve rollout and replay
torch.backends.cuda.enable_cudnn_sdp(False)

from hybrid_agent.prompts import SYSTEM_PROMPT  # noqa: E402
from hybrid_agent.tools import build_tool_executor  # noqa: E402
from hybrid_rl import dist as D  # noqa: E402
from hybrid_rl.config import (  # noqa: E402
    ADAM_BETAS, ADAM_EPS, ADV_EPS, LOG_ROLLOUTS_PER_STEP, MAX_GRAD_NORM, PYTHON_TIMEOUT,
    SEARCH_MAX_RESULTS, TASK_PRESETS, TASKS, WANDB_PROJECT, WEIGHT_DECAY, HybridRLConfig,
)
from hybrid_rl.data import epoch_batches, load_prompts, resolve_data_path  # noqa: E402
from hybrid_rl.grpo import group_advantages, policy_loss  # noqa: E402
from hybrid_rl.minibatch import build_minibatches  # noqa: E402
from hybrid_rl.optimizer import ZeroOneAdamW  # noqa: E402
from hybrid_rl.policy import load_policy, save_checkpoint  # noqa: E402
from hybrid_rl.replay import exact_rollout_replay, replay_agreement  # noqa: E402
from hybrid_rl.reward import compute_reward  # noqa: E402
from hybrid_rl.rollout import rollout_group  # noqa: E402
from hybrid_rl.trajectory import HybridTrajectory  # noqa: E402


def parse_args(argv=None) -> HybridRLConfig:
    p = argparse.ArgumentParser(description="HybridRL")
    p.add_argument("--model", required=True, help="HybridSFT checkpoint: local dir or HF model id")
    p.add_argument("--data", required=True, help="prompts parquet: local path or hf://<dataset repo>/<file>")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=None, help="default per task: 200 (math) / 150 (search)")
    p.add_argument("--save_steps", type=int, default=20, help="checkpoint every N steps (0 = final only)")
    p.add_argument("--prompts_per_step", type=int, default=64, help="unique prompts per rollout batch")
    p.add_argument("--group_size", type=int, default=8, help="rollouts per prompt")
    p.add_argument("--num_minibatches", type=int, default=8, help="optimizer updates per rollout batch")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tool_calls", type=int, default=None, help="per trajectory; default per task")
    p.add_argument("--max_response_tokens", type=int, default=None,
                   help="per-trajectory budget (text + latent + tool tokens); default per task")
    p.add_argument("--min_latent_steps", type=int, default=None, help="per latent span; default per task")
    p.add_argument("--max_latent_steps", type=int, default=None, help="per latent span; default per task")
    p.add_argument("--search_result_length", type=int, default=None, help="chars per snippet; default per task")
    p.add_argument("--freeze_experts", action="store_true", help="freeze MoE routed experts (.experts.) during RL")
    p.add_argument("--use_8bit", action="store_true", help="use 8-bit PagedAdamW in ZeroOneAdamW")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args(argv)
    preset = TASK_PRESETS[args.task]
    for key in ("max_steps", "max_tool_calls", "max_response_tokens", "min_latent_steps",
                "max_latent_steps", "search_result_length"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    cfg = HybridRLConfig(**vars(args))
    cfg.validate()
    return cfg


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def attach_group_advantages(local_groups, group_size: int):
    """Group-normalized advantages for complete, valid groups; incomplete groups are dropped
    rather than re-standardized. Returns (groups, local indices, zero-variance count, dropped count)."""
    groups, indices = [], []
    zero_variance = dropped = 0
    for local_index, group in enumerate(local_groups):
        if len(group) != int(group_size) or any(not t.valid for t in group):
            dropped += 1
            continue
        advantages, is_zero_var = group_advantages([t.reward for t in group])
        zero_variance += int(is_zero_var)
        for t, advantage in zip(group, advantages):
            t.advantage = float(advantage)
        groups.append(group)
        indices.append(local_index)
    return groups, indices, zero_variance, dropped


def main(argv=None) -> None:
    cfg = parse_args(argv)
    rank, world, local_rank = D.init()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)

    if cfg.prompts_per_step % world != 0:
        raise ValueError(f"prompts_per_step={cfg.prompts_per_step} must be divisible by world size {world}")
    prompts_per_rank = cfg.prompts_per_step // world

    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "rl_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    D.barrier()
    log_handle = (output_dir / "train.log").open("a", encoding="utf-8") if is_main else None
    metrics_handle = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if is_main else None
    rollouts_handle = (output_dir / "rollouts.jsonl").open("a", encoding="utf-8") if is_main else None

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)
            log_handle.write(message + "\n")
            log_handle.flush()

    log(f"=== HybridRL ({cfg.task}) === world={world} model={cfg.model}")
    log(f"rollout batch: {cfg.prompts_per_step} prompts x {cfg.group_size} rollouts | "
        f"{cfg.num_minibatches} minibatches/step | lr={cfg.lr:g} | steps={cfg.max_steps}")

    prompts = load_prompts(resolve_data_path(cfg.data), cfg.ability)
    log(f"prompts={len(prompts)}")

    model, tokenizer = load_policy(cfg.model, device, freeze_experts=cfg.freeze_experts)
    tools = build_tool_executor(PYTHON_TIMEOUT, SEARCH_MAX_RESULTS, cfg.search_result_length, require_search=True)
    trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    log(f"trainable backbone parameters: {sum(p.numel() for p in trainable) / 1e9:.3f}B; HybridLink frozen")
    optimizer = ZeroOneAdamW(
        [{"params": trainable, "lr": cfg.lr}], rank=rank, world_size=world,
        betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY, use_8bit=cfg.use_8bit,
    )
    log("  " + optimizer.state_summary())

    wandb_run = None
    if cfg.wandb and is_main:
        import wandb

        wandb_run = wandb.init(project=WANDB_PROJECT, name=output_dir.name, config=cfg.to_dict(), dir=str(output_dir))

    prompt_stream = epoch_batches(prompts, cfg.prompts_per_step, cfg.seed)
    rollout_rng = torch.Generator(device=device).manual_seed(cfg.seed * 1009 + rank)
    optimizer_steps_total = 0

    for step in range(1, cfg.max_steps + 1):
        step_start = time.time()
        rank_prompts = D.shard_batch(next(prompt_stream), rank, world)
        local_groups: list[list[HybridTrajectory]] = []
        rollout_oom = 0
        rollout_error = ""

        # ---- Hybrid Policy Rollout ---------------------------------------------------
        model.backbone.eval()
        model.hybridlink.eval()
        rollout_start = time.time()
        with torch.no_grad():
            for prompt in rank_prompts:
                group: list[HybridTrajectory] = []
                try:
                    trajectories = rollout_group(
                        model=model, tokenizer=tokenizer, device=device, tool_executor=tools,
                        system_prompt=SYSTEM_PROMPT, prompt=prompt, cfg=cfg, rng=rollout_rng,
                    )
                except Exception as error:
                    if _is_oom(error):  # the whole group is dropped for this step
                        rollout_oom += cfg.group_size
                        print(f"[rank {rank}] rollout OOM on prompt {prompt.idx}; group dropped", flush=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        local_groups.append(group)
                        continue
                    rollout_error = f"{type(error).__name__}: {error}"
                    local_groups.append(group)
                    break
                for traj in trajectories:
                    reward = compute_reward(traj.output, prompt.gold, cfg.task,
                                            prompt_ends_in_think=traj.prompt_ends_in_think)
                    traj.reward = float(reward.total)
                    traj._reward = reward
                    traj._prompt = prompt
                    group.append(traj)
                local_groups.append(group)
        rollout_seconds = time.time() - rollout_start

        if D.all_any(bool(rollout_error), device):
            messages = [m for m in D.gather_objects(rollout_error, world) if m]
            log("FATAL rollout failure: " + " | ".join(messages[:4]))
            D.barrier()
            raise RuntimeError("rollout failed on at least one rank")

        # ---- rewards -> group advantages -> minibatches --------------------------------
        groups, group_indices, zero_variance_groups, dropped_groups = attach_group_advantages(
            local_groups, cfg.group_size,
        )
        minibatches = build_minibatches(
            groups, cfg.num_minibatches, seed=cfg.seed * 1_000_003 + step,
            rank=rank, world_size=world, local_group_indices=group_indices, groups_per_rank=prompts_per_rank,
        )
        # zero-advantage trajectories stay in the token denominator
        counts = []
        for minibatch in minibatches:
            counts += [sum(len(t.text_actions) for t in minibatch),
                       sum(1 for t in minibatch if abs(t.advantage) > ADV_EPS)]
        counts = D.all_sum_vec(counts, device)
        token_denominators = [max(int(counts[2 * i]), 1) for i in range(len(minibatches))]
        informative = [int(counts[2 * i + 1]) for i in range(len(minibatches))]
        first_informative = next((i for i, c in enumerate(informative) if c > 0), None)

        all_trajs = [t for group in local_groups for t in group]
        valid_trajs = [t for t in all_trajs if t.valid]
        population = D.all_sum_vec([
            sum(t.reward for t in valid_trajs),
            sum(1 for t in valid_trajs if t._reward.correct),
            sum(1 for t in valid_trajs if not t._reward.format_reason),
            sum(1 for t in valid_trajs if t._reward.r_tool > 0),
            sum(t.n_text_tokens for t in valid_trajs),
            sum(t.n_latent_steps for t in valid_trajs),
            sum(t.n_latent_spans for t in valid_trajs),
            sum(t.n_tool_calls for t in valid_trajs),
            sum(t.response_len for t in valid_trajs),
            len(valid_trajs),
            zero_variance_groups,
            len(groups),
            dropped_groups,
            len(local_groups),
            rollout_oom,
        ], device)
        (reward_sum, n_correct, n_format_ok, n_tool_bonus, text_sum, latent_sum, span_sum, tool_sum,
         resp_sum, n_valid, n_zero_var, n_groups, n_dropped, n_local_groups, n_oom) = population
        n_valid = max(n_valid, 1.0)

        # ---- Exact Rollout Replay + Hybrid Policy Update --------------------------------
        model.backbone.train()
        model.hybridlink.eval()
        stats = {"loss": 0.0, "n_text": 0, "clip_count": 0, "ratio_sum": 0.0, "grad_norm_sum": 0.0,
                 "optimizer_steps": 0, "skipped_zero_adv": 0, "agree_sum": 0.0, "agree_n": 0}
        for mb_index, minibatch in enumerate(minibatches):
            if informative[mb_index] == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            local_error = ""
            for traj in minibatch:
                if abs(traj.advantage) <= ADV_EPS:
                    stats["skipped_zero_adv"] += 1
                    continue
                try:
                    text_logp, old_text_logp = exact_rollout_replay(model, traj, device, with_grad=True)
                    if not (torch.isfinite(text_logp).all() and torch.isfinite(old_text_logp).all()):
                        raise FloatingPointError("non-finite text log probability")
                    if mb_index == first_informative:
                        # before the first update the policy equals the rollout policy:
                        # replayed and recorded log-probs should agree
                        agree_sum, agree_n = replay_agreement(text_logp, old_text_logp)
                        stats["agree_sum"] += agree_sum
                        stats["agree_n"] += agree_n
                    loss, traj_stats = policy_loss(text_logp, old_text_logp, traj.advantage,
                                                   token_denominators[mb_index])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite trajectory loss")
                    loss.backward()
                    stats["loss"] += float(loss.detach())
                    stats["n_text"] += traj_stats["n_text"]
                    stats["clip_count"] += traj_stats["clip_count"]
                    stats["ratio_sum"] += traj_stats["ratio_sum"]
                except Exception as error:
                    local_error = f"rank={rank} minibatch={mb_index} prompt={traj._prompt.idx} {type(error).__name__}: {error}"
                    print(local_error, flush=True)
                    if _is_oom(error) and device.type == "cuda":
                        torch.cuda.empty_cache()
                    break

            if D.all_any(bool(local_error), device):  # fail closed on every rank
                optimizer.zero_grad(set_to_none=True)
                messages = [m for m in D.gather_objects(local_error, world) if m]
                log(f"FATAL minibatch {mb_index} cancelled on all ranks: " + " | ".join(messages[:4]))
                D.barrier()
                raise RuntimeError("minibatch update failed")

            optimizer.all_reduce_grads()
            link_has_grad = any(p.grad is not None for p in model.hybridlink.parameters())
            if D.all_any(link_has_grad, device):
                raise RuntimeError("frozen HybridLink received a gradient")
            grad_norm = optimizer.clip_grad_norm(MAX_GRAD_NORM)
            if D.all_any(not math.isfinite(grad_norm), device):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("non-finite global gradient norm; update cancelled")
            optimizer.step()
            optimizer_steps_total += 1
            stats["optimizer_steps"] += 1
            stats["grad_norm_sum"] += grad_norm

        # ---- logging ---------------------------------------------------------------------
        (g_loss, g_n_text, g_clip, g_ratio, g_gnorm, g_steps, g_skipped, g_agree_sum, g_agree_n) = D.all_sum_vec(
            [stats["loss"], stats["n_text"], stats["clip_count"], stats["ratio_sum"], stats["grad_norm_sum"],
             stats["optimizer_steps"], stats["skipped_zero_adv"], stats["agree_sum"], stats["agree_n"]], device,
        )
        updates = g_steps / max(world, 1)  # identical on every rank
        metrics = {
            "step": step,
            "reward_mean": reward_sum / n_valid,
            "accuracy": n_correct / n_valid,
            "format_ok_rate": n_format_ok / n_valid,
            "tool_bonus_rate": n_tool_bonus / n_valid,
            "text_tokens_mean": text_sum / n_valid,
            "latent_steps_mean": latent_sum / n_valid,
            "latent_spans_mean": span_sum / n_valid,
            "tool_calls_mean": tool_sum / n_valid,
            "response_len_mean": resp_sum / n_valid,
            "zero_variance_group_rate": n_zero_var / max(n_groups, 1.0),
            "dropped_group_rate": n_dropped / max(n_local_groups, 1.0),
            "rollout_oom": int(n_oom),
            "loss": g_loss / max(updates, 1.0),
            "grad_norm": g_gnorm / max(g_steps, 1.0),
            "clip_fraction": g_clip / max(g_n_text, 1.0),
            "ratio_mean": g_ratio / max(g_n_text, 1.0),
            "replay_agreement_abs_logp": g_agree_sum / max(g_agree_n, 1.0),
            "optimizer_steps": int(round(updates)),
            "skipped_zero_adv": int(g_skipped),
            "rollout_s": rollout_seconds,
            "step_s": time.time() - step_start,
        }
        log(f"step={step:04d} reward={metrics['reward_mean']:+.4f} acc={metrics['accuracy']:.3f} "
            f"fmt_ok={metrics['format_ok_rate']:.3f} zero_var={metrics['zero_variance_group_rate']:.3f} "
            f"loss={metrics['loss']:+.5f} gnorm={metrics['grad_norm']:.3f} ratio={metrics['ratio_mean']:.3f} "
            f"clip={metrics['clip_fraction']:.3f} agree={metrics['replay_agreement_abs_logp']:.4f} "
            f"latent={metrics['latent_steps_mean']:.1f} text={metrics['text_tokens_mean']:.0f} "
            f"tools={metrics['tool_calls_mean']:.2f} updates={metrics['optimizer_steps']}/{cfg.num_minibatches} "
            f"time={metrics['step_s'] / 60:.1f}m rollout={rollout_seconds / 60:.1f}m")
        if is_main:
            metrics_handle.write(json.dumps(metrics) + "\n")
            metrics_handle.flush()
            for traj in all_trajs[:LOG_ROLLOUTS_PER_STEP]:
                rollouts_handle.write(json.dumps({
                    "step": step, "question": traj._prompt.question, "gold": traj._prompt.gold,
                    "pred": traj._reward.pred, "reward": traj.reward, "r_correct": traj._reward.r_correct,
                    "r_format": traj._reward.r_format, "r_tool": traj._reward.r_tool,
                    "format_reason": traj._reward.format_reason, "n_text_tokens": traj.n_text_tokens,
                    "n_latent_steps": traj.n_latent_steps, "n_tool_calls": traj.n_tool_calls,
                    "valid": traj.valid, "output": traj.output,
                }) + "\n")
            rollouts_handle.flush()
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        if cfg.save_steps and step % cfg.save_steps == 0 and step < cfg.max_steps:
            D.barrier()
            if is_main:
                save_checkpoint(model, tokenizer, output_dir / f"checkpoint-{step}", cfg)
            D.barrier()

    D.barrier()
    if is_main:
        save_checkpoint(model, tokenizer, output_dir / f"checkpoint-{cfg.max_steps}", cfg)
        log(f"saved final checkpoint to {output_dir / f'checkpoint-{cfg.max_steps}'}")
        if wandb_run is not None:
            wandb_run.finish()
        for handle in (log_handle, metrics_handle, rollouts_handle):
            handle.close()
    D.cleanup()


if __name__ == "__main__":
    main()
