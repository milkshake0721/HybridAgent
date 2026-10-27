"""HybridSFT trainer: runs one stage, format activation, HybridLink
activation or self-conditioned HybridSFT, with the recipe's hyperparameters as defaults.

Launch on 8 GPUs with DeepSpeed ZeRO-2 (see scripts/run_hybrid_sft.sh):
  accelerate launch --config_file hybrid_sft/configs/accelerate_zero2.yaml -m hybrid_sft.train \
      --stage format_activation --model Qwen/Qwen3-8B-Base --data <jsonl or HF dataset id> \
      --output_dir outputs/hybrid_sft/format_activation
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hybrid_agent.model import add_anchor_tokens, ensure_chat_template, resolve_model_dir
from hybrid_sft.data import HybridSFTDataset, collate_batch, resolve_data_file
from hybrid_sft.model import FORMAT_ACTIVATION, HYBRIDLINK_ACTIVATION, SELF_CONDITIONED, STAGES, HybridSFTModel

# Epochs per stage. 1/1/1 and 1/1/2 are good practice as well; evaluate the saved HybridSFT
# checkpoints (e.g. on AIME and MATH-500) and take the best one as the initialization for HybridRL.
STAGE_EPOCHS = {FORMAT_ACTIVATION: 1, HYBRIDLINK_ACTIVATION: 2, SELF_CONDITIONED: 1}
STAGE_HYBRIDLINK_LR = {HYBRIDLINK_ACTIVATION: 2e-4, SELF_CONDITIONED: 5e-5}

WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
LOG_EVERY = 10
WANDB_PROJECT = "HybridAgent"


def parse_args():
    p = argparse.ArgumentParser(description="HybridSFT: one training stage")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--model", required=True,
                   help="base model for format activation, otherwise the previous stage's output (dir or HF id)")
    p.add_argument("--data", required=True, help="hybrid trajectories: local JSONL path or HF dataset repo id")
    p.add_argument("--data_file", default="train.jsonl", help="file name inside the HF dataset repo")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=None, help="default per stage: 1 / 2 / 1")
    p.add_argument("--lr", type=float, default=7e-6, help="backbone learning rate")
    p.add_argument("--hybridlink_lr", type=float, default=None,
                   help="HybridLink learning rate (default 2e-4 for activation, 5e-5 for self-conditioning)")
    p.add_argument("--batch_size", type=int, default=1, help="per device")
    p.add_argument("--grad_accum", type=int, default=2, help="must match the accelerate/DeepSpeed config")
    p.add_argument("--max_length", type=int, default=16888, help="longer trajectories are dropped")
    p.add_argument("--truncate_length", type=int, default=0, help="optional sequence truncation cap during training (0 = disabled)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=0, help="max intermediate checkpoints to keep (0 = unlimited)")
    p.add_argument(
        "--optim",
        default="adamw_torch",
        choices=("adamw_torch", "paged_adamw_8bit", "adamw_bnb_8bit"),
        help="optimizer type (default: adamw_torch; use paged_adamw_8bit for single-GPU 8-bit AdamW)",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument(
        "--resume_from", default=None,
        help="'auto' (latest resumable checkpoint-* in output_dir, else start fresh) or a checkpoint dir",
    )
    p.add_argument(
        "--save_training_state", type=int, default=1,
        help="also write optimizer/scheduler/step into checkpoint-*/training_state.pt so the run can resume",
    )
    p.add_argument(
        "--freeze_experts", type=int, default=0,
        help="for MoE models (e.g. Gemma 4 26B-A4B): freeze routed experts (.experts.) and train shared attention/MLP/router/embeddings/HybridLink",
    )
    p.add_argument(
        "--offload_expert_grads", type=int, default=0,
        help="for MoE full-parameter training: immediately offload .experts. gradients to CPU RAM during backward via custom autograd hook",
    )
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = STAGE_EPOCHS[args.stage]
    if args.hybridlink_lr is None:
        args.hybridlink_lr = STAGE_HYBRIDLINK_LR.get(args.stage, args.lr)
    return args


class _ExpertGradOffloadFn(torch.autograd.Function):
    """Redirects MoE expert weight gradients to CPU RAM immediately during backward
    so 45.7 GB of expert gradients never accumulate in GPU VRAM."""

    @staticmethod
    def forward(ctx, dummy_anchor: torch.Tensor, weight: torch.nn.Parameter, cpu_grads: dict):
        ctx.weight = weight
        ctx.cpu_grads = cpu_grads
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.cpu_grads[ctx.weight].add_(grad_output.to("cpu", non_blocking=False))
        return None, None, None


def _build_optimizer(param_groups: list[dict], optim_name: str):
    if optim_name == "paged_adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(param_groups, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    if optim_name == "adamw_bnb_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(param_groups, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)


def _restore_paged_optim_state(optim) -> None:
    base_optim = optim
    while hasattr(base_optim, "optimizer"):
        base_optim = base_optim.optimizer
    if getattr(base_optim, "is_paged", False):
        for p, p_state in base_optim.state.items():
            if not isinstance(p_state, dict):
                continue
            for key in ("state1", "state2"):
                val = p_state.get(key)
                if isinstance(val, torch.Tensor) and not getattr(val, "is_paged", False):
                    buf = base_optim.get_state_buffer(p, dtype=val.dtype)
                    buf.copy_(val)
                    p_state[key] = buf
                    del val


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Linear warmup, then cosine decay clamped at the end of training."""
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


TRAINING_STATE_FILE = "training_state.pt"


def _ckpt_step(path: Path) -> int:
    suffix = path.name.split("-")[-1]
    return int(suffix) if suffix.isdigit() else -1


def _is_complete_resumable_ckpt(path: Path) -> bool:
    return path.is_dir() and (path / TRAINING_STATE_FILE).is_file() and (path / "adapter_config.json").is_file()


def find_resume_dir(resume_from: str | None, output_dir: str) -> Path | None:
    """Resolve --resume_from to a checkpoint dir that holds both weights and training state."""
    if not resume_from:
        return None
    if resume_from != "auto":
        path = Path(resume_from)
        if not _is_complete_resumable_ckpt(path):
            raise FileNotFoundError(
                f"complete resumable checkpoint ({TRAINING_STATE_FILE} + adapter_config.json) not found in {path}"
            )
        return path
    candidates = [
        p for p in Path(output_dir).glob("checkpoint-*")
        if _is_complete_resumable_ckpt(p) and _ckpt_step(p) >= 0
    ]
    return max(candidates, key=_ckpt_step) if candidates else None


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    set_seed(args.seed)
    is_main = accelerator.is_main_process

    resume_dir = find_resume_dir(args.resume_from, args.output_dir)
    if resume_dir is not None:
        if accelerator.num_processes > 1:
            raise ValueError("--resume_from with single-file training_state.pt is only supported for single-process runs")
        # weights (backbone + HybridLink) come from the checkpoint being resumed
        model_dir = resume_dir
        if is_main:
            print(f"Resuming {args.stage} from {resume_dir}", flush=True)
    else:
        model_dir = resolve_model_dir(args.model)
        if args.resume_from == "auto" and is_main:
            print(f"No resumable checkpoint in {args.output_dir}; starting {args.stage} from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ensure_chat_template(tokenizer)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_main:
        print(f"Loading model from {model_dir} ...", flush=True)
    model = HybridSFTModel.from_pretrained(
        model_dir, dtype=torch.bfloat16, require_hybridlink=(args.stage != FORMAT_ACTIVATION),
    )
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(args.stage, freeze_experts=bool(args.freeze_experts))
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    use_expert_offload = (
        bool(args.offload_expert_grads)
        and not bool(args.freeze_experts)
        and args.stage in (FORMAT_ACTIVATION, SELF_CONDITIONED)
    )
    expert_params: list[torch.nn.Parameter] = []
    expert_modules: list[torch.nn.Module] = []
    if use_expert_offload:
        if accelerator.num_processes > 1:
            raise ValueError(
                "--offload_expert_grads is a single-GPU optimization; use DeepSpeed ZeRO-2/3 or --freeze_experts on multi-GPU"
            )
        seen_exp: set[int] = set()
        for m in model.backbone.get_decoder().modules():
            if m.__class__.__name__ == "Gemma4TextExperts":
                expert_modules.append(m)
                for attr in ("gate_up_proj", "down_proj"):
                    p = m._parameters.get(attr)
                    if p is not None and p.requires_grad and id(p) not in seen_exp:
                        seen_exp.add(id(p))
                        p.requires_grad = False
                        expert_params.append(p)
        if not expert_params:
            raise ValueError("--offload_expert_grads requires trainable Gemma4TextExperts parameters in the model")

    data_file = resolve_data_file(args.data, args.data_file)
    dataset = HybridSFTDataset(
        tokenizer, data_file, max_length=args.max_length, text_only=(args.stage == FORMAT_ACTIVATION),
    )
    if is_main:
        print(f"Dataset: {len(dataset)} trajectories from {data_file}", flush=True)
    if len(dataset) == 0:
        raise RuntimeError("no usable trajectories in the data file")

    # explicit generator seeded before prepare() and per epoch, so a resumed run replays the same sample order
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
        generator=shuffle_generator,
    )

    backbone_params = list({id(p): p for p in model.backbone.parameters() if p.requires_grad}.values())
    link_params = list({id(p): p for p in model.hybridlink.parameters() if p.requires_grad}.values())
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if link_params:
        param_groups.append({"params": link_params, "lr": args.hybridlink_lr})
    if is_main:
        exp_numel = sum(p.numel() for p in expert_params)
        if exp_numel > 0:
            print(f"Trainable parameters: backbone_gpu={sum(p.numel() for p in backbone_params):,} "
                  f"experts_cpu_offload={exp_numel:,} hybridlink={sum(p.numel() for p in link_params):,} | optim={args.optim}", flush=True)
        else:
            print(f"Trainable parameters: backbone={sum(p.numel() for p in backbone_params):,} "
                  f"hybridlink={sum(p.numel() for p in link_params):,} | optim={args.optim} | freeze_experts={bool(args.freeze_experts)}", flush=True)
        gpu_grad_trainable = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in link_params)
        total_model_params = sum(p.numel() for p in model.parameters())
        is_8bit_opt = args.optim in ("paged_adamw_8bit", "adamw_bnb_8bit")
        opt_bytes_per_param = 2 if is_8bit_opt else 12
        opt_label = "8-bit AdamW" if is_8bit_opt else "FP32 AdamW"
        est_vram_gb = (
            total_model_params * 2
            + gpu_grad_trainable * 2
            + (gpu_grad_trainable + exp_numel) * opt_bytes_per_param
        ) / 1e9
        print(f"Estimated base GPU VRAM (BF16 weights + GPU BF16 grads + {opt_label}): ~{est_vram_gb:.1f} GB", flush=True)
    optimizer = _build_optimizer(param_groups, args.optim)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.gradient_accumulation_steps != args.grad_accum:
        raise RuntimeError(
            f"gradient accumulation mismatch: --grad_accum {args.grad_accum} vs "
            f"{accelerator.gradient_accumulation_steps} in the accelerate/DeepSpeed config"
        )

    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    max_steps = args.epochs * steps_per_epoch
    if max_steps <= 0:
        raise ValueError("no optimizer steps: reduce grad_accum or batch_size, or use more data")
    warmup_steps = max(1, int(max_steps * WARMUP_RATIO))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    for group in optimizer.param_groups:  # first update at the step-0 learning rate
        group["lr"] = 0.0

    cpu_grads: dict[torch.nn.Parameter, torch.Tensor] = {}
    expert_optimizers: list = []
    expert_schedulers: list = []
    if expert_params:
        cpu_grads = {p: torch.zeros_like(p, device="cpu", dtype=torch.bfloat16) for p in expert_params}
        dummy_anchor = torch.zeros(1, device=accelerator.device, dtype=torch.bfloat16, requires_grad=True)
        for m in expert_modules:
            orig_fwd = m.forward

            def _make_patched_fwd(mod, orig):
                def _fwd(hidden_states, top_k_index, top_k_weights):
                    gu_param = mod._parameters["gate_up_proj"]
                    dn_param = mod._parameters["down_proj"]
                    if torch.is_grad_enabled():
                        mod._parameters["gate_up_proj"] = _ExpertGradOffloadFn.apply(dummy_anchor, gu_param, cpu_grads)
                        mod._parameters["down_proj"] = _ExpertGradOffloadFn.apply(dummy_anchor, dn_param, cpu_grads)
                    try:
                        return orig(hidden_states, top_k_index, top_k_weights)
                    finally:
                        mod._parameters["gate_up_proj"] = gu_param
                        mod._parameters["down_proj"] = dn_param

                return _fwd

            m.forward = _make_patched_fwd(m, orig_fwd)
        for p in expert_params:
            opt_e = _build_optimizer([{"params": [p], "lr": args.lr}], args.optim)
            sch_e = cosine_schedule_with_warmup(opt_e, warmup_steps, max_steps)
            for group in opt_e.param_groups:
                group["lr"] = 0.0
            expert_optimizers.append(opt_e)
            expert_schedulers.append(sch_e)

    start_step = 0
    if resume_dir is not None:
        state = torch.load(str(resume_dir / TRAINING_STATE_FILE), map_location="cpu", weights_only=False)
        if state.get("stage") != args.stage:
            raise ValueError(f"checkpoint stage {state.get('stage')!r} does not match --stage {args.stage!r}")
        if state.get("max_steps") != max_steps:
            raise ValueError(f"checkpoint max_steps {state.get('max_steps')} != current {max_steps}; "
                             "data/batch/epoch settings must match the original run")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_paged_optim_state(optimizer)
        if expert_optimizers and "expert_optimizers" in state:
            for opt_e, sch_e, opt_sd, sch_sd in zip(
                expert_optimizers, expert_schedulers, state["expert_optimizers"], state["expert_schedulers"]
            ):
                opt_e.load_state_dict(opt_sd)
                sch_e.load_state_dict(sch_sd)
                _restore_paged_optim_state(opt_e)
        start_step = int(state["global_step"])
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_main:
            print(f"Restored optimizer/scheduler at step {start_step}/{max_steps}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = None
    if is_main:
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training {args.stage}: {max_steps} steps "
              f"(effective batch {args.batch_size * args.grad_accum * accelerator.num_processes})", flush=True)
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project=WANDB_PROJECT, name=f"hybrid_sft-{args.stage}-{Path(args.output_dir).name}",
                config=vars(args), dir=args.output_dir,
            )

    def save(ckpt_dir: str) -> None:
        accelerator.wait_for_everyone()
        full_sd = accelerator.get_state_dict(model)  # collective
        if is_main:
            is_intermediate = Path(ckpt_dir).name.startswith("checkpoint-")
            if is_intermediate and args.save_training_state and accelerator.num_processes == 1:
                # written (atomically) before the weights: adapter_config.json, the last file of
                # save_pretrained(), therefore also marks the training state as complete
                os.makedirs(ckpt_dir, exist_ok=True)
                tmp_path = os.path.join(ckpt_dir, TRAINING_STATE_FILE + ".tmp")
                state_payload = {
                    "stage": args.stage,
                    "global_step": global_step,
                    "max_steps": max_steps,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if expert_optimizers:
                    state_payload["expert_optimizers"] = [opt_e.state_dict() for opt_e in expert_optimizers]
                    state_payload["expert_schedulers"] = [sch_e.state_dict() for sch_e in expert_schedulers]
                torch.save(state_payload, tmp_path)
                os.replace(tmp_path, os.path.join(ckpt_dir, TRAINING_STATE_FILE))
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir, tokenizer, full_sd)
            if args.save_total_limit > 0 and is_intermediate:
                parent = Path(ckpt_dir).parent
                ckpts = sorted(
                    (p for p in parent.glob("checkpoint-*") if p.is_dir() and _ckpt_step(p) >= 0),
                    key=_ckpt_step,
                )
                for old_ckpt in ckpts[:-args.save_total_limit]:
                    shutil.rmtree(old_ckpt, ignore_errors=True)
        del full_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    model.train()
    global_step = start_step
    pbar = tqdm(total=max_steps, initial=start_step, desc=args.stage, disable=not is_main)
    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}
    start_epoch = start_step // steps_per_epoch
    skip_batches = (start_step % steps_per_epoch) * args.grad_accum

    for epoch in range(start_epoch, args.epochs):
        if global_step >= max_steps:
            break
        shuffle_generator.manual_seed(args.seed + epoch)
        epoch_loader = dataloader
        if epoch == start_epoch and skip_batches > 0:
            if is_main:
                print(f"Skipping {skip_batches} already-trained batches of epoch {epoch}", flush=True)
            epoch_loader = accelerator.skip_first_batches(dataloader, skip_batches)
        for batch in epoch_loader:
            with accelerator.accumulate(model):
                cap = args.truncate_length if args.truncate_length > 0 else None
                input_ids = batch["input_ids"][:, :cap]
                attention_mask = batch["attention_mask"][:, :cap]
                loss_mask = batch["loss_mask"][:, :cap]
                latent_mask = batch["latent_mask"][:, :cap]
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    loss_mask=loss_mask, latent_mask=latent_mask, stage=args.stage,
                )
                loss = out["loss"]
                for key in ("loss", "backbone_loss", "hybridlink_loss"):
                    accum[key] += out[key].item()
                accum["n"] += 1
                del out
                accelerator.backward(loss)
                del loss

                if accelerator.sync_gradients:
                    if not expert_optimizers:
                        accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    else:
                        gpu_params = [p for p in model.parameters() if p.grad is not None]
                        total_sq = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                        for p in gpu_params:
                            total_sq.add_(p.grad.detach().float().square().sum())
                        for g_cpu in cpu_grads.values():
                            total_sq.add_(g_cpu.detach().float().square().sum().to(accelerator.device))
                        total_norm = float(total_sq.sqrt().item())
                        clip_coef = (
                            float(MAX_GRAD_NORM / (total_norm + 1e-6))
                            if (MAX_GRAD_NORM > 0 and total_norm > MAX_GRAD_NORM)
                            else 1.0
                        )
                        if clip_coef < 1.0:
                            for p in gpu_params:
                                p.grad.mul_(clip_coef)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        for p, opt_e, sch_e in zip(expert_params, expert_optimizers, expert_schedulers):
                            p.grad = cpu_grads[p].to(p.device, non_blocking=True)
                            if clip_coef < 1.0:
                                p.grad.mul_(clip_coef)
                            opt_e.step()
                            sch_e.step()
                            p.grad = None
                            cpu_grads[p].zero_()
                    global_step += 1
                    n = max(accum["n"], 1)
                    stats = {k: accum[k] / n for k in ("loss", "backbone_loss", "hybridlink_loss")}
                    stats["lr"] = scheduler.get_last_lr()[0]
                    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if k != "lr"})
                    pbar.update(1)
                    if wandb_run is not None:
                        wandb_run.log({f"train/{k}": v for k, v in stats.items()} | {"train/step": global_step})
                    if is_main and global_step % LOG_EVERY == 0:
                        print(f"step={global_step} loss={stats['loss']:.4f} lr={stats['lr']:.2e}", flush=True)
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    if global_step >= max_steps:
                        break
        if global_step >= max_steps:
            break

    save(args.output_dir)
    if is_main:
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
