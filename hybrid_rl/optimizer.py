"""ZeRO-1 AdamW for data-parallel HybridRL: every rank holds the full bf16 policy for
rollout and replay; fp32 master weights and Adam moments are sharded by parameter
ownership, gradients are reduce-summed to the owner in flat buckets, and updated
parameters are broadcast back.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class ZeroOneAdamW:
    def __init__(
        self,
        param_groups,
        *,
        rank: int,
        world_size: int,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_8bit: bool = False,
    ):
        self.rank = int(rank)
        self.world = int(world_size)
        self.betas = tuple(float(value) for value in betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.use_8bit = bool(use_8bit)
        if len(self.betas) != 2 or any(not 0.0 <= value < 1.0 for value in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError(f"invalid rank/world {self.rank}/{self.world}")
        self.step_count = 0

        flat: list[tuple[torch.nn.Parameter, float]] = []
        seen_params: set[int] = set()
        for group in param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad and id(parameter) not in seen_params:
                    seen_params.add(id(parameter))
                    flat.append((parameter, float(group["lr"])))
        if not flat:
            raise ValueError("ZeroOneAdamW received no trainable parameters")

        # greedy balanced ownership, largest tensors first
        order = sorted(range(len(flat)), key=lambda index: -flat[index][0].numel())
        loads = [0] * self.world
        self.owner = [0] * len(flat)
        for index in order:
            owner = min(range(self.world), key=lambda r: loads[r])
            self.owner[index] = owner
            loads[owner] += flat[index][0].numel()

        self.params = [parameter for parameter, _ in flat]
        self.lrs = [lr for _, lr in flat]
        self.master: dict[int, torch.Tensor] = {}
        self.master_is_parameter: dict[int, bool] = {}
        self.exp_avg: dict[int, torch.Tensor] = {}
        self.exp_avg_sq: dict[int, torch.Tensor] = {}
        self.param_steps: dict[int, int] = {}   # per-parameter bias correction, as in torch AdamW
        owned_groups: dict[float, list[torch.nn.Parameter]] = {}
        self.owned_numel = 0
        for index, (parameter, lr) in enumerate(flat):
            if self.owner[index] != self.rank:
                continue
            self.owned_numel += parameter.numel()
            if self.use_8bit:
                owned_groups.setdefault(lr, []).append(parameter)
            else:
                direct_fp32 = parameter.dtype == torch.float32
                self.master_is_parameter[index] = direct_fp32
                self.master[index] = parameter.data if direct_fp32 else parameter.detach().float().clone()
                self.exp_avg[index] = torch.zeros_like(self.master[index])
                self.exp_avg_sq[index] = torch.zeros_like(self.master[index])
                self.param_steps[index] = 0
        self._bnb_opt = None
        if self.use_8bit and owned_groups:
            import bitsandbytes as bnb

            self._bnb_opt = bnb.optim.PagedAdamW8bit(
                [{"params": params, "lr": lr} for lr, params in owned_groups.items()],
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.params:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def all_reduce_grads(self) -> None:
        """SUM each gradient onto its owner rank, in flat buckets; unused parameters
        (no gradient on any rank) keep grad=None so they receive no update."""
        if self.world == 1:
            return
        local_used = torch.tensor(
            [1 if parameter.grad is not None else 0 for parameter in self.params],
            device=self.params[0].device, dtype=torch.int32,
        )
        dist.all_reduce(local_used, op=dist.ReduceOp.MAX)
        globally_used = [bool(value) for value in local_used.cpu().tolist()]

        bucket_numel = 64 * 1024 * 1024
        by_owner: dict[int, list[int]] = {}
        for index in range(len(self.params)):
            if globally_used[index]:
                by_owner.setdefault(self.owner[index], []).append(index)
            else:
                self.params[index].grad = None
        for owner, indices in sorted(by_owner.items()):
            start = 0
            while start < len(indices):
                chunk: list[int] = []
                numel = 0
                while start < len(indices) and (
                    numel == 0 or numel + self.params[indices[start]].numel() <= bucket_numel
                ):
                    chunk.append(indices[start])
                    numel += self.params[indices[start]].numel()
                    start += 1
                flat = torch.empty(numel, device=self.params[chunk[0]].device, dtype=self.params[chunk[0]].dtype)
                offset = 0
                for index in chunk:
                    parameter = self.params[index]
                    n = parameter.numel()
                    if parameter.grad is None:
                        flat[offset:offset + n].zero_()
                    else:
                        flat[offset:offset + n].copy_(parameter.grad.reshape(-1))
                    offset += n
                dist.reduce(flat, dst=owner, op=dist.ReduceOp.SUM)
                offset = 0
                for index in chunk:
                    parameter = self.params[index]
                    n = parameter.numel()
                    parameter.grad = flat[offset:offset + n].view_as(parameter).clone() if owner == self.rank else None
                    offset += n
                del flat

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> float:
        """Global gradient norm from the owner-local shards, then scale in place."""
        total = torch.zeros((), device=self.params[0].device, dtype=torch.float32)
        for index, parameter in enumerate(self.params):
            if self.owner[index] == self.rank and parameter.grad is not None:
                total.add_(parameter.grad.detach().float().square().sum())
        if self.world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        norm = float(total.sqrt())
        if max_norm > 0 and norm > max_norm:
            scale = float(max_norm) / (norm + 1e-6)
            for index, parameter in enumerate(self.params):
                if self.owner[index] == self.rank and parameter.grad is not None:
                    parameter.grad.mul_(scale)
        return norm

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        if self.use_8bit:
            if self._bnb_opt is not None:
                self._bnb_opt.step()
            if self.world > 1:
                for index, parameter in enumerate(self.params):
                    dist.broadcast(parameter.data, src=self.owner[index])
            return
        beta1, beta2 = self.betas
        for index, parameter in enumerate(self.params):
            if self.owner[index] != self.rank or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            master = self.master[index]
            mean = self.exp_avg[index]
            variance = self.exp_avg_sq[index]
            lr = self.lrs[index]
            self.param_steps[index] += 1
            correction1 = 1.0 - beta1 ** self.param_steps[index]
            correction2 = 1.0 - beta2 ** self.param_steps[index]
            if self.weight_decay:
                master.mul_(1.0 - lr * self.weight_decay)
            mean.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            denominator = (variance / correction2).sqrt().add_(self.eps)
            master.addcdiv_(mean / correction1, denominator, value=-lr)
            if not self.master_is_parameter[index]:
                parameter.data.copy_(master.to(parameter.dtype))
        if self.world > 1:
            for index, parameter in enumerate(self.params):
                dist.broadcast(parameter.data, src=self.owner[index])

    def state_summary(self) -> str:
        mode = "PagedAdamW8bit (BF16 weights + BF16 grads)" if self.use_8bit else "ZeRO-1 FP32 AdamW"
        return (f"{mode}: {len(self.params)} tensors; rank {self.rank} owns "
                f"{self.owned_numel / 1e6:.1f}M parameters")
