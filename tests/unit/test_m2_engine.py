"""Unit tests for Milestone M2 (Features F4, F5, F6):
- F4: Multi-question batched & concurrent HybridGenerationEngine / HybridEngine
- F5: Vectorized step execution, multi-token FEED prefill, pre-allocated attn_mask,
      gated _row_tail, and lazy BF16 _BANK_CACHE readout
- F6: Multimodal vision support (Gemma4ForConditionalGeneration + AutoProcessor /
      HybridMultimodalProcessor fallback + images/pixel_values in engine)
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Dual-Mode Runtime Support:
# Uses real PyTorch + Transformers when installed; otherwise registers a
# genuine NumPy-backed PyTorch + Transformers runtime in sys.modules so the
# full M2 unit test suite executes cleanly in zero-dependency CPU containers.
# ---------------------------------------------------------------------------
def _ensure_torch_and_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    torch_mod = types.ModuleType("torch")

    class _DType:
        def __init__(self, name: str, np_dtype: Any, is_float: bool = False) -> None:
            self.name = name
            self.np_dtype = np.dtype(np_dtype)
            self.is_floating_point = is_float

        def __repr__(self) -> str:
            return f"torch.{self.name}"

        def __str__(self) -> str:
            return f"torch.{self.name}"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _DType) and self.name == other.name

        def __hash__(self) -> int:
            return hash(self.name)

    float32 = _DType("float32", np.float32, True)
    float64 = _DType("float64", np.float64, True)
    bfloat16 = _DType("bfloat16", np.float32, True)
    float16 = _DType("float16", np.float16, True)
    int64 = _DType("int64", np.int64, False)
    int32 = _DType("int32", np.int32, False)
    int16 = _DType("int16", np.int16, False)
    int8 = _DType("int8", np.int8, False)
    uint8 = _DType("uint8", np.uint8, False)
    bool_dt = _DType("bool", np.bool_, False)

    torch_mod.float32 = float32
    torch_mod.float = float32
    torch_mod.float64 = float64
    torch_mod.double = float64
    torch_mod.bfloat16 = bfloat16
    torch_mod.float16 = float16
    torch_mod.half = float16
    torch_mod.int64 = int64
    torch_mod.long = int64
    torch_mod.int32 = int32
    torch_mod.int = int32
    torch_mod.int16 = int16
    torch_mod.int8 = int8
    torch_mod.uint8 = uint8
    torch_mod.bool = bool_dt
    torch_mod.dtype = _DType

    def _infer_dtype(arr: np.ndarray, explicit: _DType | None = None) -> _DType:
        if explicit is not None:
            return explicit
        if arr.dtype == np.bool_:
            return bool_dt
        if arr.dtype in (np.int64, np.int32, np.int16, np.int8, np.uint8):
            if arr.dtype == np.uint8:
                return uint8
            if arr.dtype == np.int32:
                return int32
            return int64
        if arr.dtype == np.float16:
            return float16
        return float32

    class device:
        def __init__(self, dev: str | "device" = "cpu") -> None:
            if isinstance(dev, device):
                self.type = dev.type
                self._str = dev._str
            else:
                s = str(dev)
                self._str = s
                self.type = s.split(":")[0]

        def __repr__(self) -> str:
            return f"device(type={self.type!r})"

        def __str__(self) -> str:
            return self._str

        def __eq__(self, other: object) -> bool:
            if isinstance(other, device):
                return self.type == other.type
            if isinstance(other, str):
                return self.type == other.split(":")[0]
            return False

        def __hash__(self) -> int:
            return hash(self.type)

    torch_mod.device = device

    class Generator:
        def __init__(self, device: Any = "cpu") -> None:
            self.device = torch_mod.device(device)
            self._rng = np.random.default_rng(0)

        def manual_seed(self, seed: int) -> "Generator":
            self._rng = np.random.default_rng(int(seed))
            return self

    torch_mod.Generator = Generator

    def _unwrap_idx(idx: Any) -> Any:
        if isinstance(idx, Tensor):
            return idx._arr
        if isinstance(idx, tuple):
            return tuple(_unwrap_idx(x) for x in idx)
        if isinstance(idx, list):
            if any(isinstance(x, Tensor) for x in idx):
                return [_unwrap_idx(x) for x in idx]
            return idx
        return idx

    class Tensor:
        def __init__(
            self,
            data: Any,
            *,
            dtype: _DType | None = None,
            device: Any = "cpu",
            requires_grad: bool = False,
        ) -> None:
            if isinstance(data, Tensor):
                arr = data._arr
                dt = dtype or data.dtype
                dev = device if device is not None else data.device
            elif isinstance(data, np.ndarray):
                arr = data
                dt = _infer_dtype(arr, dtype)
                dev = device
            else:
                arr = np.asarray(data)
                if dtype is None:
                    if arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
                        dt = float32
                    else:
                        dt = _infer_dtype(arr, None)
                else:
                    dt = dtype
                dev = device
            if arr.dtype != dt.np_dtype:
                arr = arr.astype(dt.np_dtype)
            self._arr: np.ndarray = arr
            self.dtype: _DType = dt
            self.device: device = torch_mod.device(dev if dev is not None else "cpu")
            self.requires_grad: bool = bool(requires_grad)
            self.grad: Tensor | None = None

        @property
        def shape(self) -> tuple[int, ...]:
            return self._arr.shape

        @property
        def ndim(self) -> int:
            return self._arr.ndim

        def dim(self) -> int:
            return self._arr.ndim

        def size(self, d: int | None = None) -> Any:
            return self._arr.shape if d is None else self._arr.shape[d]

        def numel(self) -> int:
            return int(self._arr.size)

        def data_ptr(self) -> int:
            return int(self._arr.ctypes.data)

        def is_floating_point(self) -> bool:
            return self.dtype.is_floating_point

        def detach(self) -> "Tensor":
            return Tensor(self._arr, dtype=self.dtype, device=self.device)

        def clone(self) -> "Tensor":
            return Tensor(self._arr.copy(), dtype=self.dtype, device=self.device)

        def contiguous(self) -> "Tensor":
            if self._arr.flags["C_CONTIGUOUS"]:
                return self
            return Tensor(np.ascontiguousarray(self._arr), dtype=self.dtype, device=self.device)

        def cpu(self) -> "Tensor":
            return self.to(device="cpu")

        def cuda(self, *args: Any, **kwargs: Any) -> "Tensor":
            return self.to(device="cuda")

        def to(self, *args: Any, **kwargs: Any) -> "Tensor":
            target_dtype = kwargs.get("dtype", None)
            target_device = kwargs.get("device", None)
            for a in args:
                if isinstance(a, _DType):
                    target_dtype = a
                elif isinstance(a, (device, str)):
                    target_device = a
                elif isinstance(a, Tensor):
                    target_dtype = a.dtype
                    target_device = a.device
            new_dt = target_dtype if target_dtype is not None else self.dtype
            new_dev = torch_mod.device(target_device) if target_device is not None else self.device
            if new_dt == self.dtype and new_dev == self.device:
                return self
            new_arr = self._arr.astype(new_dt.np_dtype) if self._arr.dtype != new_dt.np_dtype else self._arr.copy()
            return Tensor(new_arr, dtype=new_dt, device=new_dev)

        def float(self) -> "Tensor":
            return self.to(dtype=float32)

        def bfloat16(self) -> "Tensor":
            return self.to(dtype=bfloat16)

        def half(self) -> "Tensor":
            return self.to(dtype=float16)

        def long(self) -> "Tensor":
            return self.to(dtype=int64)

        def int(self) -> "Tensor":
            return self.to(dtype=int32)

        def bool(self) -> "Tensor":
            return self.to(dtype=bool_dt)

        def view(self, *shape: Any) -> "Tensor":
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(self._arr.reshape(shape), dtype=self.dtype, device=self.device)

        def reshape(self, *shape: Any) -> "Tensor":
            return self.view(*shape)

        def unsqueeze(self, dim: int) -> "Tensor":
            return Tensor(np.expand_dims(self._arr, axis=dim), dtype=self.dtype, device=self.device)

        def squeeze(self, dim: int | None = None) -> "Tensor":
            if dim is None:
                return Tensor(np.squeeze(self._arr), dtype=self.dtype, device=self.device)
            if self._arr.shape[dim] == 1:
                return Tensor(np.squeeze(self._arr, axis=dim), dtype=self.dtype, device=self.device)
            return self

        def expand(self, *sizes: Any) -> "Tensor":
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            target = list(sizes)
            cur = list(self._arr.shape)
            if len(target) > len(cur):
                cur = [1] * (len(target) - len(cur)) + cur
            for i, s in enumerate(target):
                if s == -1:
                    target[i] = cur[i]
            return Tensor(np.broadcast_to(self._arr, tuple(target)), dtype=self.dtype, device=self.device)

        def repeat(self, *repeats: Any) -> "Tensor":
            if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
                repeats = tuple(repeats[0])
            return Tensor(np.tile(self._arr, repeats), dtype=self.dtype, device=self.device)

        def repeat_interleave(self, repeats: int, dim: int = 0) -> "Tensor":
            return Tensor(np.repeat(self._arr, repeats, axis=dim), dtype=self.dtype, device=self.device)

        def permute(self, *dims: Any) -> "Tensor":
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = tuple(dims[0])
            return Tensor(np.transpose(self._arr, dims), dtype=self.dtype, device=self.device)

        def transpose(self, dim0: int, dim1: int) -> "Tensor":
            axes = list(range(self.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            return Tensor(np.transpose(self._arr, axes), dtype=self.dtype, device=self.device)

        @property
        def T(self) -> "Tensor":
            return Tensor(self._arr.T, dtype=self.dtype, device=self.device)

        def tolist(self) -> Any:
            return self._arr.tolist()

        def item(self) -> Any:
            return self._arr.item()

        def numpy(self) -> np.ndarray:
            return self._arr

        def argmax(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
            res = np.argmax(self._arr, axis=dim)
            if keepdim and dim is not None:
                res = np.expand_dims(res, axis=dim)
            return Tensor(res, dtype=int64, device=self.device)

        def sum(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> "Tensor":
            res = np.sum(self._arr, axis=dim, keepdims=keepdim)
            out_dt = int64 if self.dtype == bool_dt else self.dtype
            return Tensor(res, dtype=out_dt, device=self.device)

        def mean(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> "Tensor":
            res = np.mean(self._arr, axis=dim, keepdims=keepdim)
            return Tensor(res, dtype=float32, device=self.device)

        def pow(self, exponent: float) -> "Tensor":
            return Tensor(np.power(self._arr, exponent), dtype=self.dtype, device=self.device)

        def rsqrt(self) -> "Tensor":
            return Tensor(1.0 / np.sqrt(self._arr), dtype=self.dtype, device=self.device)

        def sqrt(self) -> "Tensor":
            return Tensor(np.sqrt(self._arr), dtype=self.dtype, device=self.device)

        def exp(self) -> "Tensor":
            return Tensor(np.exp(self._arr), dtype=self.dtype, device=self.device)

        def log(self) -> "Tensor":
            return Tensor(np.log(self._arr), dtype=self.dtype, device=self.device)

        def clamp(self, min: float | None = None, max: float | None = None) -> "Tensor":
            return Tensor(np.clip(self._arr, min, max), dtype=self.dtype, device=self.device)

        def clamp_min(self, min_val: float) -> "Tensor":
            return Tensor(np.maximum(self._arr, min_val), dtype=self.dtype, device=self.device)

        def clamp_max(self, max_val: float) -> "Tensor":
            return Tensor(np.minimum(self._arr, max_val), dtype=self.dtype, device=self.device)

        def cumsum(self, dim: int = -1) -> "Tensor":
            return Tensor(np.cumsum(self._arr, axis=dim), dtype=self.dtype, device=self.device)

        def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
            out = self._arr.copy()
            m = mask._arr.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
            out = np.where(m, value, out)
            return Tensor(out, dtype=self.dtype, device=self.device)

        def masked_fill_(self, mask: "Tensor", value: float) -> "Tensor":
            m = mask._arr.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
            self._arr[m] = value
            return self

        def scatter(self, dim: int, index: "Tensor", src: "Tensor" | float) -> "Tensor":
            out = self.clone()
            return out.scatter_(dim, index, src)

        def scatter_(self, dim: int, index: "Tensor", src: "Tensor" | float) -> "Tensor":
            idx_arr = index._arr
            src_arr = src._arr if isinstance(src, Tensor) else np.full_like(idx_arr, src, dtype=self._arr.dtype)
            np.put_along_axis(self._arr, idx_arr, src_arr, axis=dim)
            return self

        def index_select(self, dim: int, index: "Tensor") -> "Tensor":
            idx_arr = index._arr.reshape(-1)
            return Tensor(np.take(self._arr, idx_arr, axis=dim), dtype=self.dtype, device=self.device)

        def all(self) -> "Tensor":
            return Tensor(np.all(self._arr), dtype=bool_dt, device=self.device)

        def any(self) -> "Tensor":
            return Tensor(np.any(self._arr), dtype=bool_dt, device=self.device)

        def eq(self, other: Any) -> "Tensor":
            return self == other

        def le(self, other: Any) -> "Tensor":
            return self <= other

        def ge(self, other: Any) -> "Tensor":
            return self >= other

        def lt(self, other: Any) -> "Tensor":
            return self < other

        def gt(self, other: Any) -> "Tensor":
            return self > other

        def __len__(self) -> int:
            return int(self._arr.shape[0])

        def __iter__(self):
            for i in range(self.shape[0]):
                yield self[i]

        def __bool__(self) -> bool:
            return bool(self._arr.item() if self._arr.size == 1 else np.any(self._arr))

        def __int__(self) -> int:
            return int(self._arr.item())

        def __float__(self) -> float:
            return float(self._arr.item())

        def __getitem__(self, idx: Any) -> "Tensor":
            raw = _unwrap_idx(idx)
            res = self._arr[raw]
            return Tensor(np.asarray(res), dtype=self.dtype, device=self.device)

        def __setitem__(self, idx: Any, val: Any) -> None:
            raw = _unwrap_idx(idx)
            v = val._arr if isinstance(val, Tensor) else val
            self._arr[raw] = v

        def _binop(self, other: Any, op: Any, out_dtype: _DType | None = None) -> "Tensor":
            o = other._arr if isinstance(other, Tensor) else other
            res = op(self._arr, o)
            dt = out_dtype or (other.dtype if isinstance(other, Tensor) and not self.is_floating_point() and other.is_floating_point() else self.dtype)
            return Tensor(np.asarray(res), dtype=dt, device=self.device)

        def __add__(self, other: Any) -> "Tensor":
            return self._binop(other, np.add)

        def __radd__(self, other: Any) -> "Tensor":
            return self._binop(other, lambda a, b: np.add(b, a))

        def __iadd__(self, other: Any) -> "Tensor":
            o = other._arr if isinstance(other, Tensor) else other
            self._arr += o
            return self

        def __sub__(self, other: Any) -> "Tensor":
            return self._binop(other, np.subtract)

        def __rsub__(self, other: Any) -> "Tensor":
            return self._binop(other, lambda a, b: np.subtract(b, a))

        def __mul__(self, other: Any) -> "Tensor":
            return self._binop(other, np.multiply)

        def __rmul__(self, other: Any) -> "Tensor":
            return self._binop(other, lambda a, b: np.multiply(b, a))

        def __truediv__(self, other: Any) -> "Tensor":
            return self._binop(other, np.true_divide, out_dtype=float32 if not self.is_floating_point() else self.dtype)

        def __rtruediv__(self, other: Any) -> "Tensor":
            return self._binop(other, lambda a, b: np.true_divide(b, a), out_dtype=float32)

        def __matmul__(self, other: "Tensor") -> "Tensor":
            return Tensor(np.matmul(self._arr, other._arr), dtype=self.dtype, device=self.device)

        def __neg__(self) -> "Tensor":
            return Tensor(-self._arr, dtype=self.dtype, device=self.device)

        def __invert__(self) -> "Tensor":
            return Tensor(~self._arr, dtype=self.dtype, device=self.device)

        def __eq__(self, other: object) -> "Tensor":  # type: ignore[override]
            return self._binop(other, np.equal, out_dtype=bool_dt)

        def __ne__(self, other: object) -> "Tensor":  # type: ignore[override]
            return self._binop(other, np.not_equal, out_dtype=bool_dt)

        def __lt__(self, other: Any) -> "Tensor":
            return self._binop(other, np.less, out_dtype=bool_dt)

        def __le__(self, other: Any) -> "Tensor":
            return self._binop(other, np.less_equal, out_dtype=bool_dt)

        def __gt__(self, other: Any) -> "Tensor":
            return self._binop(other, np.greater, out_dtype=bool_dt)

        def __ge__(self, other: Any) -> "Tensor":
            return self._binop(other, np.greater_equal, out_dtype=bool_dt)

        def __repr__(self) -> str:
            return f"Tensor({self._arr!r}, dtype={self.dtype})"

    torch_mod.Tensor = Tensor

    class _NoGradContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

        def __call__(self, fn: Any) -> Any:
            return fn

    torch_mod.no_grad = lambda: _NoGradContext()
    torch_mod.inference_mode = lambda *a, **kw: _NoGradContext()

    def _shape_tuple(size: Any) -> tuple[int, ...]:
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            return tuple(int(x) for x in size[0])
        return tuple(int(x) for x in size)

    torch_mod.tensor = lambda data, dtype=None, device="cpu", **kw: Tensor(
        np.array(data._arr if isinstance(data, Tensor) else data, copy=True), dtype=dtype, device=device
    )
    torch_mod.as_tensor = lambda data, dtype=None, device="cpu": (
        data.to(dtype=dtype, device=device)
        if isinstance(data, Tensor)
        else Tensor(np.asarray(data), dtype=dtype, device=device)
    )
    torch_mod.zeros = lambda *size, dtype=float32, device="cpu", **kw: Tensor(
        np.zeros(_shape_tuple(size), dtype=(dtype or float32).np_dtype), dtype=dtype or float32, device=device
    )
    torch_mod.zeros_like = lambda t, dtype=None, device=None, **kw: Tensor(
        np.zeros_like(t._arr, dtype=(dtype or t.dtype).np_dtype), dtype=dtype or t.dtype, device=device or t.device
    )
    torch_mod.ones = lambda *size, dtype=float32, device="cpu", **kw: Tensor(
        np.ones(_shape_tuple(size), dtype=(dtype or float32).np_dtype), dtype=dtype or float32, device=device
    )
    torch_mod.ones_like = lambda t, dtype=None, device=None, **kw: Tensor(
        np.ones_like(t._arr, dtype=(dtype or t.dtype).np_dtype), dtype=dtype or t.dtype, device=device or t.device
    )
    torch_mod.empty = lambda *size, dtype=float32, device="cpu", **kw: Tensor(
        np.empty(_shape_tuple(size), dtype=(dtype or float32).np_dtype), dtype=dtype or float32, device=device
    )
    torch_mod.empty_like = lambda t, dtype=None, device=None, **kw: Tensor(
        np.empty_like(t._arr, dtype=(dtype or t.dtype).np_dtype), dtype=dtype or t.dtype, device=device or t.device
    )
    torch_mod.full = lambda size, fill_value, dtype=float32, device="cpu", **kw: Tensor(
        np.full(_shape_tuple((size,)), fill_value, dtype=(dtype or float32).np_dtype),
        dtype=dtype or float32,
        device=device,
    )
    torch_mod.full_like = lambda t, fill_value, dtype=None, device=None, **kw: Tensor(
        np.full_like(t._arr, fill_value, dtype=(dtype or t.dtype).np_dtype),
        dtype=dtype or t.dtype,
        device=device or t.device,
    )
    torch_mod.arange = lambda *args, dtype=int64, device="cpu", **kw: Tensor(
        np.arange(*args, dtype=(dtype or int64).np_dtype), dtype=dtype or int64, device=device
    )
    torch_mod.randn = lambda *size, dtype=float32, device="cpu", **kw: Tensor(
        np.random.default_rng(42).standard_normal(_shape_tuple(size)).astype((dtype or float32).np_dtype),
        dtype=dtype or float32,
        device=device,
    )
    torch_mod.cat = lambda tensors, dim=0: Tensor(
        np.concatenate([t._arr for t in tensors], axis=dim),
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    torch_mod.stack = lambda tensors, dim=0: Tensor(
        np.stack([t._arr for t in tensors], axis=dim),
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    torch_mod.where = lambda cond, x, y: Tensor(
        np.where(
            cond._arr if isinstance(cond, Tensor) else cond,
            x._arr if isinstance(x, Tensor) else x,
            y._arr if isinstance(y, Tensor) else y,
        ),
        dtype=x.dtype if isinstance(x, Tensor) else (y.dtype if isinstance(y, Tensor) else float32),
        device=x.device if isinstance(x, Tensor) else "cpu",
    )
    torch_mod.isfinite = lambda t: Tensor(np.isfinite(t._arr), dtype=bool_dt, device=t.device)
    torch_mod.isinf = lambda t: Tensor(np.isinf(t._arr), dtype=bool_dt, device=t.device)
    torch_mod.isnan = lambda t: Tensor(np.isnan(t._arr), dtype=bool_dt, device=t.device)
    torch_mod.log = lambda t: t.log()
    torch_mod.exp = lambda t: t.exp()
    torch_mod.cumsum = lambda t, dim=-1: t.cumsum(dim=dim)

    def _softmax(t: Tensor, dim: int = -1) -> Tensor:
        x = t._arr.astype(np.float64)
        mx = np.max(x, axis=dim, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        ex = np.exp(x - mx)
        ex = np.where(np.isfinite(x), ex, 0.0)
        sm = np.sum(ex, axis=dim, keepdims=True)
        sm = np.maximum(sm, 1e-30)
        return Tensor((ex / sm).astype(np.float32), dtype=float32, device=t.device)

    torch_mod.softmax = _softmax

    class _TopKResult:
        def __init__(self, values: Tensor, indices: Tensor) -> None:
            self.values = values
            self.indices = indices

        def __iter__(self):
            yield self.values
            yield self.indices

    def _topk(t: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> _TopKResult:
        arr = t._arr if largest else -t._arr
        idx = np.argsort(-arr, axis=dim)
        slc = [slice(None)] * arr.ndim
        slc[dim] = slice(0, k)
        top_idx = idx[tuple(slc)]
        top_vals = np.take_along_axis(t._arr, top_idx, axis=dim)
        return _TopKResult(
            Tensor(top_vals, dtype=t.dtype, device=t.device),
            Tensor(top_idx, dtype=int64, device=t.device),
        )

    torch_mod.topk = _topk

    def _sort(t: Tensor, dim: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:
        arr = -t._arr if descending else t._arr
        idx = np.argsort(arr, axis=dim, kind="stable")
        vals = np.take_along_axis(t._arr, idx, axis=dim)
        return (
            Tensor(vals, dtype=t.dtype, device=t.device),
            Tensor(idx, dtype=int64, device=t.device),
        )

    torch_mod.sort = _sort

    def _multinomial(
        probs: Tensor,
        num_samples: int = 1,
        replacement: bool = False,
        generator: Generator | None = None,
    ) -> Tensor:
        rng = generator._rng if generator is not None else np.random.default_rng()
        arr = probs._arr.astype(np.float64)
        orig_1d = arr.ndim == 1
        if orig_1d:
            arr = arr[None, :]
        rows_out = []
        v = arr.shape[-1]
        for row in arr:
            row = np.maximum(row, 0.0)
            s = row.sum()
            p = (row / s) if s > 0 else (np.ones(v) / v)
            choice = rng.choice(v, size=num_samples, replace=replacement or num_samples > 1, p=p)
            rows_out.append(choice)
        out = np.stack(rows_out, axis=0)
        if orig_1d:
            out = out[0]
        return Tensor(out, dtype=int64, device=probs.device)

    torch_mod.multinomial = _multinomial

    def _save(obj: Any, path: str | os.PathLike) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _load(path: str | os.PathLike, map_location: Any = None, weights_only: bool = True) -> Any:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    torch_mod.save = _save
    torch_mod.load = _load

    # torch.cuda stub
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    cuda_mod.synchronize = lambda *a, **kw: None
    torch_mod.cuda = cuda_mod
    sys.modules["torch.cuda"] = cuda_mod

    # torch.nn & torch.nn.functional
    nn_mod = types.ModuleType("torch.nn")
    F_mod = types.ModuleType("torch.nn.functional")

    class Parameter(Tensor):
        def __init__(self, data: Any, requires_grad: bool = True) -> None:
            super().__init__(data, dtype=data.dtype if isinstance(data, Tensor) else float32, requires_grad=requires_grad)

    class Module:
        def __init__(self) -> None:
            self._parameters: dict[str, Parameter] = {}
            self._modules: dict[str, "Module"] = {}
            self.training: bool = True

        def __setattr__(self, name: str, value: Any) -> None:
            if isinstance(value, Parameter):
                if "_parameters" not in self.__dict__:
                    self.__dict__["_parameters"] = {}
                self.__dict__["_parameters"][name] = value
            elif isinstance(value, Module):
                if "_modules" not in self.__dict__:
                    self.__dict__["_modules"] = {}
                self.__dict__["_modules"][name] = value
            super().__setattr__(name, value)

        def parameters(self):
            for p in self.__dict__.get("_parameters", {}).values():
                yield p
            for m in self.__dict__.get("_modules", {}).values():
                yield from m.parameters()

        def named_parameters(self, prefix: str = ""):
            for k, p in self.__dict__.get("_parameters", {}).items():
                name = f"{prefix}.{k}" if prefix else k
                yield name, p
            for k, m in self.__dict__.get("_modules", {}).items():
                sub = f"{prefix}.{k}" if prefix else k
                yield from m.named_parameters(sub)

        def named_modules(self, prefix: str = ""):
            yield prefix, self
            for k, m in self.__dict__.get("_modules", {}).items():
                sub = f"{prefix}.{k}" if prefix else k
                yield from m.named_modules(sub)

        def to(self, *args: Any, **kwargs: Any) -> "Module":
            for k, p in self.__dict__.get("_parameters", {}).items():
                t = p.to(*args, **kwargs)
                self.__dict__["_parameters"][k] = Parameter(t, requires_grad=p.requires_grad)
                super().__setattr__(k, self.__dict__["_parameters"][k])
            for m in self.__dict__.get("_modules", {}).values():
                m.to(*args, **kwargs)
            return self

        def eval(self) -> "Module":
            self.training = False
            for m in self.__dict__.get("_modules", {}).values():
                m.eval()
            return self

        def train(self, mode: bool = True) -> "Module":
            self.training = mode
            for m in self.__dict__.get("_modules", {}).values():
                m.train(mode)
            return self

        def state_dict(self) -> dict[str, Tensor]:
            return {k: v.detach().clone() for k, v in self.named_parameters()}

        def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True) -> Any:
            own = dict(self.named_parameters())
            for k, v in state_dict.items():
                if k in own:
                    own[k]._arr = v._arr.astype(own[k].dtype.np_dtype).copy()
                elif strict:
                    raise KeyError(f"Unexpected key in state_dict: {k}")
            return None

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    class Linear(Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: Any = float32) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            rng = np.random.default_rng(in_features * 31 + out_features)
            w = (rng.standard_normal((out_features, in_features)) * 0.02).astype(np.float32)
            self.weight = Parameter(Tensor(w, dtype=dtype or float32))
            self.bias = Parameter(torch_mod.zeros(out_features, dtype=dtype or float32)) if bias else None

        def forward(self, x: Tensor) -> Tensor:
            out = x @ self.weight.T
            if self.bias is not None:
                out = out + self.bias
            return out

    class LayerNorm(Module):
        def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype: Any = float32) -> None:
            super().__init__()
            self.eps = eps
            self.weight = Parameter(torch_mod.ones(normalized_shape, dtype=dtype or float32))
            self.bias = Parameter(torch_mod.zeros(normalized_shape, dtype=dtype or float32))

        def forward(self, x: Tensor) -> Tensor:
            arr = x._arr.astype(np.float32)
            mean = arr.mean(axis=-1, keepdims=True)
            var = arr.var(axis=-1, keepdims=True)
            normed = (arr - mean) / np.sqrt(var + self.eps)
            out = normed * self.weight._arr + self.bias._arr
            return Tensor(out, dtype=x.dtype, device=x.device)

    class Embedding(Module):
        def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Any = float32) -> None:
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            rng = np.random.default_rng(1234)
            w = rng.standard_normal((num_embeddings, embedding_dim)).astype(np.float32)
            self.weight = Parameter(Tensor(w, dtype=dtype or float32))

        def forward(self, input_ids: Tensor) -> Tensor:
            return Tensor(self.weight._arr[input_ids._arr], dtype=self.weight.dtype, device=input_ids.device)

    class SiLU(Module):
        def forward(self, x: Tensor) -> Tensor:
            return F_mod.silu(x)

    init_mod = types.ModuleType("torch.nn.init")
    init_mod.zeros_ = lambda t: (t._arr.fill(0.0), t)[1]
    init_mod.ones_ = lambda t: (t._arr.fill(1.0), t)[1]
    init_mod.normal_ = lambda t, mean=0.0, std=0.02: t

    def _normalize(x: Tensor, p: float = 2.0, dim: int = -1, eps: float = 1e-12) -> Tensor:
        arr = x._arr.astype(np.float32)
        norm = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True)
        norm = np.maximum(norm, eps)
        return Tensor(arr / norm, dtype=x.dtype, device=x.device)

    def _silu(x: Tensor) -> Tensor:
        a = x._arr.astype(np.float32)
        return Tensor(a / (1.0 + np.exp(-a)), dtype=x.dtype, device=x.device)

    def _interpolate(
        x: Tensor,
        size: tuple[int, int] | None = None,
        mode: str = "bilinear",
        align_corners: bool = False,
    ) -> Tensor:
        if size is None or (x.shape[-2], x.shape[-1]) == tuple(size):
            return x
        th, tw = size
        arr = x._arr
        ys = np.linspace(0, arr.shape[-2] - 1, th).astype(np.int64)
        xs = np.linspace(0, arr.shape[-1] - 1, tw).astype(np.int64)
        out = arr[..., ys[:, None], xs[None, :]]
        return Tensor(out, dtype=x.dtype, device=x.device)

    F_mod.normalize = _normalize
    F_mod.silu = _silu
    F_mod.softmax = _softmax
    F_mod.interpolate = _interpolate

    nn_mod.Module = Module
    nn_mod.Parameter = Parameter
    nn_mod.Linear = Linear
    nn_mod.LayerNorm = LayerNorm
    nn_mod.Embedding = Embedding
    nn_mod.SiLU = SiLU
    nn_mod.init = init_mod
    nn_mod.functional = F_mod

    torch_mod.nn = nn_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = F_mod
    sys.modules["torch.nn.init"] = init_mod

    # transformers stub
    tf_mod = types.ModuleType("transformers")
    tf_modeling_utils = types.ModuleType("transformers.modeling_utils")
    tf_modeling_utils.maybe_load_adapters = lambda path, download_kwargs, **adapter_kwargs: (
        None,
        path,
        adapter_kwargs,
    )

    class PreTrainedTokenizerBase:
        pass

    class _AutoFactory:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise OSError("No local model in stub AutoFactory")

    class AutoModelForCausalLM(_AutoFactory):
        pass

    class AutoModelForImageTextToText(_AutoFactory):
        pass

    class Gemma4ForConditionalGeneration(_AutoFactory):
        pass

    class AutoTokenizer(_AutoFactory):
        pass

    class AutoProcessor(_AutoFactory):
        pass

    class AutoConfig(_AutoFactory):
        pass

    tf_mod.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    tf_mod.AutoModelForCausalLM = AutoModelForCausalLM
    tf_mod.AutoModelForImageTextToText = AutoModelForImageTextToText
    tf_mod.Gemma4ForConditionalGeneration = Gemma4ForConditionalGeneration
    tf_mod.AutoTokenizer = AutoTokenizer
    tf_mod.AutoProcessor = AutoProcessor
    tf_mod.AutoConfig = AutoConfig
    tf_mod.modeling_utils = tf_modeling_utils
    sys.modules["transformers"] = tf_mod
    sys.modules["transformers.modeling_utils"] = tf_modeling_utils

    # safetensors & huggingface_hub stubs
    st_mod = types.ModuleType("safetensors")
    st_torch_mod = types.ModuleType("safetensors.torch")
    st_mod.safe_open = lambda *a, **kw: contextlib.nullcontext()
    st_torch_mod.load_file = lambda *a, **kw: {}
    sys.modules["safetensors"] = st_mod
    sys.modules["safetensors.torch"] = st_torch_mod

    hf_mod = types.ModuleType("huggingface_hub")
    hf_mod.snapshot_download = lambda repo_id, revision=None, **kw: str(repo_id)
    sys.modules["huggingface_hub"] = hf_mod


_ensure_torch_and_transformers()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import hybrid_agent  # noqa: E402
from hybrid_agent import (  # noqa: E402
    DECODE,
    DEFAULT_GEMMA4_PROCESSOR_REPO,
    DONE,
    ENTRY_ANCHOR,
    EXIT_ANCHOR,
    FEED,
    IDLE,
    LATENT,
    HybridAgentModel,
    HybridEngine,
    HybridGenerationEngine,
    HybridLink,
    HybridMultimodalProcessor,
    HybridTrajectory,
    LatentSpanReadout,
    RowState,
    ToolExecutor,
    add_anchor_tokens,
    ensure_chat_template,
    latent_readout,
    load_hybrid_agent,
    load_hybrid_processor,
    render_with_readout,
)
from hybrid_agent.constants import (  # noqa: E402
    RESULT_CLOSE,
    RESULT_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)
from hybrid_agent.hybridlink import HYBRIDLINK_CONFIG  # noqa: E402
from hybrid_agent.readout import (  # noqa: E402
    _BANK_CACHE,
    nearest_token_ids,
    normalized_embeddings,
)
from hybrid_agent.sampling import (  # noqa: E402
    expand_cache_batch,
    forward_step,
    sample_rows,
    token_log_probs,
)


# ---------------------------------------------------------------------------
# Deterministic Synthetic Test Fixtures (Tokenizer, Cache, Backbone, Tools)
# ---------------------------------------------------------------------------
class DummyBatchEncoding(dict):
    def to(self, dev: Any) -> "DummyBatchEncoding":
        out = DummyBatchEncoding()
        for k, v in self.items():
            out[k] = v.to(dev) if isinstance(v, torch.Tensor) else v
        return out


class DummyTokenizer:
    """Deterministic word/tag tokenizer for testing all M2 engine transitions."""

    def __init__(self) -> None:
        base_tokens = [
            "<pad>",           # 0
            "<eos>",           # 1
            ENTRY_ANCHOR,      # 2: <latent>
            EXIT_ANCHOR,       # 3: </latent>
            THINK_OPEN,        # 4: <think>
            THINK_CLOSE,       # 5: </think>
            "<python>",        # 6
            "</python>",       # 7
            "<search>",        # 8
            "</search>",       # 9
            RESULT_OPEN,       # 10: <result>
            RESULT_CLOSE,      # 11: </result>
            "<answer>",        # 12
            "</answer>",       # 13
            "\\boxed{42}",     # 14
            "<|image|>",       # 15
            "print(6*7)",      # 16
            "42",              # 17
            "obs_tok_1",       # 18
            "obs_tok_2",       # 19
            "obs_tok_3",       # 20
            "obs_tok_4",       # 21
            "step_word_a",     # 22
            "step_word_b",     # 23
            "\\boxed{A}",      # 24
            "\\boxed{B}",      # 25
            "\\boxed{C}",      # 26
        ]
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(base_tokens)}
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(base_tokens)}
        self.pad_token_id: int = 0
        self.eos_token_id: int = 1
        self.unk_token_id: int | None = None
        self.padding_side: str = "left"
        self.chat_template: str | None = None
        self.all_special_tokens: list[str] = [
            "<pad>", "<eos>", ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE
        ]
        self.additional_special_tokens: list[str] = [
            ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE
        ]
        self.decode_calls: list[list[int]] = []

    def __len__(self) -> int:
        return max(len(self.id_to_token), 64)

    def _get_or_add(self, tok: str) -> int:
        if tok not in self.token_to_id:
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = tok
            self.token_to_id[tok] = new_id
        return self.token_to_id[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.token_to_id:
            return [self.token_to_id[text]]
        import re

        pieces = re.findall(r"<[^>]+>|[^\s<]+", text)
        if not pieces:
            return [self._get_or_add("empty")]
        return [self._get_or_add(p) for p in pieces]

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids_list = [int(i) for i in token_ids]
        self.decode_calls.append(ids_list)
        pieces: list[str] = []
        for idx in ids_list:
            tok = self.id_to_token.get(idx, f"tok_{idx}")
            if skip_special_tokens and tok in ("<pad>", "<eos>"):
                continue
            pieces.append(tok)
        return "".join(pieces)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def add_special_tokens(self, special_tokens_dict: dict[str, list[str]]) -> int:
        added = 0
        for tok in special_tokens_dict.get("additional_special_tokens", []):
            if tok not in self.token_to_id:
                self._get_or_add(tok)
                added += 1
            if tok not in self.all_special_tokens:
                self.all_special_tokens.append(tok)
            if tok not in self.additional_special_tokens:
                self.additional_special_tokens.append(tok)
        return added

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        prompt = " | ".join(parts)
        if add_generation_prompt:
            prompt = f"{prompt} {THINK_OPEN}\n"
        return prompt

    def save_pretrained(self, output_dir: str | os.PathLike) -> None:
        Path(output_dir, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        padding: bool | str = False,
        **kwargs: Any,
    ) -> DummyBatchEncoding:
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if return_tensors == "pt":
                return DummyBatchEncoding(
                    input_ids=torch.tensor([ids], dtype=torch.long),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                )
            return DummyBatchEncoding(input_ids=ids, attention_mask=[1] * len(ids))

        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        if padding or return_tensors == "pt":
            max_l = max((len(e) for e in encoded), default=0)
            padded = []
            masks = []
            for e in encoded:
                pad_n = max_l - len(e)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * pad_n + e)
                    masks.append([0] * pad_n + [1] * len(e))
                else:
                    padded.append(e + [self.pad_token_id] * pad_n)
                    masks.append([1] * len(e) + [0] * pad_n)
            if return_tensors == "pt":
                return DummyBatchEncoding(
                    input_ids=torch.tensor(padded, dtype=torch.long),
                    attention_mask=torch.tensor(masks, dtype=torch.long),
                )
            return DummyBatchEncoding(input_ids=padded, attention_mask=masks)
        return DummyBatchEncoding(input_ids=encoded)


class DummyCacheLayer:
    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.keys = k
        self.values = v
        self.batch_size = int(k.shape[0])


class DummyDynamicCache:
    def __init__(self, batch_size: int, seq_len: int, hidden_size: int = 16, device: Any = "cpu") -> None:
        k = torch.ones((batch_size, 2, seq_len, hidden_size // 2), dtype=torch.float32, device=device)
        v = torch.ones((batch_size, 2, seq_len, hidden_size // 2), dtype=torch.float32, device=device)
        self.layers = [DummyCacheLayer(k, v)]


class DummyModelOutput:
    def __init__(self, past_key_values: Any, logits: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.past_key_values = past_key_values
        self.logits = logits
        self.hidden_states = hidden_states


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        eye = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for i in range(vocab_size):
            for d in range(hidden_size):
                eye[i, d] = math.cos((i + 1) * (d + 1) * 0.41) + (1.5 if d == (i % hidden_size) else 0.0)
        self.weight = nn.Parameter(eye)
        self.embed_scale = torch.tensor(math.sqrt(hidden_size), dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.view(-1)
        out = self.weight.index_select(0, flat).view(*input_ids.shape, self.embedding_dim)
        return out * self.embed_scale.to(device=out.device, dtype=out.dtype)


class ScriptedBackbone(nn.Module):
    """Scriptable causal / multimodal backbone that records every prefill & decode step
    and emits deterministic scripted tokens per row."""

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 16,
        *,
        with_vision: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = DummyEmbedding(vocab_size, hidden_size)
        self.config = types.SimpleNamespace(
            eos_token_id=1,
            vision_config=types.SimpleNamespace(
                image_size=32,
                patch_size=8,
                pooling_kernel_size=2,
                default_output_length=4,
            )
            if with_vision
            else None,
        )
        self.vision_tower = nn.Linear(hidden_size, hidden_size) if with_vision else None
        self.embed_vision = nn.Linear(hidden_size, hidden_size) if with_vision else None
        self.prefill_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []
        self.row_scripts: dict[int, list[int]] = {}
        self.row_cursors: dict[int, int] = {}
        self.default_token_id: int = 1  # <eos>

    def get_input_embeddings(self) -> DummyEmbedding:
        return self.embed

    def resize_token_embeddings(self, new_num: int) -> None:
        if new_num != self.embed.num_embeddings:
            self.embed = DummyEmbedding(new_num, self.hidden_size)
            self.vocab_size = new_num

    def set_script(self, row_idx: int, token_ids: list[int]) -> None:
        self.row_scripts[row_idx] = list(token_ids)
        self.row_cursors[row_idx] = 0

    def _next_token_for_row(self, row_idx: int) -> int:
        script = self.row_scripts.get(row_idx, [])
        cur = self.row_cursors.get(row_idx, 0)
        if cur < len(script):
            self.row_cursors[row_idx] = cur + 1
            return script[cur]
        return self.default_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> DummyModelOutput:
        is_prefill = past_key_values is None
        if input_ids is not None:
            B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
            dev = input_ids.device
        else:
            B, S = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
            dev = inputs_embeds.device

        record = {
            "input_ids": input_ids.clone() if input_ids is not None else None,
            "inputs_embeds": inputs_embeds.clone() if inputs_embeds is not None else None,
            "attention_mask": attention_mask.clone() if attention_mask is not None else None,
            "position_ids": position_ids.clone() if position_ids is not None else None,
            "cache_position": cache_position.clone() if cache_position is not None else None,
            "pixel_values": pixel_values.clone() if pixel_values is not None else None,
            "logits_to_keep": logits_to_keep,
            "seq_len": S,
        }
        if is_prefill:
            self.prefill_calls.append(record)
            cache = DummyDynamicCache(B, S, self.hidden_size, device=dev)
        else:
            self.step_calls.append(record)
            prev_len = past_key_values.layers[0].keys.shape[2]
            cache = DummyDynamicCache(B, prev_len + S, self.hidden_size, device=dev)

        logits = torch.full((B, 1, self.vocab_size), -50.0, dtype=torch.float32, device=dev)
        hidden = torch.zeros((B, 1, self.hidden_size), dtype=torch.float32, device=dev)
        for r in range(B):
            tok = self._next_token_for_row(r)
            logits[r, 0, tok] = 50.0
            hidden[r, 0] = self.embed.weight[tok % self.embed.num_embeddings]
        return DummyModelOutput(past_key_values=cache, logits=logits, hidden_states=(hidden,))


# ===========================================================================
# 1. TestM2Sampling (F4 & F5)
# ===========================================================================
class TestM2Sampling(unittest.TestCase):
    def test_sample_rows_greedy_vectorized(self) -> None:
        """Verify temperature<=0.0 uses vectorized argmax + single .tolist() without multinomial."""
        logits = torch.tensor(
            [
                [1.0, 5.0, 9.0, 2.0],
                [8.0, 3.0, 1.0, 0.0],
                [0.0, 7.0, 2.0, 4.0],
            ],
            dtype=torch.float32,
        )
        blocked = torch.zeros((3, 4), dtype=torch.bool)
        blocked[0, 2] = True  # block token 2 on row 0 -> should pick token 1 (val 5.0)

        lp = token_log_probs(logits, temperature=0.0, top_p=0.95, top_k=2, blocked_rows=blocked)
        self.assertEqual(float(lp[0, 1].item()), 0.0)
        self.assertTrue(math.isinf(float(lp[0, 2].item())))

        with mock.patch.object(torch, "multinomial", wraps=torch.multinomial) as mock_multi:
            res = sample_rows(lp, [2, 0], rng=None, temperature=0.0)
            self.assertEqual(mock_multi.call_count, 0)
            self.assertEqual(res, {0: 1, 2: 1})

            # Also verify greedy fast path when temperature=None on deterministic 0/-inf log_probs
            res_auto = sample_rows(lp, [0, 1, 2], rng=None)
            self.assertEqual(mock_multi.call_count, 0)
            self.assertEqual(res_auto, {0: 1, 1: 0, 2: 1})

    def test_sample_rows_stochastic_batched_multinomial(self) -> None:
        """Verify stochastic sampling calls torch.multinomial ONCE on the 2D batch and handles empty rows."""
        self.assertEqual(sample_rows(torch.zeros((2, 4)), [], rng=None, temperature=0.7), {})

        logits = torch.tensor(
            [
                [0.0, 10.0, -10.0, -10.0],
                [-10.0, -10.0, 10.0, 0.0],
                [-10.0, -10.0, -10.0, 10.0],
            ],
            dtype=torch.float32,
        )
        blocked = torch.zeros((3, 4), dtype=torch.bool)
        lp = token_log_probs(logits, temperature=0.7, top_p=0.9, top_k=2, blocked_rows=blocked)

        rng = torch.Generator(device="cpu").manual_seed(123)
        with mock.patch.object(torch, "multinomial", wraps=torch.multinomial) as mock_multi:
            out = sample_rows(lp, [0, 1, 2], rng=rng, temperature=0.7)
            self.assertEqual(mock_multi.call_count, 1)
            called_probs = mock_multi.call_args[0][0]
            self.assertEqual(called_probs.shape, (3, 4))
            self.assertEqual(out, {0: 1, 1: 2, 2: 3})

    def test_expand_cache_batch_and_multi_token_forward_step(self) -> None:
        """Verify expand_cache_batch supports 1->B and M->M*K, and forward_step supports K>1 tokens."""
        cache = DummyDynamicCache(batch_size=2, seq_len=5, hidden_size=16)
        expand_cache_batch(cache, batch_size=6)
        self.assertEqual(cache.layers[0].keys.shape[0], 6)
        self.assertEqual(cache.layers[0].batch_size, 6)

        backbone = ScriptedBackbone(vocab_size=32, hidden_size=16)
        backbone.set_script(0, [14])
        backbone.set_script(1, [17])
        step_cache = DummyDynamicCache(batch_size=2, seq_len=4, hidden_size=16)
        embeds = torch.zeros((2, 3, 16), dtype=torch.float32)
        attn_mask = torch.ones((2, 7), dtype=torch.long)
        pos_ids = torch.tensor([[4, 5, 6], [4, 5, 6]], dtype=torch.long)
        cache_pos = torch.tensor([4, 5, 6], dtype=torch.long)

        new_cache, next_logits, last_hidden = forward_step(
            backbone,
            input_embeds=embeds,
            past_key_values=step_cache,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            cache_position=cache_pos,
        )
        self.assertEqual(next_logits.shape, (2, 32))
        self.assertEqual(last_hidden.shape, (2, 16))
        self.assertEqual(int(next_logits[0].argmax().item()), 14)
        self.assertEqual(int(next_logits[1].argmax().item()), 17)
        self.assertEqual(backbone.step_calls[-1]["logits_to_keep"], 1)


# ===========================================================================
# 2. TestM2Readout (F5)
# ===========================================================================
class TestM2Readout(unittest.TestCase):
    def test_bf16_bank_cache_and_nearest_token_ids(self) -> None:
        """Verify _BANK_CACHE stores bfloat16 normalized embeddings and nearest_token_ids works on 1D/2D/empty."""
        _BANK_CACHE.clear()
        embed = DummyEmbedding(vocab_size=32, hidden_size=16)
        bank = normalized_embeddings(embed.weight)
        self.assertEqual(bank.dtype, torch.bfloat16)
        self.assertEqual(bank.shape, (32, 16))
        self.assertEqual(len(_BANK_CACHE), 1)

        # 2D lookup
        queries = torch.stack([embed.weight[5], embed.weight[12]], dim=0)
        nearest = nearest_token_ids(queries, bank)
        self.assertEqual(nearest.tolist(), [5, 12])

        # 1D lookup
        single = nearest_token_ids(embed.weight[7], bank)
        self.assertEqual(int(single.item()), 7)

        # Empty lookup
        empty = nearest_token_ids(torch.zeros((0, 16)), bank)
        self.assertEqual(empty.numel(), 0)

        # Top-k offline readout
        tok = DummyTokenizer()
        ro = latent_readout(queries, embed.weight, tok, top_k=2)
        self.assertEqual(len(ro), 2)
        self.assertEqual(ro[0][0][1], 5)
        self.assertEqual(ro[1][0][1], 12)

    def test_lazy_readout_disabled_vs_enabled(self) -> None:
        """Verify record_latent_readout=False never builds _readout_bank or calls nearest_token_ids,
        while record_latent_readout=True computes nearest_token_ids lazily once at span close."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))

        script = [2, 22, 3, 5, 12, 14, 13, 1]

        # 1. record_latent_readout=False
        backbone.set_script(0, script)
        eng_off = HybridGenerationEngine(
            model,
            tok,
            "What is 6*7?",
            min_latent_steps=1,
            max_latent_steps=8,
            temperature=0.0,
            record_latent_readout=False,
        )
        with mock.patch("hybrid_agent.engine.nearest_token_ids", wraps=nearest_token_ids) as mock_nearest:
            eng_off.run()
            self.assertEqual(mock_nearest.call_count, 0)
            self.assertIsNone(eng_off._readout_bank)
            res_off = eng_off.row_result(eng_off.rows[0])
            self.assertNotIn("latent_readouts", res_off)
            self.assertEqual(res_off["latent_steps"], 1)
            self.assertEqual(res_off["anchor_exits"], 1)

        # 2. record_latent_readout=True: 2 latent steps before anchor exit -> only 1 call to nearest_token_ids
        backbone.set_script(0, [2, 22, 23, 3, 5, 12, 14, 13, 1])
        eng_on = HybridGenerationEngine(
            model,
            tok,
            "What is 6*7?",
            min_latent_steps=2,
            max_latent_steps=8,
            temperature=0.0,
            record_latent_readout=True,
        )
        with mock.patch("hybrid_agent.engine.nearest_token_ids", wraps=nearest_token_ids) as mock_nearest:
            eng_on.run()
            self.assertEqual(mock_nearest.call_count, 1)
            self.assertEqual(mock_nearest.call_args[0][0].shape, (2, 16))
            res_on = eng_on.row_result(eng_on.rows[0])
            self.assertEqual(len(res_on["latent_readouts"]), 1)
            self.assertEqual(res_on["latent_readouts"][0]["steps"], 2)
            self.assertEqual(res_on["latent_readouts"][0]["exit"], "anchor")
            self.assertIn("<latent>", res_on["output_rendered"])

        # 3. Unclosed latent span at budget termination + record_hidden & record_token_logprobs
        backbone.set_script(0, [2, 22, 23, 22])
        eng_budget = HybridGenerationEngine(
            model,
            tok,
            "Budget test?",
            max_response_tokens=3,
            min_latent_steps=5,
            max_latent_steps=10,
            temperature=0.0,
            record_latent_readout=True,
            record_hidden=True,
            record_token_logprobs=True,
        )
        eng_budget.run()
        res_b = eng_budget.row_result(eng_budget.rows[0])
        self.assertEqual(res_b["end_reason"], "budget")
        self.assertEqual(len(res_b["latent_readouts"]), 1)
        self.assertEqual(res_b["latent_readouts"][0]["steps"], 2)
        self.assertEqual(res_b["latent_readouts"][0]["exit"], "budget")
        self.assertEqual(len(res_b["latent_vectors"]), 2)
        self.assertEqual(len(res_b["token_logprobs"]), 1)


# ===========================================================================
# 3. TestM2StepOptimizations (F5)
# ===========================================================================
class TestM2StepOptimizations(unittest.TestCase):
    def test_preallocated_attn_mask_and_gated_row_tail(self) -> None:
        """Verify _attn_mask_buf avoids torch.cat and _row_tail is only called on '>' tokens."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))

        # Emit several normal non-tag words (22, 23, 17, 14) which have no '>' character,
        # followed by </think> (5), <answer> (12), 42 (17), </answer> (13), <eos> (1).
        backbone.set_script(0, [22, 23, 17, 14, 5, 12, 17, 13, 1])
        eng = HybridGenerationEngine(
            model,
            tok,
            "Test optimizations",
            temperature=0.0,
        )
        with mock.patch.object(eng, "_row_tail", wraps=eng._row_tail) as spy_tail:
            eng.run()
            self.assertLessEqual(spy_tail.call_count, 2)
            self.assertIsNotNone(eng._attn_mask_buf)
            self.assertEqual(eng.attn_mask.data_ptr(), eng._attn_mask_buf.data_ptr())
            self.assertEqual(eng.attn_mask.shape, (1, eng.global_len))


# ===========================================================================
# 4. TestM2MultiTokenFeed (F5)
# ===========================================================================
class TestM2MultiTokenFeed(unittest.TestCase):
    def test_single_row_and_mixed_batch_multi_token_feed(self) -> None:
        """Verify tool <result> observations with K>1 tokens are prefilled in a single forward_step
        for both all-FEED and mixed-mode batches while preserving kv_ledger and hooks."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))

        tool_exec = mock.MagicMock(spec=ToolExecutor)
        tool_exec.execute.return_value = "obs_tok_1 obs_tok_2 obs_tok_3 obs_tok_4"

        backbone.set_script(0, [5, 6, 16, 7, 4, 5, 12, 14, 13, 1])

        fed_hook_counts = [0]

        class HookedEngine(HybridGenerationEngine):
            def _on_env_token_fed(self, row: RowState) -> None:
                fed_hook_counts[0] += 1

        eng = HookedEngine(
            model,
            tok,
            "Compute 6*7 with python",
            tool_executor=tool_exec,
            temperature=0.0,
        )
        eng.run()

        multi_steps = [c for c in backbone.step_calls if c["seq_len"] > 1]
        self.assertEqual(len(multi_steps), 1)
        feed_len = multi_steps[0]["seq_len"]
        self.assertGreaterEqual(feed_len, 4)
        self.assertEqual(fed_hook_counts[0], feed_len)
        self.assertEqual(eng.rows[0].python_calls, 1)
        self.assertIn("\\boxed{42}", eng.rows[0].full_output)

        # Also test mixed-mode batch where Row 0 is in FEED (with 4 queued tokens) and Row 1 is in DECODE
        backbone2 = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        backbone2.set_script(0, [14, 1])
        backbone2.set_script(1, [17, 1])
        model2 = HybridAgentModel(backbone2, HybridLink(16))
        eng2 = HybridGenerationEngine(
            model2,
            tok,
            ["Question 0", "Question 1"],
            temperature=0.0,
        )
        eng2._init_generation()
        pos1_before = int(eng2.positions[1].item())
        logits1_before = eng2.next_logits[1].clone()

        eng2.rows[0].mode = FEED
        eng2.rows[0].feed_queue = [18, 19, 20, 21]
        eng2._decode_step(rng=None)

        self.assertEqual(eng2.rows[0].mode, DECODE)
        self.assertEqual(len(eng2.rows[0].feed_queue), 0)
        self.assertEqual(int(eng2.positions[1].item()), pos1_before)
        self.assertTrue(bool((eng2.next_logits[1] == logits1_before).all()))


# ===========================================================================
# 5. TestM2MultiQuestionEngine (F4)
# ===========================================================================
class TestM2MultiQuestionEngine(unittest.TestCase):
    def test_multi_question_left_padded_prefill_and_lockstep(self) -> None:
        """Verify multi-question batch with different prompt lengths left-pads prefill and tracks per-row positions."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))

        backbone.set_script(0, [5, 12, 24, 13, 1])
        backbone.set_script(1, [5, 12, 25, 13, 1])
        backbone.set_script(2, [5, 12, 26, 13, 1])

        questions = [
            "Short?",
            "A much longer second question with many distinct words in it?",
            "Medium length question here?",
        ]
        eng = HybridGenerationEngine(model, tok, questions, temperature=0.0)
        self.assertEqual(eng.B, 3)
        self.assertTrue(eng.is_multi_question)

        eng.run()
        self.assertEqual(len(backbone.prefill_calls), 1)
        prefill = backbone.prefill_calls[0]
        self.assertEqual(prefill["input_ids"].shape[0], 3)

        mask0 = prefill["attention_mask"][0].tolist()
        mask1 = prefill["attention_mask"][1].tolist()
        self.assertEqual(mask0[0], 0)
        self.assertEqual(mask0[-1], 1)
        self.assertTrue(all(m == 1 for m in mask1))
        self.assertLess(eng.rows[0].prompt_len, eng.rows[1].prompt_len)

        r0 = eng.row_result(eng.rows[0])
        r1 = eng.row_result(eng.rows[1])
        r2 = eng.row_result(eng.rows[2])
        self.assertIn("\\boxed{A}", r0["output"])
        self.assertIn("\\boxed{B}", r1["output"])
        self.assertIn("\\boxed{C}", r2["output"])

    def test_dual_signature_and_single_question_batch_rollouts(self) -> None:
        """Verify both legacy 6-positional signature and PROJECT.md 3-positional signature work."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))
        tool_exec = mock.MagicMock(spec=ToolExecutor)

        eng_new = HybridGenerationEngine(model, tok, "Single question?", batch_size=2, temperature=0.0)
        self.assertEqual(eng_new.B, 2)
        self.assertEqual(eng_new.max_response_tokens, 8192)
        self.assertEqual(eng_new.max_latent_steps, 32)
        self.assertEqual(eng_new.min_latent_steps, 0)
        self.assertEqual(eng_new.max_rounds, 8)

        eng_leg = HybridGenerationEngine(
            model,
            tok,
            torch.device("cpu"),
            tool_exec,
            "Custom sys prompt",
            "Legacy question?",
            batch_size=3,
            temperature=0.0,
        )
        self.assertEqual(eng_leg.B, 3)
        self.assertEqual(eng_leg.system_prompt, "Custom sys prompt")
        self.assertEqual(eng_leg.question, "Legacy question?")
        self.assertIs(eng_leg.tool_executor, tool_exec)
        eng_leg.run()
        self.assertEqual(len(eng_leg.rows), 3)

    def test_hybrid_engine_run_and_run_batch_ordering(self) -> None:
        """Verify HybridEngine.run and HybridEngine.run_batch preserve exact input order across batches."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16)
        model = HybridAgentModel(backbone, HybridLink(16))

        he = HybridEngine(model, tok, temperature=0.0, default_batch_size=2)

        backbone.set_script(0, [5, 12, 14, 13, 1])
        traj = he.run("What is 6*7?")
        self.assertIsInstance(traj, HybridTrajectory)
        self.assertEqual(traj.final_answer, "42")

        questions = [f"Question number {i} with length {'x' * (i * 5)}" for i in range(5)]
        trajs = he.run_batch(questions, batch_size=2, max_concurrency=2, sort_by_length=True)
        self.assertEqual(len(trajs), 5)
        for i, t in enumerate(trajs):
            self.assertEqual(t.raw_result["question"], questions[i])


# ===========================================================================
# 6. TestM2MultimodalVision (F6)
# ===========================================================================
class TestM2MultimodalVision(unittest.TestCase):
    def test_load_hybrid_agent_processor_fallback_and_vision_preservation(self) -> None:
        """Verify load_hybrid_agent preserves vision_tower/embed_vision and synthesizes processor when processor_config.json is absent."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16, with_vision=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "vision_config": {
                    "image_size": 32,
                    "patch_size": 8,
                    "pooling_kernel_size": 2,
                    "default_output_length": 4,
                },
            }
            (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            link = HybridLink(16)
            torch.save(link.state_dict(), tmp_path / "adapter.pt")
            (tmp_path / "adapter_config.json").write_text(
                json.dumps(HYBRIDLINK_CONFIG), encoding="utf-8"
            )

            with (
                mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}),
                mock.patch("hybrid_agent.model.AutoTokenizer.from_pretrained", return_value=tok),
                mock.patch(
                    "transformers.Gemma4ForConditionalGeneration.from_pretrained",
                    return_value=backbone,
                    create=True,
                ) as mock_gemma4,
                mock.patch("hybrid_agent.backends.patch_bf16_experts_fastpath", return_value=0),
                mock.patch("hybrid_agent.backends.enable_tensorrt_acceleration") as mock_trt,
            ):
                model, loaded_tok = load_hybrid_agent(
                    tmp_path,
                    device="cpu",
                    backend="tensorrt",
                    max_batch_size=128,
                    load_processor=True,
                )
                self.assertEqual(mock_gemma4.call_count, 1)
                self.assertTrue(model.has_vision)
                self.assertIsNotNone(model.vision_tower)
                self.assertIsNotNone(model.embed_vision)
                self.assertIsNotNone(model.processor)
                mock_trt.assert_called_once()
                self.assertEqual(mock_trt.call_args.kwargs.get("max_batch_size"), 128)

                pil_img = Image.new("RGB", (48, 48), color=(120, 200, 80))
                proc_out = model.processor(text="Describe this image", images=pil_img, return_tensors="pt")
                self.assertIn("pixel_values", proc_out)
                self.assertEqual(proc_out["pixel_values"].shape, (1, 3, 32, 32))

    def test_multimodal_images_in_engine_run_and_run_batch(self) -> None:
        """Verify PIL images and tensor pixel_values are passed through HybridEngine.run and run_batch into backbone prefill."""
        tok = DummyTokenizer()
        backbone = ScriptedBackbone(vocab_size=len(tok), hidden_size=16, with_vision=True)
        proc = HybridMultimodalProcessor(tok, image_size=32, patch_size=8)
        model = HybridAgentModel(backbone, HybridLink(16), processor=proc)

        he = HybridEngine(model, tok, processor=proc, temperature=0.0, default_batch_size=3)

        img1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
        backbone.set_script(0, [5, 12, 24, 13, 1])
        traj = he.run("What color is the image?", images=img1)
        self.assertEqual(traj.final_answer, "A")
        self.assertIsNotNone(backbone.prefill_calls[-1]["pixel_values"])
        self.assertEqual(backbone.prefill_calls[-1]["pixel_values"].shape, (1, 3, 32, 32))

        img2 = Image.new("RGB", (32, 32), color=(0, 255, 0))
        img3 = Image.new("RGB", (32, 32), color=(0, 0, 255))
        backbone.set_script(0, [5, 12, 24, 13, 1])
        backbone.set_script(1, [5, 12, 25, 13, 1])
        backbone.set_script(2, [5, 12, 26, 13, 1])

        trajs = he.run_batch(
            ["Image Q1", "Text-only Q2", "Two-image Q3"],
            images=[img1, None, [img2, img3]],
            batch_size=3,
        )
        self.assertEqual([t.final_answer for t in trajs], ["A", "B", "C"])
        self.assertIsNotNone(backbone.prefill_calls[-1]["pixel_values"])
        self.assertEqual(backbone.prefill_calls[-1]["pixel_values"].shape, (3, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
