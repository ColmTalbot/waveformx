# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The models as PyTorch functions: a thin wrapper around the JAX models.

:func:`polarizations_function` and :func:`td_polarizations_function` return
functions of a :class:`~waveformx.Binary` of torch tensors (or a mapping of
its fields) and a tensor of frequencies (or times) that give ``(hplus,
hcross)`` as torch tensors. The waveform is computed by the compiled JAX
model, and its derivatives by the compiled ``jax.vjp`` of the model, so
``loss.backward()`` and ``torch.autograd.grad`` reach every field that
requires a gradient (and the grid, for the models that can be
differentiated with respect to it, such as TaylorF2 and IMRPhenomXAS).

Tensors pass between PyTorch and JAX through DLPack, without a copy when
both live on a device that JAX also has (the CPU, or a CUDA GPU with a CUDA
build of JAX); otherwise (Apple's ``mps``, say) they are copied through the
CPU and the result is returned on the device of the inputs.

The precision follows the inputs: if every floating-point tensor among them
is float32 the model runs in float32 (complex64 output), else in float64.
Python numbers and sequences are accepted for any field. A leading batch
axis of the fields (``mass_1`` of shape ``(n,)``, ``spin_1`` of shape ``(n,
3)``; the other fields are broadcast) maps the model over the binaries with
``jax.vmap``.

The functions are opaque to ``torch.compile`` and ``torch.func`` (``vmap``,
``jacrev``, ...), give first derivatives only, and make no parameter
checks (as :func:`waveformx.polarizations_function`).

>>> import torch
>>> import waveformx.torch
>>> f = waveformx.torch.polarizations_function("IMRPhenomXAS")
>>> mass_1 = torch.tensor(36.0, dtype=torch.float64, requires_grad=True)
>>> binary = waveformx.Binary(mass_1, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.4), 500.0, 0.7, 1.2, 20.0)
>>> hplus, hcross = f(binary, torch.linspace(20.0, 1024.0, 4017, dtype=torch.float64))
>>> (hplus.abs() ** 2).sum().backward()
>>> mass_1.grad
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch

from . import _api
from ._binary import BINARY_NAMES, Binary
from ._options import WaveformOptions, parse_options

__all__ = ["polarizations_function", "td_polarizations_function", "to_jax", "to_torch"]


def to_jax(tensor: torch.Tensor) -> jax.Array:
    """A torch tensor as a JAX array: shared through DLPack where possible,
    else copied through the CPU."""
    tensor = tensor.detach().resolve_conj().contiguous()
    try:
        return jnp.from_dlpack(tensor)
    except Exception:  # a device that JAX does not have
        return jnp.asarray(tensor.cpu().numpy())


def to_torch(array: jax.Array, device: Optional[torch.device] = None) -> torch.Tensor:
    """A JAX array as a torch tensor on ``device`` (the array's own by
    default): shared through DLPack where possible, else copied."""
    try:
        tensor = torch.from_dlpack(array)
    except Exception:
        tensor = torch.from_numpy(np.array(array))
    return tensor if device is None else tensor.to(device)


@functools.lru_cache(maxsize=None)
def _compiled(approximant: str, options: WaveformOptions, method: str, batched: bool):
    """The compiled model and the compiled ``jax.vjp`` of it with respect to
    the binary (and the grid if ``wrt_grid``), cached per options;
    ``jax.jit`` caches them per shape and dtype."""
    function = functools.partial(getattr(_api.models.get(approximant), method), options=options)
    if batched:
        function = jax.vmap(function, in_axes=(0, None))

    def vjp(binary, grid, cotangents, wrt_grid):
        if wrt_grid:
            return jax.vjp(function, binary, grid)[1](cotangents)
        # (not every model can be differentiated with respect to its grid)
        return jax.vjp(lambda b: function(b, grid), binary)[1](cotangents)[0], None

    return jax.jit(function), jax.jit(vjp, static_argnums=3)


class _Model(torch.autograd.Function):
    """``(hplus, hcross)`` of the grid and the binary's fields, with the
    backward pass of ``jax.vjp`` (which evaluates the model again)."""

    @staticmethod
    def forward(ctx, key, device, grid, *fields):
        function, _ = _compiled(*key)
        ctx.key, ctx.device = key, device
        ctx.save_for_backward(grid, *fields)
        hplus, hcross = function(Binary(*map(to_jax, fields)), to_jax(grid))
        return to_torch(hplus, device), to_torch(hcross, device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_hplus, grad_hcross):
        _, vjp = _compiled(*ctx.key)
        grid, *fields = ctx.saved_tensors
        # PyTorch's gradient with respect to z = a + ib is dL/da + i dL/db,
        # JAX's cotangent of z is its conjugate
        cotangents = tuple(to_jax(g.conj() if g.is_complex() else g) for g in (grad_hplus, grad_hcross))
        wrt_grid = ctx.needs_input_grad[2]
        d_binary, d_grid = vjp(Binary(*map(to_jax, fields)), to_jax(grid), cotangents, wrt_grid)
        grads = [None if d is None else to_torch(d, ctx.device) for d in (d_grid, *d_binary)]
        return (None, None, *(g if need else None for g, need in zip(grads, ctx.needs_input_grad[2:])))


def _fields(binary: Any) -> list:
    if isinstance(binary, Binary):
        return list(binary)
    if isinstance(binary, Mapping):
        unknown = set(binary) - set(BINARY_NAMES)
        if unknown:
            raise TypeError(f"unknown fields of a Binary: {sorted(unknown)}")
        missing = [name for name in BINARY_NAMES[:4] if name not in binary]
        if missing:
            raise TypeError(f"missing fields of a Binary: {missing}")
        return [binary.get(name, Binary._field_defaults.get(name)) for name in BINARY_NAMES]
    raise TypeError(f"binary must be a waveformx.Binary or a mapping of its fields, got {type(binary).__name__}")


def _tensors(value: Any) -> list:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in _tensors(item)]
    return []


def _as_tensor(value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, (list, tuple)) and _tensors(value):
        # a spin of tensors: stacked, so that the gradient reaches them
        return torch.stack([_as_tensor(x, dtype, device) for x in value])
    return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=dtype, device=device)


def _wrap(approximant: str, options: WaveformOptions, method: str) -> Callable:
    def function(binary: Any, grid: Any):
        values = [*_fields(binary), grid]
        tensors = [t for v in values for t in _tensors(v)]
        floats = [t for t in tensors if t.is_floating_point()]
        dtype = torch.float32 if floats and all(t.dtype == torch.float32 for t in floats) else torch.float64
        device = grid.device if isinstance(grid, torch.Tensor) else tensors[0].device if tensors else torch.device("cpu")
        *fields, grid = (_as_tensor(v, dtype, device) for v in values)
        sizes = {f.shape[0] for i, f in enumerate(fields) if f.ndim > (1 if i in (2, 3) else 0)}
        if len(sizes) > 1:
            raise ValueError(f"the batch axes of the binary's fields differ: {sorted(sizes)}")
        batched = bool(sizes)
        if batched:
            (n,) = sizes
            fields = [torch.broadcast_to(f, (n, 3) if i in (2, 3) else (n,)) for i, f in enumerate(fields)]
        return _Model.apply((approximant, options, method, batched), device, grid, *fields)

    function.__name__ = method
    function.__qualname__ = f"{approximant}.{method}"
    return function


def polarizations_function(approximant: str, options: Optional[WaveformOptions] = None, **option_keywords) -> Callable:
    """The PyTorch function ``f(binary, frequencies) -> (hplus, hcross)`` of a
    frequency-domain approximant (see :func:`waveformx.polarizations_function`):
    ``binary`` is a :class:`~waveformx.Binary` or a mapping of its fields,
    as tensors (with an optional leading batch axis), numbers or sequences;
    the polarizations are complex tensors of shape ``(len(frequencies),)``,
    or ``(n, len(frequencies))`` for a batch."""
    _api._model(approximant, "frequency")
    return _wrap(approximant, parse_options(options, option_keywords), "polarizations")


def td_polarizations_function(approximant: str, options: Optional[WaveformOptions] = None, **option_keywords) -> Callable:
    """The PyTorch function ``f(binary, times) -> (hplus, hcross)`` of a
    time-domain approximant at ``times`` (s, relative to the peak; see
    :func:`waveformx.td_polarizations_function`), as real tensors."""
    _api._model(approximant, "time")
    return _wrap(approximant, parse_options(options, option_keywords), "polarizations_at")
