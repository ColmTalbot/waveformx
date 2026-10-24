# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pure functions of IMRPhenomXHM in MLX: binaries in, the
polarizations (and the modes) out, with derivatives for ``mx.grad`` and
``mx.vjp`` (custom VJPs contracting the forward-mode derivatives with the
cotangents in a kernel, as for IMRPhenomXAS)."""

from __future__ import annotations

import functools

import mlx.core as mx

from ...._options import STRAIN_SCALE
from ...xas.mlx._api import _complex, _inputs, _real
from .. import core
from ._kernels import kernels

__all__ = ["polarizations", "modes", "polarizations_and_jacobian"]


def _grad(grad):
    # (none for the first frequency, which is not a parameter)
    return mx.concatenate([grad, mx.zeros((grad.shape[0], 1), dtype=grad.dtype)], axis=1)


def _cotangents(cotangents, outputs):
    return [(mx.zeros_like(o) if c is None else c).astype(mx.complex64) for c, o in zip(cotangents, outputs)]


@functools.lru_cache(maxsize=None)
def _core(amp_scale):
    """``f(inputs (B, 9), frequencies) -> (h_+, h_x)`` with its custom VJP."""
    k = kernels(amp_scale)

    @mx.custom_function
    def core_(x, frequencies):
        hp, hc = k.polarizations(k.tables(x), frequencies)
        return _complex(hp), _complex(hc)

    @core_.vjp
    def core_vjp(primals, cotangents, outputs):
        x, frequencies = primals
        tables, dtables = k.tables_jvp(mx.stop_gradient(x))
        cot_p, cot_c = _cotangents(cotangents, outputs)
        grad = k.cotangent_product(tables, dtables, frequencies, _real(cot_p), _real(cot_c))
        return _grad(grad), mx.zeros_like(frequencies)

    return core_


@functools.lru_cache(maxsize=None)
def _core_modes(amp_scale):
    """``f(inputs, frequencies) -> (h_+, h_x, modes (B, N, 5))``."""
    k = kernels(amp_scale)

    @mx.custom_function
    def core_(x, frequencies):
        hp, hc, h = k.polarizations(k.tables(x), frequencies, with_modes=True)
        return _complex(hp), _complex(hc), h.view(mx.complex64).reshape(h.shape[:-1])

    @core_.vjp
    def core_vjp(primals, cotangents, outputs):
        x, frequencies = primals
        tables, dtables = k.tables_jvp(mx.stop_gradient(x))
        cot_p, cot_c, cot_m = _cotangents(cotangents, outputs)
        cot_m = cot_m.reshape(cot_m.shape + (1,)).view(mx.float32)
        grad = k.cotangent_product(tables, dtables, frequencies, _real(cot_p), _real(cot_c), cot_m)
        return _grad(grad), mx.zeros_like(frequencies)

    return core_


def _scale(scaled_strain):
    return STRAIN_SCALE if scaled_strain else 1.0


def polarizations(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` (complex64 MLX arrays) of IMRPhenomXHM at
    ``frequencies`` (Hz, positive and increasing; ``(N,)`` or, for a batch,
    ``(B, N)``): ``(N,)`` each for one binary, ``(B, N)`` for a batch (a
    :class:`~waveformjax.Binary` whose fields have a leading axis). Only the
    aligned components of the spins enter. The strain is multiplied by
    ``1e21`` unless ``scaled_strain`` is false; NaN where LAL fails.
    Differentiable with ``mx.grad`` with respect to the fields of the
    binary."""
    x, f, has_batch = _inputs(binary, frequencies)
    hp, hc = _core(_scale(scaled_strain))(x, f)
    return (hp, hc) if has_batch else (hp[0], hc[0])


def modes(binary, frequencies, scaled_strain=True):
    """The modes ``{(l, m): h_{l,-m}(f)}`` (``m > 0``: the modes of
    negative ``m`` at the positive frequencies; ``h_{l,m}(-f) = (-1)^l
    conj(h_{l,-m}(f))``), those whose sum with the spherical harmonics is
    :func:`polarizations` (the (3,2) recycling the (2,2) in its ringdown;
    the higher modes at every frequency). Differentiable with ``mx.grad``."""
    x, f, has_batch = _inputs(binary, frequencies)
    _, _, h = _core_modes(_scale(scaled_strain))(x, f)
    out = {mode: h[..., k] for k, mode in enumerate(core.MODES)}
    return out if has_batch else {mode: value[0] for mode, value in out.items()}


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary: each ``(..., N, 12)`` complex64 (zero for the
    in-plane spins)."""
    x, f, has_batch = _inputs(binary, frequencies)
    k = kernels(_scale(scaled_strain))
    tables, dtables = k.tables_jvp(mx.stop_gradient(x))
    hp, hc, jp, jc = k.polarizations_and_jacobian(tables, dtables, f)
    out = (_complex(hp), _complex(hc)), (_complex(jp), _complex(jc))
    if has_batch:
        return out
    return (out[0][0][0], out[0][1][0]), (out[1][0][0], out[1][1][0])
