# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pure function of the lean IMRPhenomXPNR in MLX: binaries in, the
polarizations out, with derivatives for ``mx.grad`` and ``mx.vjp``.

The derivatives are those of the forward-mode Jacobian with respect to the
twelve parameters of each binary (:func:`._pipeline.evaluate` with the
tangents), contracted with the cotangents by a custom VJP (an
``mx.custom_function``), as the JAX model contracts its forward-mode
Jacobian. ``mx.jvp`` and ``mx.vmap`` do not apply (the function takes the
batch itself).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from ...._options import STRAIN_SCALE, WaveformOptions
from .. import model
from . import _pipeline

__all__ = ["batched", "polarizations", "polarizations_and_jacobian"]


def _as_float32(value):
    if isinstance(value, mx.array):
        return value if value.dtype == mx.float32 else value.astype(mx.float32)
    return mx.array(np.asarray(value, dtype=np.float32))


def batched(binary):
    """``binary`` with float32 MLX fields broadcast to a leading batch axis
    (the scalars to ``(B,)``, the spins to ``(B, 3)``), and whether it had
    one."""
    fields = [_as_float32(value) for value in binary]
    spin = [name in ("spin_1", "spin_2") for name in binary._fields]
    sizes = {int(value.shape[0]) for value, s in zip(fields, spin) if value.ndim == (2 if s else 1)}
    if len(sizes) > 1:
        raise ValueError(f"the fields of the binary have different batch sizes {sorted(sizes)}")
    size = next(iter(sizes), 1)
    fields = [mx.broadcast_to(value, (size, 3) if s else (size,)) for value, s in zip(fields, spin)]
    return type(binary)(*fields), bool(sizes)


def _parameters(binary):
    """The twelve parameters of each binary, ``(B, 12)`` (differentiable)."""
    b = binary
    return mx.concatenate([b.mass_1[:, None], b.mass_2[:, None], b.spin_1, b.spin_2, b.luminosity_distance[:, None],
                           b.inclination[:, None], b.phi_ref[:, None], b.f_ref[:, None]], axis=1)


def _leaves(params):
    return {"mass_1": params[:, 0], "mass_2": params[:, 1], "spin_1": params[:, 2:5], "spin_2": params[:, 5:8],
            "luminosity_distance": params[:, 8], "inclination": params[:, 9], "phi_ref": params[:, 10],
            "f_ref": params[:, 11]}


def _frequencies(frequencies, batch):
    f = _as_float32(frequencies)
    if f.ndim == 1:
        f = f[None, :]
    if f.shape[0] not in (1, batch):
        raise ValueError(f"frequencies of shape {f.shape} for {batch} binaries")
    return mx.stop_gradient(f)


@functools.lru_cache(maxsize=None)
def _core(modes, amp_scale, reference_mf):
    """``f(params (B, 12), frequencies) -> (B, N, 4)`` with its custom VJP."""

    @mx.custom_function
    def core(params, frequencies):
        h, _, _ = _pipeline.evaluate_chunked(_leaves(params), frequencies, modes, amp_scale,
                                             reference_mf=reference_mf)
        return h

    @core.vjp
    def core_vjp(primals, cotangent, output):
        params, frequencies = primals
        _, _, jacobian = _pipeline.evaluate_chunked(_leaves(mx.stop_gradient(params)), frequencies, modes,
                                                    amp_scale, _pipeline.N_PARAMETERS, reference_mf)
        if isinstance(cotangent, (list, tuple)):
            cotangent = cotangent[0]
        terms = jacobian * cotangent[..., None]
        # (no contribution where the cotangent vanishes, whatever the
        # Jacobian, as in reverse mode)
        terms = mx.where((cotangent == 0)[..., None], 0.0, terms)
        return mx.sum(terms, axis=(1, 2)), mx.zeros_like(frequencies)

    return core


def _options(mode_array, scaled_strain, reference_mf):
    # (the checks of WaveformOptions)
    reference_mf = WaveformOptions(lean=True, reference_mf=reference_mf).reference_mf
    modes = model.coprecessing_modes(mode_array)
    return tuple(modes), STRAIN_SCALE if scaled_strain else 1.0, reference_mf


def polarizations(binary, frequencies, mode_array=None, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` (complex64 MLX arrays) of the lean IMRPhenomXPNR at
    ``frequencies`` (Hz, positive and increasing; ``(N,)`` or, for a batch,
    ``(B, N)``): ``(N,)`` each for one binary, ``(B, N)`` for a batch (a
    :class:`~waveformjax.Binary` whose fields have a leading axis). The
    strain is multiplied by ``1e21`` unless ``scaled_strain`` is false;
    NaN where the model fails. Differentiable with ``mx.grad`` with
    respect to the fields of the binary. With ``reference_mf``
    (``WaveformOptions.reference_mf``), the binary is given at the
    mass-scaled reference frequency ``reference_mf / M`` (``f_ref`` zero)."""
    binary, has_batch = batched(binary)
    params = _parameters(binary)
    f = _frequencies(frequencies, params.shape[0])
    h = _core(*_options(mode_array, scaled_strain, reference_mf))(params, f)
    hp = h[..., 0] + 1j * h[..., 1]
    hc = h[..., 2] + 1j * h[..., 3]
    return (hp, hc) if has_batch else (hp[0], hc[0])


def polarizations_and_jacobian(binary, frequencies, mode_array=None, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their forward-mode Jacobian with respect to the
    twelve parameters of the binary (``mass_1``, ``mass_2``, ``spin_1``,
    ``spin_2``, ``luminosity_distance``, ``inclination``, ``phi_ref``,
    ``f_ref``): each ``(..., N, 12)`` complex64."""
    binary, has_batch = batched(binary)
    params = mx.stop_gradient(_parameters(binary))
    f = _frequencies(frequencies, params.shape[0])
    modes, amp_scale, reference_mf = _options(mode_array, scaled_strain, reference_mf)
    h, _, jacobian = _pipeline.evaluate_chunked(_leaves(params), f, modes, amp_scale, _pipeline.N_PARAMETERS,
                                                reference_mf)
    hp = h[..., 0] + 1j * h[..., 1]
    hc = h[..., 2] + 1j * h[..., 3]
    jp = jacobian[..., 0, :] + 1j * jacobian[..., 1, :]
    jc = jacobian[..., 2, :] + 1j * jacobian[..., 3, :]
    if has_batch:
        return (hp, hc), (jp, jc)
    return (hp[0], hc[0]), (jp[0], jc[0])
