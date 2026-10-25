# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pure function of IMRPhenomXAS in MLX: binaries in, the polarizations
out, with derivatives for ``mx.grad`` and ``mx.vjp``.

A call is two kernels: the set-up of the batch, then the waveform at the
frequencies. The derivatives come from a custom VJP (an
``mx.custom_function``): the set-up with its forward-mode derivatives,
then one kernel that contracts the derivatives at each frequency with the
cotangents and sums them over the frequencies. ``mx.jvp`` and ``mx.vmap``
do not apply (the function takes the batch itself).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from ...._options import REFERENCE_MF, STRAIN_SCALE, WaveformOptions
from ...xpnr._math import MTSUN_SI
from ...xpnr.mlx._api import _frequencies, batched
from ._kernels import N_INPUTS, kernels

__all__ = ["polarizations", "polarizations_and_jacobian", "to_mass_scaled_reference", "to_physical_reference"]


def _scaled_reference(m1, m2, f_ref, reference_mf, where):
    """``f_ref`` replaced by the mass-scaled reference ``reference_mf / M``
    (NaN where ``f_ref`` is not zero; ``where`` is ``np.where`` or
    ``mx.where``), as :func:`...xas.model.with_reference`."""
    return where(f_ref == 0.0, reference_mf / (MTSUN_SI * (m1 + m2)), np.nan)


def _host_inputs(binary, frequencies, reference_mf=None):
    """The inputs of the set-up ``(B, 9)``, the frequencies ``(B or 1, N)``
    and whether there is a batch axis, for a binary of numbers or numpy
    arrays: the parameters made on the host (one transfer). ``None`` if
    any field of the binary is an MLX array."""
    if any(isinstance(value, mx.array) for value in binary):
        return None
    fields = [np.asarray(value, dtype=np.float32) for value in binary]
    spin = [name in ("spin_1", "spin_2") for name in binary._fields]
    sizes = {value.shape[0] for value, s in zip(fields, spin) if value.ndim == (2 if s else 1)}
    if len(sizes) > 1:
        raise ValueError(f"the fields of the binary have different batch sizes {sorted(sizes)}")
    batch = next(iter(sizes), 1)
    b = dict(zip(binary._fields, fields))
    f_ref = b["f_ref"]
    if reference_mf is not None:
        f_ref = _scaled_reference(b["mass_1"], b["mass_2"], f_ref, np.float32(reference_mf), np.where)
    columns = [b["mass_1"], b["mass_2"], b["spin_1"][..., 2], b["spin_2"][..., 2], b["luminosity_distance"],
               b["inclination"], b["phi_ref"], f_ref]
    x = np.empty((batch, N_INPUTS - 1), dtype=np.float32)
    for i, column in enumerate(columns):
        x[:, i] = column
    if isinstance(frequencies, mx.array):
        f = _frequencies(frequencies, batch)
        x = mx.concatenate([mx.array(x), mx.broadcast_to(f[:, :1], (batch, 1))], axis=1)
    else:
        f = np.asarray(frequencies, dtype=np.float32)
        f = f[None] if f.ndim == 1 else f
        if f.shape[0] not in (1, batch):
            raise ValueError(f"frequencies of shape {f.shape} for {batch} binaries")
        x = mx.array(np.concatenate([x, np.broadcast_to(f[:, :1], (batch, 1))], axis=1))
        f = mx.array(f)
    return x, f, bool(sizes)


def _inputs(binary, frequencies, reference_mf=None):
    """The inputs of the set-up ``(B, 9)`` (the aligned-spin parameters and
    the first frequency, which ``f_ref = 0`` stands for), differentiable
    with respect to the fields of ``binary``, the frequencies ``(B or 1,
    N)`` and whether the binary had a batch axis. With ``reference_mf``
    (the mass-scaled reference), ``f_ref`` is ``reference_mf / M`` (NaN
    where the binary's is not zero): the derivatives with respect to the
    masses follow it through ``mx.grad``."""
    host = _host_inputs(binary, frequencies, reference_mf)
    if host is not None:
        return host
    binary, has_batch = batched(binary)
    b = binary
    batch = b.mass_1.shape[0]
    f = _frequencies(frequencies, batch)
    f_ref = b.f_ref
    if reference_mf is not None:
        f_ref = _scaled_reference(b.mass_1, b.mass_2, mx.stop_gradient(f_ref), reference_mf, mx.where)
    x = mx.stack([b.mass_1, b.mass_2, b.spin_1[:, 2], b.spin_2[:, 2], b.luminosity_distance, b.inclination,
                  b.phi_ref, f_ref, mx.broadcast_to(f[:, 0], (batch,))], axis=1)
    return x, f, has_batch


def _complex(x):
    """``(B, N, 2)`` float32 as ``(B, N)`` complex64."""
    return x.view(mx.complex64).reshape(x.shape[:-1])


def _real(z):
    """``(B, N)`` complex64 as ``(B, N, 2)`` float32."""
    return z.reshape(z.shape + (1,)).view(mx.float32)


@functools.lru_cache(maxsize=None)
def _core(amp_scale):
    """``f(inputs (B, 9), frequencies) -> (h_+, h_x)`` with its custom VJP."""
    k = kernels(amp_scale)

    @mx.custom_function
    def core(x, frequencies):
        hp, hc = k.polarizations(k.table(x), frequencies)
        return _complex(hp), _complex(hc)

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        x, frequencies = primals
        table, dtable = k.table_jvp(mx.stop_gradient(x))
        cot_p, cot_c = (mx.zeros_like(o) if c is None else c for c, o in zip(cotangents, outputs))
        grad = k.cotangent_product(table, dtable, frequencies, _real(cot_p.astype(mx.complex64)),
                                   _real(cot_c.astype(mx.complex64)))
        # (none for the first frequency, which is not a parameter)
        grad = mx.concatenate([grad, mx.zeros((grad.shape[0], 1), dtype=grad.dtype)], axis=1)
        return grad, mx.zeros_like(frequencies)

    return core


def _scale(scaled_strain):
    return STRAIN_SCALE if scaled_strain else 1.0


def reference(reference_mf):
    """``reference_mf`` checked as ``WaveformOptions`` does (``None``: LAL's
    convention)."""
    return WaveformOptions(reference_mf=reference_mf).reference_mf


def scaled_jacobian(binary, jacobians, reference_mf):
    """The Jacobians ``(..., N, 12)`` of the kernels (with respect to the
    inputs, ``f_ref`` standing for ``reference_mf / M``) as Jacobians with
    respect to the binary: the column of ``f_ref`` added to those of the
    masses (``d f_ref / d m_i = -f_ref / M``) and zero."""
    if reference_mf is None:
        return jacobians
    m1, m2 = (mx.array(np.asarray(v, np.float32)) if not isinstance(v, mx.array) else v
              for v in (binary.mass_1, binary.mass_2))
    total = (m1 + m2)[..., None]
    d = -(reference_mf / MTSUN_SI) / (total * total)
    out = []
    for j in jacobians:
        column = j[..., 11] * d.astype(mx.complex64)
        j = mx.concatenate([(j[..., 0] + column)[..., None], (j[..., 1] + column)[..., None], j[..., 2:11],
                            mx.zeros_like(j[..., 11:])], axis=-1)
        out.append(j)
    return tuple(out)


def polarizations(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` (complex64 MLX arrays) of IMRPhenomXAS at
    ``frequencies`` (Hz, positive and increasing; ``(N,)`` or, for a batch,
    ``(B, N)``): ``(N,)`` each for one binary, ``(B, N)`` for a batch (a
    :class:`~waveformx.Binary` whose fields have a leading axis). Only
    the aligned components of the spins enter. The strain is multiplied
    by ``1e21`` unless ``scaled_strain`` is false; NaN where LAL fails.
    Differentiable with ``mx.grad`` with respect to the fields of the
    binary. With ``reference_mf`` the binary is given at ``M f_ref =
    reference_mf`` (``WaveformOptions.reference_mf``; its ``f_ref`` zero,
    NaN otherwise)."""
    x, f, has_batch = _inputs(binary, frequencies, reference(reference_mf))
    hp, hc = _core(_scale(scaled_strain))(x, f)
    return (hp, hc) if has_batch else (hp[0], hc[0])


def polarizations_and_jacobian(binary, frequencies, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobians with respect to the twelve
    parameters of the binary (``mass_1``, ``mass_2``, ``spin_1``,
    ``spin_2``, ``luminosity_distance``, ``inclination``, ``phi_ref``,
    ``f_ref``): each ``(..., N, 12)`` complex64 (zero for the in-plane
    spins). ``reference_mf`` as for :func:`polarizations`."""
    reference_mf = reference(reference_mf)
    x, f, has_batch = _inputs(binary, frequencies, reference_mf)
    k = kernels(_scale(scaled_strain))
    table, dtable = k.table_jvp(mx.stop_gradient(x))
    hp, hc, jp, jc = k.polarizations_and_jacobian(table, dtable, f)
    jacobians = scaled_jacobian(binary, (_complex(jp), _complex(jc)), reference_mf)
    out = (_complex(hp), _complex(hc)), jacobians
    if has_batch:
        return out
    return (out[0][0][0], out[0][1][0]), (out[1][0][0], out[1][1][0])



def _converters():
    from ...taylorf2.mlx._tables import converters
    from .. import reference as xas_reference

    return converters(xas_reference.to_mass_scaled_reference, xas_reference.to_physical_reference)


def to_mass_scaled_reference(binary, reference_mf=REFERENCE_MF, f_min=None):
    """:func:`..reference.to_mass_scaled_reference` in MLX (float32, the
    fields of the binary MLX arrays or numbers, one binary or a batch):
    differentiable with ``mx.grad``. For IMRPhenomXHM too."""
    return _converters()[0](binary, reference(reference_mf), f_min)


def to_physical_reference(binary, f_ref, reference_mf=REFERENCE_MF):
    """:func:`..reference.to_physical_reference` in MLX (see
    :func:`to_mass_scaled_reference`)."""
    return _converters()[1](binary, f_ref, reference(reference_mf))
