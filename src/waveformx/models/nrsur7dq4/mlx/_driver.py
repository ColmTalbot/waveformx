# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``SimInspiralChooseTDWaveform`` and ``SimInspiralChooseTDModes`` for
NRSur7dq4 in MLX: LAL's output grids, computed on the host in float64 as
:mod:`.._driver` computes them, and the float32 waveform evaluated on them
by the GPU.

The grid of one binary is :func:`.._driver.setup`. The grids of a batch are
computed together (:func:`grids`): the orbital-frequency fits of every
binary in one compiled float64 JAX call, then LAL's search for the start
time, its checks, the number of samples and the epoch in numpy over the
batch. This is what :func:`.._driver.setup` does one binary at a time, at
a cost per binary (~0.15 ms) larger than the GPU's for the whole waveform.
A binary that fails a check is set up again by :func:`.._driver.setup`,
which raises LAL's error.

The waveform is evaluated by a compiled function on a grid of bucketed
length (:func:`.._driver.bucket`), so that waveforms of similar lengths
share one compilation, and trimmed. A batch is sorted by length, so that
each chunk is padded little, and its chunks are bucketed too.
"""

from __future__ import annotations

import functools
from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import mlx.core as mx
import numpy as np

from ...._binary import Binary, check_binary, parameter_table, row
from ...._results import BatchTimeSeries, TimeSeriesModes, TimeSeriesPolarizations
from ....exceptions import InvalidParameterError, ModelError, WaveformError
from .. import _driver as _jax_driver
from .._data import MODES
from .._data import default_data as jax_default_data
from .._dynamics import OMEGA_REF_MAX, OMEGA_REF_ZERO, START_TIME, reference_omegas
from .._waveform import MSUN_SI, MTSUN_SI, output_ells, strain_scale
from . import _waveform
from ._data import NRSurData, default_data

__all__ = ["Grids", "grids", "evaluate", "on_grid", "td_polarizations", "td_modes", "batch_td_polarizations"]

#: Largest batch evaluated at once by :func:`batch_td_polarizations`.
BATCH_CHUNK = 1024
#: Largest number of samples (binaries times samples) evaluated at once.
CHUNK_SAMPLES = 1 << 23


class Grids(NamedTuple):
    """The output grids of a batch (numpy arrays, one entry per binary; see
    :class:`.._driver.Setup`): ``n`` samples at the dimensionless times
    ``t0 + dt k`` from ``epoch`` (s), the effective ``f_ref`` (``f_min``
    for ``f_ref = 0``) and the total mass ``m_tot_sec``."""

    f_ref: np.ndarray
    t0: np.ndarray
    dt: np.ndarray
    n: np.ndarray
    epoch: np.ndarray
    m_tot_sec: np.ndarray


@jax.jit
def _omegas(data, q, chi_a, chi_b):
    return jax.vmap(reference_omegas, in_axes=(None, 0, 0, 0))(data, q, chi_a, chi_b)


def _reference_times(t_ds, omegas, omega_ref):
    """``PrecessingNRSur_get_t_ref`` over a batch (as
    :func:`.._driver._reference_time`): the reference times and whether
    LAL fails."""
    zero = np.abs(omega_ref) < OMEGA_REF_ZERO
    above = omegas[:, 1:] > omega_ref[:, None]
    found = np.any(above, axis=1)
    fail = ~zero & ((omega_ref > OMEGA_REF_MAX) | (omega_ref < omegas[:, 0]) | ~found)
    imax = 1 + np.argmax(above, axis=1)
    rows = np.arange(len(omegas))
    omega_max = omegas[rows, imax]
    omega_min = omegas[rows, imax - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (t_ds[imax - 1] * (omega_max - omega_ref) + t_ds[imax] * (omega_ref - omega_min)) / (omega_max - omega_min)
    return np.where(zero, START_TIME, t), fail


def grids(table, delta_t: float, f_min: float, options) -> Tuple[Grids, Dict[int, Exception]]:
    """The grids of the binaries of ``table`` (``(n, 12)``, see
    :func:`waveformx.parameter_table`) and ``{index: error}`` for those
    LAL refuses (their entries in the grids are meaningless): as
    :func:`.._driver.setup` for every row, in numpy over the batch."""
    _jax_driver.check_mode_array(options.mode_array)
    if not (np.isfinite(delta_t) and delta_t > 0):
        raise InvalidParameterError(f"invalid sampling interval {delta_t}")
    data = jax_default_data()
    table = np.asarray(table, dtype=np.float64)
    # PrecessingNRSur_switch_labels_if_needed (_driver._host_labels)
    m1 = table[:, 0] * MSUN_SI
    m2 = table[:, 1] * MSUN_SI
    switched = m1 < m2
    m1, m2 = np.where(switched, m2, m1), np.where(switched, m1, m2)
    flip = np.array([-1.0, -1.0, 1.0])
    s1, s2 = table[:, 2:5], table[:, 5:8]
    chi_a = np.where(switched[:, None], flip * s2, s1)
    chi_b = np.where(switched[:, None], flip * s1, s2)
    mass_a = m1 / MSUN_SI
    mass_b = m2 / MSUN_SI
    q = mass_a / mass_b
    m_tot_sec = (mass_a + mass_b) * MTSUN_SI
    # get_dimless_omega
    f_ref = table[:, 11]
    omega_min = (m_tot_sec * f_min) * np.pi
    if options.reference_mf is None:
        omega_ref = np.where(f_ref == 0, omega_min, (m_tot_sec * f_ref) * np.pi)
        fail = omega_ref + 1e-13 < omega_min
    else:
        omega_ref = np.full_like(m_tot_sec, np.pi * options.reference_mf)
        fail = f_ref != 0
    # PrecessingNRSur_IntegrateDynamics
    fail |= q > _jax_driver.Q_MAX
    for chi in (chi_a, chi_b):
        fail |= np.sqrt(np.sum(chi * chi, axis=1)) > _jax_driver.CHI_MAX
    t_ds = np.asarray(data.t_ds)
    omegas = np.asarray(_omegas(data, q, chi_a, chi_b))
    _, fail_ref = _reference_times(t_ds, omegas, omega_ref)
    fail |= fail_ref
    # the start of the output (PrecessingNRSur_StartFrequency with the
    # masses in kg)
    q_start = m1 / m2
    omega_start = omegas[:, 0].copy()
    other = np.flatnonzero(q_start != q)
    if other.size:
        omega_start[other] = np.asarray(_omegas(data, q_start[other], chi_a[other], chi_b[other]))[:, 0]
    start_freq = omega_start / (np.pi * (((m1 + m2) / MSUN_SI) * MTSUN_SI))
    from_f_min = f_min >= start_freq
    t_min, fail_min = _reference_times(t_ds, omegas, omega_min)
    fail |= np.where(from_f_min, fail_min, f_min > 0)
    t_coorb = np.asarray(data.t_coorb)
    t0 = np.where(from_f_min, t_min, t_coorb[0])
    dt = delta_t / m_tot_sec
    with np.errstate(invalid="ignore"):
        n = np.maximum(np.ceil((t_coorb[-1] - t0) / dt), 0)
    n = np.where(fail, 0, n).astype(np.int64)
    epoch = np.array([_jax_driver.gps_epoch(m * t) if not bad else np.nan
                      for m, t, bad in zip(m_tot_sec, t0, fail)])
    out = Grids(np.where((f_ref != 0) | (options.reference_mf is not None), f_ref, f_min), t0, dt, n, epoch, m_tot_sec)
    errors = {}
    for index in np.flatnonzero(fail):
        # LAL's error, from the checks in its order
        try:
            grid = _jax_driver.setup(row(table, index), delta_t, f_min, options, data)
        except WaveformError as error:
            errors[int(index)] = error
        else:
            for name, value in zip(Grids._fields, (grid.binary.f_ref, grid.t0, grid.dt, grid.n, grid.epoch,
                                                   grid.m_tot_sec)):
                getattr(out, name)[index] = value
    return out, errors


@functools.lru_cache(maxsize=None)
def _compiled(data: NRSurData, kind: str, mode_array, scale: float, reference_mf=None):
    """The compiled evaluation of :func:`evaluate`, cached per data,
    output and options (and by MLX per shape)."""

    def function(binary, k, k0, dt, c):
        # t0 + dt k as (k - k0) dt + (t0 + k0 dt), with k0 the sample
        # nearest the peak: rounded relative to tau rather than to t0
        tau = (k[None] - k0[:, None]) * dt[:, None] + c[:, None]
        if kind == "modes":
            return tuple(_waveform.modes_at_dimensionless(data, binary, tau, mode_array, scale, reference_mf).values())
        return _waveform.polarizations_at_dimensionless(data, binary, tau, mode_array, scale, reference_mf)

    return mx.compile(function)


def _float32(values) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.float32))


def evaluate(data: NRSurData, binaries: Binary, t0, dt, n: int, options, kind: str = "polarizations"):
    """The polarizations (``kind="polarizations"``, ``(h_+, h_x)``, each
    ``(B, n)``) or the modes (``kind="modes"``, ``{(l, m): (B, n)}``) of
    the batched ``binaries`` (numpy float64 fields with a leading axis, the
    effective ``f_ref``) at the dimensionless times ``t0 + dt k``, ``k = 0
    .. n - 1`` (``t0`` and ``dt`` numpy arrays ``(B,)``)."""
    t0 = np.asarray(t0, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    k0 = np.round(-t0 / dt)
    c = t0 + k0 * dt
    binary = Binary(*(_float32(value) for value in binaries))
    function = _compiled(data, kind, options.mode_array, strain_scale(options), options.reference_mf)
    out = function(binary, mx.arange(n, dtype=mx.float32), _float32(k0), _float32(dt), _float32(c))
    if kind == "modes":
        ells = output_ells(options.mode_array)
        return dict(zip([mode for mode in MODES if mode[0] in ells], out))
    return out


def on_grid(data: NRSurData, grid, options, kind: str = "polarizations"):
    """:func:`evaluate` for one binary on its grid (a
    :class:`.._driver.Setup`), trimmed to its ``n`` samples: ``(h_+, h_x)``
    or ``{(l, m): h_lm}``, each ``(n,)``."""
    binary = Binary(*(np.asarray(value, dtype=np.float64)[None] for value in grid.binary))
    out = evaluate(data, binary, [grid.t0], [grid.dt], _jax_driver.bucket(grid.n), options, kind)
    if kind == "modes":
        return {mode: h[0, : grid.n] for mode, h in out.items()}
    return tuple(h[0, : grid.n] for h in out)


def td_polarizations(binary, delta_t: float, f_min: float, options, data: Optional[NRSurData] = None):
    """``SimInspiralChooseTDWaveform``: a
    :class:`~waveformx.TimeSeriesPolarizations` of float32 MLX arrays with
    LAL's length and epoch."""
    data = default_data() if data is None else data
    grid = _jax_driver.setup(binary, float(delta_t), float(f_min), options)
    hp, hc = on_grid(data, grid, options)
    return TimeSeriesPolarizations(hp, hc, grid.epoch, float(delta_t))


def td_modes(binary, delta_t: float, f_min: float, options, data: Optional[NRSurData] = None):
    """``SimInspiralChooseTDModes`` (``lmax = 4``): a
    :class:`~waveformx.TimeSeriesModes` of complex64 MLX arrays."""
    data = default_data() if data is None else data
    grid = _jax_driver.setup(binary, float(delta_t), float(f_min), options)
    return TimeSeriesModes(on_grid(data, grid, options, kind="modes"), grid.epoch, float(delta_t))


def _chunks(lengths: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """The indices of ``lengths`` in chunks, longest first, of at most
    ``chunk_size`` binaries and :data:`CHUNK_SAMPLES` samples."""
    order = np.argsort(-lengths, kind="stable")
    out, start = [], 0
    while start < len(order):
        n = _jax_driver.bucket(lengths[order[start]])
        size = min(chunk_size, max(1, CHUNK_SAMPLES // n))
        out.append(order[start : start + size])
        start += size
    return out


def batch_td_polarizations(
    parameters,
    delta_t: float,
    f_min: float,
    options,
    *,
    strict: bool = False,
    chunk_size: Optional[int] = None,
    data: Optional[NRSurData] = None,
) -> BatchTimeSeries:
    """The polarizations of many binaries (one series each, of LAL's length
    and epoch), as :func:`waveformx.batch_td_polarizations`:
    ``parameters`` is a mapping of arrays, a structured array or an ``(n,
    12)`` table (see :func:`waveformx.parameter_table`); binaries that
    fail LAL's checks, or whose waveform is not finite, are reported in
    ``ok`` and ``errors`` (``None`` in ``waveforms``), unless ``strict``
    (then the first error is raised). The binaries are evaluated on the GPU
    at most ``chunk_size`` (default :data:`BATCH_CHUNK`) at a time."""
    data = default_data() if data is None else data
    delta_t, f_min = float(delta_t), float(f_min)
    if not np.isfinite(f_min) or f_min < 0:
        raise InvalidParameterError("f_min must be non-negative")
    table = parameter_table(parameters)
    errors: Dict[int, Exception] = {}
    for index in range(len(table)):
        try:
            check_binary(row(table, index))
        except WaveformError as error:
            errors[index] = error
    checked = np.array([i for i in range(len(table)) if i not in errors], dtype=np.int64)
    grid, failed = grids(table[checked], delta_t, f_min, options) if checked.size else (None, {})
    errors.update({int(checked[i]): error for i, error in failed.items()})
    if strict and errors:
        raise errors[min(errors)]
    waveforms: List[Optional[TimeSeriesPolarizations]] = [None] * len(table)
    good = np.array([i for i in range(len(checked)) if i not in failed], dtype=np.int64)
    chunk = BATCH_CHUNK if chunk_size is None else max(1, int(chunk_size))
    for members in _chunks(grid.n[good], chunk) if good.size else []:
        members = good[members]
        # pad to a power of two, so that chunks share compilations
        padded = np.concatenate([members, np.full((1 << (len(members) - 1).bit_length()) - len(members), members[-1])])
        effective = table[checked[padded]].copy()
        effective[:, 11] = grid.f_ref[padded]
        binaries = Binary(effective[:, 0], effective[:, 1], effective[:, 2:5], effective[:, 5:8],
                          effective[:, 8], effective[:, 9], effective[:, 10], effective[:, 11])
        n = _jax_driver.bucket(int(grid.n[members].max()))
        hp, hc = evaluate(data, binaries, grid.t0[padded], grid.dt[padded], n, options)
        finite = mx.all(mx.isfinite(hp) & mx.isfinite(hc), axis=1)
        mx.eval(hp, hc, finite)
        finite = np.array(finite)
        for k, index in enumerate(members):
            original = int(checked[index])
            if not finite[k]:
                errors[original] = ModelError(f"NRSur7dq4 could not be evaluated for binary {original}")
                continue
            length = int(grid.n[index])
            waveforms[original] = TimeSeriesPolarizations(hp[k, :length], hc[k, :length], float(grid.epoch[index]),
                                                          delta_t)
    if strict and errors:
        raise errors[min(errors)]
    ok = np.array([w is not None for w in waveforms], dtype=bool)
    return BatchTimeSeries(waveforms, ok, errors)
