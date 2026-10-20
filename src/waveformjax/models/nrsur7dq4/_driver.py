# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``SimInspiralChooseTDWaveform`` and ``SimInspiralChooseTDModes`` for
NRSur7dq4: LAL's checks, the start and length of the output, and the
evaluation on LAL's time grid.

The length of the output depends on the binary (through the time at which
the orbital frequency reaches ``pi f_min``), so it is computed on the host:
a compiled pass evaluates the orbital-frequency fits at the dynamics nodes,
and LAL's search for the start time, its checks, the number of samples and
the epoch follow in numpy, operation by operation. The waveform is then
evaluated by the compiled pure function on a grid of fixed (bucketed)
length, so that waveforms of similar lengths share one compilation, and
trimmed to the binary's own length.
"""

from __future__ import annotations

import functools
import math
from typing import List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..._binary import Binary, binaries_from_table, row
from ..._results import TimeSeriesModes, TimeSeriesPolarizations
from ...exceptions import InvalidParameterError, ModelError
from . import _waveform
from ._data import LMAX, NRSurData, default_data
from ._dynamics import OMEGA_REF_MAX, OMEGA_REF_ZERO, START_TIME, reference_omegas

__all__ = ["Setup", "setup", "check_mode_array", "td_polarizations", "td_modes", "batch_td_polarizations", "gps_epoch"]

#: Largest mass ratio LAL allows without ``unlimited_extrapolation``.
Q_MAX = 6.01
#: Largest spin magnitude LAL allows.
CHI_MAX = 1.0
#: Binaries evaluated at once by :func:`batch_td_polarizations`.
BATCH_CHUNK = 64
#: Smallest length of the evaluation grids (see :func:`bucket`).
MIN_BUCKET = 256


class Setup(NamedTuple):
    """The output grid of a binary: ``n`` samples at the dimensionless times
    ``t0 + dt k`` (units of the total mass ``m_tot_sec``), starting at
    ``epoch`` (s, LAL's GPS time rounded to nanoseconds), and the binary with
    ``f_ref = 0`` replaced by ``f_min`` (``get_dimless_omega``)."""

    binary: Binary
    t0: float
    dt: float
    n: int
    epoch: float
    m_tot_sec: float


def check_mode_array(mode_array) -> None:
    """LAL's check of the ``ModeArray`` (``ell <= 4``), and ``|m| <= ell``."""
    if mode_array is None:
        return
    for ell, m in mode_array:
        if not 2 <= ell <= LMAX or abs(m) > ell:
            raise InvalidParameterError(
                f"NRSur7dq4 has no mode ({ell}, {m}): a requested mode is not available (ell must be 2..={LMAX})"
            )


def gps_epoch(t: float) -> float:
    """``XLALGPSAdd(&zero, t)`` then ``XLALGPSGetREAL8``: ``t`` rounded to
    nanoseconds as LAL's ``LIGOTimeGPS``."""
    seconds = math.floor(t)
    nanoseconds = round((t - seconds) * 1e9)
    total = 1_000_000_000 * seconds + nanoseconds
    whole = abs(total) // 1_000_000_000 * (1 if total >= 0 else -1)
    return float(whole) + float(total - 1_000_000_000 * whole) / 1e9


def _host_labels(binary):
    """``PrecessingNRSur_switch_labels_if_needed`` and the derived masses, in
    numpy: ``(switched, m1_si, m2_si, chi_a, chi_b, q, m_tot_sec)``."""
    m1 = float(binary.mass_1) * _waveform.MSUN_SI
    m2 = float(binary.mass_2) * _waveform.MSUN_SI
    s1 = np.asarray(binary.spin_1, dtype=float)
    s2 = np.asarray(binary.spin_2, dtype=float)
    switched = m1 < m2
    if switched:
        m1, m2 = m2, m1
        s1, s2 = np.array([-s2[0], -s2[1], s2[2]]), np.array([-s1[0], -s1[1], s1[2]])
    mass_a = m1 / _waveform.MSUN_SI
    mass_b = m2 / _waveform.MSUN_SI
    return switched, m1, m2, s1, s2, mass_a / mass_b, (mass_a + mass_b) * _waveform.MTSUN_SI


@jax.jit
def _omegas(data, q, chi_a, chi_b):
    return reference_omegas(data, q, chi_a, chi_b)


def _reference_time(t_ds: np.ndarray, omegas: np.ndarray, omega_ref: float) -> float:
    """``PrecessingNRSur_get_t_ref`` in numpy, with its errors."""
    if abs(omega_ref) < OMEGA_REF_ZERO:
        return START_TIME
    if omega_ref > OMEGA_REF_MAX:
        raise InvalidParameterError(f"reference frequency omega_ref={omega_ref:.4f} > 0.2, too large")
    omega_min = omegas[0]
    if omega_ref < omega_min:
        raise InvalidParameterError(
            f"got omega_ref or omega_low={omega_ref:.4f} smaller than the minimum omega={omega_min:.4f} "
            "for this configuration"
        )
    n = len(t_ds)
    imax = 1
    omega_max = omegas[imax]
    while omega_max <= omega_ref and imax < n:
        imax += 1
        omega_min = omega_max
        if imax == n:
            raise ModelError(
                f"NRSur7dq4 failed: tried all nodes and still have omega={omega_min:.4f} <= omega_ref={omega_ref:.4f}"
            )
        omega_max = omegas[imax]
    t_min = t_ds[imax - 1]
    t_max = t_ds[imax]
    return (t_min * (omega_max - omega_ref) + t_max * (omega_ref - omega_min)) / (omega_max - omega_min)


def setup(binary, delta_t: float, f_min: float, options, data: Optional[NRSurData] = None) -> Setup:
    """LAL's checks for concrete values, in LAL's order, and the output grid
    (``XLALSimInspiralPrecessingNRSurPolarizations`` up to the
    interpolation). Raises :class:`~waveformjax.InvalidParameterError` where
    LAL fails with ``XLAL_EINVAL`` or ``XLAL_EDOM`` and
    :class:`~waveformjax.ModelError` where it fails otherwise."""
    check_mode_array(options.mode_array)
    if not (np.isfinite(delta_t) and delta_t > 0):
        raise InvalidParameterError(f"invalid sampling interval {delta_t}")
    switched, m1_si, m2_si, chi_a, chi_b, q, m_tot_sec = _host_labels(binary)
    f_ref = float(binary.f_ref)
    # get_dimless_omega
    omega_min = (m_tot_sec * f_min) * np.pi
    omega_ref = omega_min if f_ref == 0 else (m_tot_sec * f_ref) * np.pi
    if omega_ref + 1e-13 < omega_min:
        raise InvalidParameterError(f"fRef ({f_ref}) cannot be lesser than fMin ({f_min})")
    # PrecessingNRSur_IntegrateDynamics
    if q > Q_MAX:
        raise InvalidParameterError(
            f"too much extrapolation: mass ratio q = {q:.4f} > {Q_MAX:.4f}, the maximum allowed value"
        )
    for label, chi in (("A", chi_a), ("B", chi_b)):
        magnitude = float(np.sqrt(np.sum(chi * chi)))
        if magnitude > CHI_MAX:
            raise InvalidParameterError(
                f"too much extrapolation: spin magnitude |chi{label}| = {magnitude:.4f} > {CHI_MAX:.4f}, "
                "the maximum allowed value"
            )
    if data is None:
        data = default_data()
    t_ds = np.asarray(data.t_ds)
    omegas = np.asarray(_omegas(data, q, jnp.asarray(chi_a), jnp.asarray(chi_b)))
    _reference_time(t_ds, omegas, omega_ref)
    # the start of the output (PrecessingNRSur_StartFrequency with the
    # masses in kg)
    q_start = m1_si / m2_si
    if q_start == q:
        omega_start = omegas[0]
    else:
        omega_start = float(_omegas(data, q_start, jnp.asarray(chi_a), jnp.asarray(chi_b))[0])
    start_freq = omega_start / (np.pi * (((m1_si + m2_si) / _waveform.MSUN_SI) * _waveform.MTSUN_SI))
    if f_min >= start_freq:
        t0 = _reference_time(t_ds, omegas, omega_min)
    elif f_min > 0:
        raise InvalidParameterError(
            f"fMin should be 0 or >= {start_freq:.8f} for this configuration, got {f_min:.8f}"
        )
    else:
        t0 = float(np.asarray(data.t_coorb)[0])
    dt = delta_t / m_tot_sec
    tf = float(np.asarray(data.t_coorb)[-1])
    n = math.ceil((tf - t0) / dt)
    effective = binary._replace(f_ref=f_ref if f_ref != 0 else float(f_min))
    return Setup(effective, float(t0), float(dt), max(int(n), 0), gps_epoch(m_tot_sec * t0), float(m_tot_sec))


def bucket(n: int) -> int:
    """The length of the grid on which ``n`` samples are evaluated: the next
    power of two (at least :data:`MIN_BUCKET`), so that compilations are
    shared."""
    return max(MIN_BUCKET, 1 << max(0, int(n) - 1).bit_length())


@functools.partial(jax.jit, static_argnames=("n", "mode_array"))
def _polarizations_grid(data, binary, t0, dt, n, mode_array):
    tau = t0 + dt * jnp.arange(n)
    return _waveform.polarizations_at_dimensionless(data, binary, tau, mode_array)


@functools.partial(jax.jit, static_argnames=("n", "mode_array"))
def _modes_grid(data, binary, t0, dt, n, mode_array):
    tau = t0 + dt * jnp.arange(n)
    return _waveform.modes_at_dimensionless(data, binary, tau, mode_array)


@functools.partial(jax.jit, static_argnames=("n", "mode_array"))
def _batch_polarizations_grid(data, binaries, t0, dt, n, mode_array):
    def one(binary, t0, dt):
        tau = t0 + dt * jnp.arange(n)
        return _waveform.polarizations_at_dimensionless(data, binary, tau, mode_array)

    return jax.vmap(one)(binaries, t0, dt)


def _as_jax(binary: Binary) -> Binary:
    return Binary(*(jnp.asarray(value, dtype=jnp.float64) for value in binary))


def td_polarizations(binary, delta_t: float, f_min: float, options) -> TimeSeriesPolarizations:
    """``SimInspiralChooseTDWaveform`` for NRSur7dq4."""
    data = default_data()
    grid = setup(binary, delta_t, f_min, options, data)
    hp, hc = _polarizations_grid(data, _as_jax(grid.binary), grid.t0, grid.dt, bucket(grid.n), options.mode_array)
    return TimeSeriesPolarizations(hp[: grid.n], hc[: grid.n], grid.epoch, float(delta_t))


def td_modes(binary, delta_t: float, f_min: float, options) -> TimeSeriesModes:
    """``SimInspiralChooseTDModes`` (``lmax = 4``) for NRSur7dq4: the inertial
    modes of every ``ell`` that has a coorbital mode in the mode array."""
    data = default_data()
    grid = setup(binary, delta_t, f_min, options, data)
    modes = _modes_grid(data, _as_jax(grid.binary), grid.t0, grid.dt, bucket(grid.n), options.mode_array)
    return TimeSeriesModes({mode: h[: grid.n] for mode, h in modes.items()}, grid.epoch, float(delta_t))


def batch_td_polarizations(
    table, ok, delta_t: float, f_min: float, options, chunk_size: Optional[int] = None
) -> List[Optional[TimeSeriesPolarizations]]:
    """The polarizations of the binaries of ``table`` (``(n, 12)``, see
    :func:`waveformjax.parameter_table`) whose ``ok`` is true, ``None``
    for the others. The binaries are evaluated ``chunk_size`` at a time with
    ``jax.vmap`` on a common grid of samples (each from its own start) and
    trimmed to their own lengths."""
    data = default_data()
    chunk = BATCH_CHUNK if chunk_size is None else max(1, int(chunk_size))
    out: List[Optional[TimeSeriesPolarizations]] = [None] * len(table)
    indices = [int(i) for i in np.flatnonzero(ok)]
    grids = {}
    for index in indices:
        grids[index] = setup(row(table, index), delta_t, f_min, options, data)
    effective = np.array(table, dtype=float, copy=True)
    for index, grid in grids.items():
        effective[index, 11] = float(grid.binary.f_ref)
    for start in range(0, len(indices), chunk):
        members = indices[start : start + chunk]
        # pad to a full chunk, so that every chunk shares one compilation
        padded = members + [members[-1]] * (min(chunk, len(indices)) - len(members))
        n = bucket(max(grids[i].n for i in members))
        binaries = binaries_from_table(effective[padded])
        t0 = jnp.asarray([grids[i].t0 for i in padded])
        dt = jnp.asarray([grids[i].dt for i in padded])
        hp, hc = _batch_polarizations_grid(data, binaries, t0, dt, n, options.mode_array)
        hp, hc = np.asarray(hp), np.asarray(hc)
        for k, index in enumerate(members):
            grid = grids[index]
            out[index] = TimeSeriesPolarizations(
                jnp.asarray(hp[k, : grid.n]), jnp.asarray(hc[k, : grid.n]), grid.epoch, float(delta_t)
            )
    return out
