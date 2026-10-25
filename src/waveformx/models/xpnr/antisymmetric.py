# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The antisymmetric part of the co-precessing (2,|2|) modes of
IMRPhenomXPNR (Ghosh et al. 2024, arXiv:2310.16980;
``LALSimIMRPhenomX_AntisymmetricWaveform.c``), after waveformrs's
``imrphenomx/pnr.rs``, with the fix of lalsuite !2593.

The amplitude is the symmetric (2,2) amplitude times a ratio ``kappa(f)``
smoothed by LAL's moving average over 81 frequencies. LAL averages in
place, so each window reads the averages already written for the 40
frequencies before its centre: an order-40 linear recurrence, evaluated here
in blocks of 40 values (a ``lax.scan`` over the blocks with a triangular
solve each), which gives LAL's values to rounding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.linalg  # noqa: F401

from . import pnr
from ._generated import inspiral as _inspiral
from ._generated import intermediate as _intermediate
from ._generated import ringdown as _ringdown
from ._math import FRAC_PI_2, FRAC_PI_4, FencedModule, PI, Struct, barrier, dbl_epsilon, fenced, fma, fnms, hz_to_mf, powers, real
from ._transforms import cond
from .amp22 import amp_ansatz, amp_coefficients
from .phase22 import phase_22, phase_coefficients, phase_xas, time_shift_22

inspiral = FencedModule(_inspiral)
intermediate = FencedModule(_intermediate)
ringdown = FencedModule(_ringdown)

__all__ = ["antisymmetric_kappa", "antisymmetric_waveform", "smooth_amp_ratio"]

WIDTH = 80


def _kappa_function(prec, wf):
    """The amplitude ratio ``kappa(v)`` below the ringdown frequency and its
    value at the ringdown frequency."""
    m2 = wf.m2
    delta = fnms(2.0, m2, 1.0)
    eta = wf.eta
    theta = prec.theta_antisymmetric
    chi = prec.chi_single_spin_antisymmetric
    b0, b1, b2, b3 = 18.0387, 15.4509, 55.1140, -203.6290
    b = fma(b3 * eta, theta, fma(b2, theta, fma(b1, eta, b0)))
    sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)

    def kappa_at(v):
        v2 = v * v
        v3 = v2 * v
        v5 = v3 * v2
        num = 21.0 * v2 * (1.0 + delta) * chi * sin_theta
        den = 2.0 * fma(-(28.0 * v3 * (1.0 + delta - eta) * chi), cos_theta,
                        fma(v2, fma(55.0, eta, -107.0), fma(84.0 * PI, v3, 42.0)))
        return num / den * fma(b, v5, 1.0)

    return kappa_at, kappa_at(jnp.cbrt(PI * wf.f_ring))


def smooth_amp_ratio(kappa, frequencies):
    """The moving average of ``IMRPhenomX_PNR_GenerateAntisymmetricAmpRatio``,
    in place as LAL does it: for ``id = 0, 1, ...``, ``kappa[id + 40] =
    sum_j kappa[id + j] df_j / sum_j df_j`` over ``j = 0 .. 80`` with ``df_j
    = f[id + j + 1] - f[id + j]`` (fewer frequencies on short grids), each
    window reading the values the earlier windows wrote."""
    length = kappa.shape[-1]
    width = WIDTH
    if width > length - 1:
        width = length // 2
    half = width // 2
    end = length - width - 1
    if end <= 0:
        return kappa
    df = jnp.diff(frequencies)
    # window id covers kappa[id .. id + width] with weights df[id .. id + width]:
    # its sums from running sums (the same to rounding)
    zero = jnp.zeros(1, real())
    running_df = jnp.concatenate([zero, jnp.cumsum(df)])
    running_kdf = jnp.concatenate([zero, jnp.cumsum(kappa[:-1] * df)])
    total_df = running_df[width + 1:width + 1 + end] - running_df[:end]
    # the part of each window that reads values not yet overwritten: j >= half
    fixed = running_kdf[width + 1:width + 1 + end] - running_kdf[half:half + end]
    # u_{id + half} = (sum_{j < half} u_{id + j} w_j + fixed_id) / W_id with
    # u the averaged values (the original kappa below index half), solved
    # in blocks of `half` outputs
    n_blocks = -(-end // half)
    pad = n_blocks * half - end
    rhs = jnp.pad(fixed / total_df, (0, pad)).reshape(n_blocks, half)

    # within a block, output i reads u_{block * half + i + j}: the previous
    # block's outputs for i + j < half, this block's earlier outputs
    # otherwise; a unit lower-triangular system per block
    # (numpy constants: the branches below must not close over traced values)
    i_index = np.arange(half)[:, None]
    k_index = np.arange(half)[None, :]
    j_previous = k_index - i_index
    j_current = k_index + half - i_index

    def systems(w):
        # the system of a block (w[j, i]: weight of u_{id + j} for output
        # id = block * half + i): u = from_previous @ previous + lower @ u + r
        from_previous = jnp.where(j_previous >= 0, w[jnp.clip(j_previous, 0, half - 1), i_index], 0.0)
        lower = jnp.where(k_index < i_index, w[jnp.clip(j_current, 0, half - 1), i_index], 0.0)
        return from_previous, jnp.eye(half, dtype=real()) - lower

    def general(operands):
        df, total_df, rhs, initial = operands
        weights = jnp.stack([df[j:j + end] for j in range(half)]) / total_df  # (half, end)
        weights = jnp.pad(weights, ((0, 0), (0, pad))).reshape(half, n_blocks, half)  # [j, block, i]

        def block(previous, inputs):
            w, r = inputs
            from_previous, matrix = systems(w)
            out = jax.scipy.linalg.solve_triangular(matrix, r + from_previous @ previous, lower=True,
                                                    unit_diagonal=True)
            return out, out

        return jax.lax.scan(block, initial, (jnp.moveaxis(weights, 1, 0), rhs))[1]

    def uniform(operands):
        # equal frequency steps: every block has the same system, solved once
        # for the previous block's outputs and for all the right-hand sides
        df, total_df, rhs, initial = operands
        # (the first block's weights, zero beyond the outputs on short grids)
        index = np.arange(half)[:, None] + np.arange(half)[None, :]  # [j, i]
        valid = np.arange(half) < end
        total = jnp.where(valid, jnp.pad(total_df, (0, pad))[:half], 1.0)
        w = jnp.where(valid, df[jnp.clip(index, 0, df.shape[0] - 1)] / total, 0.0)
        from_previous, matrix = systems(w)
        solved = jax.scipy.linalg.solve_triangular(matrix, jnp.concatenate([from_previous, rhs.T], axis=1),
                                                   lower=True, unit_diagonal=True)
        a, c = solved[:, :half], solved[:, half:].T

        def block(previous, c_b):
            out = jnp.sum(a * previous, axis=1) + c_b
            return out, out

        return jax.lax.scan(block, initial, c)[1]

    # (steps equal to 1e-12, or to a few roundings in float32: the weights
    # of every window differ by at most as much from the first window's)
    equal_steps = jnp.all(jnp.abs(df - df[0]) <= max(1e-12, 8.0 * dbl_epsilon()) * jnp.abs(df[0]))
    outputs = cond(equal_steps, uniform, general, (df, total_df, rhs, kappa[:half]))
    smoothed = outputs.reshape(-1)[:end]
    return kappa.at[half:half + end].set(smoothed)


@fenced
def antisymmetric_coefficients(prec, wf, version):
    """``IMRPhenomX_PNR_GenerateAntisymmetricPhaseCoefficients`` and the
    frequency-independent quantities of
    ``IMRPhenomX_PNR_GenerateAntisymmetricWaveform``."""
    phase22 = phase_coefficients(wf)
    amp22 = amp_coefficients(wf)
    lina = 0.0
    linb = time_shift_22(phase22, wf)
    inveta = 1.0 / wf.eta
    phiref22 = fma(2.0, wf.phi0, fma(-inveta, phase_22(wf.mf_ref, powers(wf.mf_ref), phase22),
                                     -(linb * wf.mf_ref)) - lina) + FRAC_PI_4
    # the alpha parameters again, with the offsets of alpha after the PNR
    # angle interpolants (IMRPhenomX_PNR_GenerateAntisymmetricPhaseCoefficients)
    banked = version == 330 and "st_bank" in prec
    ap = prec.st_bank_params.alpha if banked else pnr.precompute_alpha_coefficients(prec, wf, version)
    ap, error_ap = pnr.alpha_connection(prec, wf, ap, version)
    delta_mf = pnr.ANTISYMMETRIC_DMF
    mf_t, pts = pnr.antisymmetric_points(wf)
    if banked:
        # (evaluated with the set-up, pnr.st_bank)
        pts = prec.st_bank.antisymmetric.mf
        mf_t = pts[2]
    alphas, errors = pnr.region_alpha_at(prec, wf, pts, ap, version)
    # the alpha of the interpolants differs from this one by a constant
    # (the offset of alpha, changed by the interpolants): the same
    # alpha with the parameters and the offset of the grid
    grid_prec = pnr._copy(prec)
    grid_prec.alpha_offset = prec.alpha_offset_grid
    alphas_grid, _ = pnr.region_alpha_at(grid_prec, wf, pts, prec.alpha_params, version)
    alpha_der_mf_t = (alphas[0] - alphas[1]) / 2.0 / delta_mf
    alpha_mf_t = alphas[2]
    p = powers(mf_t)
    dphi_ins = inspiral.phase_22_ansatz_exact(mf_t, p, phase22)
    phi_ins = inspiral.phase_22_ansatz_int_exact(mf_t, p, phase22)
    dphi_rd = ringdown.phase_22_ansatz_exact(mf_t, p, phase22) + phase22.c2_mrd
    phi_rd = fma(phase22.c2_mrd, mf_t, ringdown.phase_22_ansatz_int_exact(mf_t, p, phase22) + phase22.c1_mrd)
    dphi_int = intermediate.phase_22_ansatz_exact(mf_t, p, phase22) + phase22.c2_int
    phi_int = fma(phase22.c2_int, mf_t, intermediate.phase_22_ansatz_int_exact(mf_t, p, phase22) + phase22.c1_int)
    ins = mf_t < phase22.f_phase_match_in
    rd = mf_t > phase22.f_phase_match_im
    dphi = jnp.where(ins, dphi_ins, jnp.where(rd, dphi_rd, dphi_int))
    phi = jnp.where(ins, phi_ins, jnp.where(rd, phi_rd, phi_int))
    dphi = dphi * inveta + linb
    phi = phi * inveta + (fma(linb, mf_t, lina) + phiref22)
    a0 = dphi / 2.0 - alpha_der_mf_t
    phi_a0 = -alphas[3] - FRAC_PI_2
    error = jnp.any(errors) | error_ap
    phi_b0 = fma(a0, mf_t, alpha_mf_t - phi / 2.0) + phi_a0
    phase_offset = prec.phi_j_sf
    return Struct(alpha_params=ap, phase22=phase22, amp22=amp22, mf_t=mf_t, alpha_shift=alphas[2] - alphas_grid[2],
                  lina=lina, linb=linb, inveta=inveta, phiref22=phiref22,
                  a0=a0, phi_a0=phi_a0, phi_b0=phi_b0, phase_offset=phase_offset,
                  amp_prefactor=wf.amp0 * wf.amp_norm, error=error)


def antisymmetric_kappa(prec, wf, frequencies):
    """The amplitude ratio of the antisymmetric mode at ``frequencies``
    (Hz) after LAL's moving average (:func:`smooth_amp_ratio`), which reads
    all the frequencies."""
    prec, wf = barrier((prec, wf))
    kappa_at, kappa_rd = _kappa_function(prec, wf)
    mf = hz_to_mf(frequencies, wf.m_tot)
    below = mf < wf.f_ring
    kappa = jnp.where(below, kappa_at(jnp.cbrt(PI * jnp.where(below, mf, wf.f_ring))), kappa_rd)
    return smooth_amp_ratio(kappa, frequencies)


def antisymmetric_waveform(prec, wf, frequencies, version, alpha=None, kappa=None, co=None):
    """``IMRPhenomX_PNR_GenerateAntisymmetricWaveform``: amplitude and phase
    of the antisymmetric (2,2) co-precessing mode at ``frequencies`` (Hz),
    ``h = amp exp(i phase)``, the error flag and the coefficients of the
    symmetric (2,2) mode it is built on (``amp22``, ``phase22``, ``linb``).
    ``alpha`` is the PNR ``alpha`` of the interpolants at the frequencies,
    without the offset (:func:`.pnr.region_alpha_many` if ``None``),
    ``co`` its coefficients (:func:`antisymmetric_coefficients`) if computed
    already, ``kappa`` the amplitude ratio
    (:func:`antisymmetric_kappa`, of all the frequencies when these are a
    part of them)."""
    prec, wf = barrier((prec, wf))
    if kappa is None:
        kappa = antisymmetric_kappa(prec, wf, frequencies)
    mf = hz_to_mf(frequencies, wf.m_tot)
    if co is None:
        co = antisymmetric_coefficients(prec, wf, version)
    # the PNR alpha enters below mf_t only: computed up to the last
    # frequency there
    error = co.error
    if alpha is None:
        used = mf < co.mf_t
        n_used = jnp.max(jnp.where(used, jnp.arange(mf.shape[0]), -1)) + 1
        alpha, errors = pnr.region_alpha_many(prec, wf, mf, co.alpha_params, version, n_used)
        mf_cut = wf.f_max_prime * wf.m_sec
        error = error | jnp.any(errors & used & (mf <= mf_cut))
        alpha_pnr = alpha
    else:
        # (LAL evaluates alpha again, with its parameters after the offsets
        # of the interpolants: theirs up to a constant)
        alpha_pnr = alpha + co.alpha_shift
    mf = wf.m_sec * frequencies
    p = powers(mf)
    phi = phase_xas(mf, p, co.phase22) * co.inveta
    phi = phi + (fma(co.linb, mf, co.lina) + co.phiref22)
    a = amp_ansatz(mf, p, co.amp22)
    amp_as = kappa * a
    phi_as = jnp.where(mf < co.mf_t, fma(co.a0, mf, phi / 2.0 + alpha_pnr) + co.phi_a0, phi + co.phi_b0)
    xas22 = Struct(amp=co.amp22, phase=co.phase22, linb=co.linb)
    return co.amp_prefactor * p.m_seven_sixths * amp_as, co.phase_offset + phi_as, error, xas22
