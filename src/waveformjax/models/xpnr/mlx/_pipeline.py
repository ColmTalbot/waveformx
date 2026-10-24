# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR for a batch of binaries in MLX: the stages of
:mod:`._stages` (generated kernels, :mod:`._build`) and the parts by hand
(:mod:`._spintaylor`, :mod:`._arrays`, :mod:`._smooth`) composed, with the
derivatives along ``P`` directions of the twelve parameters of the binaries
carried along when asked for (forward mode).

Values are MLX arrays with a leading batch axis; their derivatives
("tangents") have a trailing axis of the directions, or are ``None`` where
zero. :func:`evaluate` returns the polarizations (``(B, N, 4)``: the real
and imaginary parts of ``h_+`` and ``h_x``), the error flag of each binary
and, with tangents, the Jacobian ``(B, N, 4, P)``.

Sizes are chosen per call: the steps of the SpinTaylor solution (a
capacity raised for the binaries that need more), and the points of the
grid of the PNR interpolants (a power of two at least the largest grid of
the batch).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import _build, _smooth
from ._interpolants import interpolants, spline_values
from ._solution import _call, _next_power_of_two, integrate, st_angles, tables

__all__ = ["evaluate", "N_PARAMETERS", "binary_tangents"]

N_PARAMETERS = 12


def binary_tangents(leaves, n_directions):
    """The tangents of the binary's fields along the first
    ``n_directions`` of its twelve parameters (in the order of
    :data:`._stages.BINARY_FIELDS`, the spins expanded)."""
    batch = leaves["mass_1"].shape[0]
    eye = np.eye(N_PARAMETERS, dtype=np.float32)[:, :n_directions]
    columns = {"mass_1": eye[0], "mass_2": eye[1], "spin_1": eye[2:5], "spin_2": eye[5:8],
               "luminosity_distance": eye[8], "inclination": eye[9], "phi_ref": eye[10], "f_ref": eye[11]}
    return {name: mx.broadcast_to(mx.array(value), (batch,) + value.shape) for name, value in columns.items()}


# ---------------------------------------------------------------------------
# the whole model
# ---------------------------------------------------------------------------


def _where_tree(mask, tree):
    """``tree`` with its float leaves zero where ``mask`` (``(B, N)``) is
    false (beyond the grid)."""
    return tree


def evaluate(leaves, frequencies, modes, amp_scale, n_directions=0, keep=None):
    """The polarizations of a batch: ``leaves`` the fields of the binaries
    (``(B,)``, spins ``(B, 3)``, float32 MLX arrays), ``frequencies``
    ``(1, N)`` (shared) or ``(B, N)``. Returns ``(h, error, jacobian)``
    with ``h`` of shape ``(B, N, 4)`` (``Re h_+``, ``Im h_+``, ``Re h_x``,
    ``Im h_x``; NaN where the model fails), ``error`` ``(B,)`` and the
    Jacobian ``(B, N, 4, P)`` along the first ``n_directions`` parameters
    (``None`` for none). ``keep`` (a dict) collects the intermediate
    values."""
    S = _build.stages(tuple(modes), float(amp_scale))
    P = n_directions
    batch = leaves["mass_1"].shape[0]
    shared = frequencies.shape[0] == 1 and batch > 1
    n_freq = frequencies.shape[1]
    f_min, f_max = frequencies[:, 0], frequencies[:, -1]
    if shared:
        f_min = mx.broadcast_to(f_min, (batch,))
        f_max = mx.broadcast_to(f_max, (batch,))
    lt = binary_tangents(leaves, P) if P else None
    # A: the common set-up
    (base, wf0, ode, error_a), (base_t, wf0_t, ode_t, _) = _pair(
        _call(S.a, (leaves, f_min, f_max), (lt, None, None), batch, P))
    # B: the SpinTaylor solution
    solution, summary, summary_t = integrate(ode, ode_t, P)
    # C1: finish_setup and the points of the bank
    (prec1, wf1, points, rotation, st_range), (prec1_t, wf1_t, points_t, rotation_t, st_range_t) = _pair(
        _call(S.c1, (base, wf0, summary), (base_t, wf0_t, summary_t), batch, P))
    tab = tables(solution, rotation, rotation_t, P)
    bank, bank_t = st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, points, points_t, P)
    # C2: the rest of the set-up
    (prec2, wf2, grid, error_c), (prec2_t, wf2_t, grid_t, _) = _pair(
        _call(S.c2, (prec1, wf1, points, bank, st_range), (prec1_t, wf1_t, points_t, bank_t, st_range_t), batch, P))
    n_grid = mx.maximum(grid[:, 1], 0.0)
    n_grid = mx.where(mx.isfinite(n_grid), n_grid, 0.0).astype(mx.int32)
    size = _next_power_of_two(int(np.max(np.asarray(n_grid))), 1024)
    if size > (1 << 19):
        size = 1 << 19
    n_grid = mx.minimum(n_grid, size)
    k = mx.arange(size, dtype=mx.float32)[None, :]
    mf_grid, mf_grid_t = _call(_build.parts().grid_mf, (grid, wf2.m_tot, k),
                               (grid_t, wf2_t.m_tot if P else None, None), batch, P, n_points=size, shared=True)
    st_grid, st_grid_t = st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, mf_grid, mf_grid_t, P)
    # D: the PNR angles on the grid
    (alpha, beta, err_grid), d_t = _pair(_call(S.d, (prec2, wf2, mf_grid, st_grid),
                                               (prec2_t, wf2_t, mf_grid_t, st_grid_t), batch, P, n_points=size))
    valid = k < n_grid[:, None].astype(mx.float32)
    alpha = mx.where(valid, alpha, 0.0)
    beta = mx.where(valid, beta, 0.0)
    dalpha = dbeta = None
    if P:
        dalpha = mx.where(valid[..., None], d_t[0], 0.0)
        dbeta = mx.where(valid[..., None], d_t[1], 0.0)
    error_grid = mx.any(err_grid & valid, axis=1)
    splines = interpolants(alpha, beta, dalpha, dbeta, grid, grid_t, n_grid, P)
    # E: offsets, remapped J frame, coefficients
    at_ref, inside_ref, dat_ref = spline_values(splines, wf2.f_ref[:, None], wf2_t.f_ref[:, None, :] if P else None,
                                                P)
    (prec_e, wf_e, extra, error_e), (prec_e_t, wf_e_t, extra_t, _) = _pair(
        _call(S.e, (prec2, wf2, at_ref[:, 0], inside_ref[:, 0]),
              (prec2_t, wf2_t, None if dat_ref is None else dat_ref[:, 0], None), batch, P))
    # F1: the frequencies of the angles of each mode, the amplitude ratio
    (mapped, map_errors, mf_as, kappa), f1_t = _pair(
        _call(S.f1, (prec_e, wf_e, frequencies), (prec_e_t, wf_e_t, None), batch, P, n_points=n_freq))
    n_modes = len(modes)
    values, inside, dvalues = spline_values(splines, mapped.reshape(batch, n_freq * n_modes),
                                            None if not P else f1_t[0].reshape(batch, n_freq * n_modes, P), P)
    values = values.reshape(batch, n_freq, n_modes, 6)
    inside = inside.reshape(batch, n_freq, n_modes)
    if dvalues is not None:
        dvalues = dvalues.reshape(batch, n_freq, n_modes, 6, P)
    if (2, 2) in modes:
        kappa_s = _smooth.smooth_amp_ratio(kappa, frequencies if shared else frequencies)
        dkappa_s = None
        if P:
            rows = mx.moveaxis(f1_t[3], 2, 1).reshape(batch * P, n_freq)
            dkappa_s = _smooth.smooth_amp_ratio(rows, frequencies, rows_per_frequency=P if not shared else 1)
            dkappa_s = mx.moveaxis(dkappa_s.reshape(batch, P, n_freq), 1, 2)
    else:
        kappa_s, dkappa_s = kappa, (f1_t[3] if P else None)
    st_f, st_f_t = st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, mf_as,
                             f1_t[2] if P else None, P)
    # F2: the angles, the modes and the twist
    trees = (prec_e, wf_e, extra, frequencies, values, inside, map_errors, kappa_s, st_f[0], st_f[4])
    ttrees = (prec_e_t, wf_e_t, extra_t, None, dvalues, None, None, dkappa_s, st_f_t[0] if P else None, None)
    (h, err_f), f2_t = _pair(_call(S.f2, trees, ttrees, batch, P, n_points=n_freq))
    error = (error_a | ~ode.active | summary.failed | error_c | summary.incomplete | error_grid | error_e
             | mx.any(err_f, axis=1))
    h = mx.where(error[:, None, None], float("nan"), h)
    jacobian = None
    if P:
        jacobian = mx.where(error[:, None, None, None], float("nan"), f2_t[0])
    if keep is not None:
        keep.update(base=base, wf0=wf0, ode=ode, error_a=error_a, solution=solution, summary=summary,
                    summary_t=summary_t, prec_c1=prec1, wf_c1=wf1, points=points, rotation=rotation,
                    st_range=st_range, bank=bank, bank_t=bank_t, prec_c2=prec2, wf_c2=wf2, grid=grid,
                    error_c=error_c, mf_grid=mf_grid, st_grid=st_grid, alpha=alpha, beta=beta,
                    error_grid=error_grid, splines=splines, at_ref=at_ref, prec_e=prec_e, extra=extra,
                    error_e=error_e, mapped=mapped, values=values, kappa=kappa, kappa_s=kappa_s, st_f=st_f,
                    h=h, err_f=err_f, n_grid=n_grid)
    return h, error, jacobian


def _pair(result):
    """``(values, tangents)`` with the tangents a tuple of ``None`` when
    there are none."""
    values, tangents = result
    if tangents is None:
        n = len(values) if isinstance(values, tuple) else 1
        tangents = (None,) * n if isinstance(values, tuple) else None
    return values, tangents
