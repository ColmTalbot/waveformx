# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
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

The batch is evaluated in two parts: :func:`setup` (the common set-up, the
SpinTaylor solution and the set-up of version 330, up to the grid of the
PNR interpolants) for the whole batch, then :func:`finish` (the angles on
the grid, the interpolants and the waveform) for groups of binaries whose
grids have similar sizes (:func:`_groups`). Sizes are chosen per call,
from values read back from the GPU: the steps of the SpinTaylor solution
(a capacity raised for the binaries that need more) and the points of the
grids (a power of two at least the largest grid of the group). Large
batches go in chunks (:func:`evaluate_chunked`).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import _build, _smooth
from ._interpolants import ends_gather, interpolants, intervals, spline_values
from ._kernels import Packed
from ._solution import _call, _next_power_of_two, _run, integrate, st_angles, tables

__all__ = ["evaluate", "evaluate_chunked", "setup", "finish", "N_PARAMETERS", "binary_tangents"]

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


#: The points of the grids of the PNR interpolants evaluated together (the
#: binaries are grouped by the size of their grid, and the groups split
#: below this many points): with tangents and without.
POINTS_PER_GROUP = {True: 1 << 21, False: 1 << 24}
#: The largest grid of the PNR interpolants (JAX's capacity).
MAX_GRID = 1 << 19


def _select(tree, index):
    """The binaries ``index`` of a pytree of MLX arrays with a leading
    batch axis (``None`` and Python leaves kept), or of a
    :class:`._kernels.Packed`."""
    import jax

    if isinstance(tree, Packed):
        return tree.select(index)

    def pick(x):
        return x[index] if isinstance(x, mx.array) and x.ndim > 0 else x

    return jax.tree_util.tree_map(pick, tree, is_leaf=lambda x: x is None)


#: The cost of a group of binaries (its launches) in points of the grid.
GROUP_COST = 1 << 19


def _groups(n_grid, n_directions):
    """The binaries in groups by the size of their grid: ``[(indices,
    size)]`` with ``size`` a power of two at least each grid of the group.
    The binaries sorted by size are cut into runs that minimise the padded
    points plus a cost per group (:data:`GROUP_COST`: its kernels are
    launched one after the other), each run within the budget of
    :data:`POINTS_PER_GROUP`."""
    budget = POINTS_PER_GROUP[bool(n_directions)]
    sizes = np.array([min(_next_power_of_two(max(int(n), 1), 1024), MAX_GRID) for n in n_grid])
    distinct = sorted(set(sizes.tolist()))
    counts = [int(np.sum(sizes == size)) for size in distinct]
    # cheapest cut of the distinct sizes into runs (each padded to its largest)
    best = [0.0] + [np.inf] * len(distinct)
    cut = [0] * (len(distinct) + 1)
    for j in range(1, len(distinct) + 1):
        for i in range(j):
            n = sum(counts[i:j])
            cost = best[i] + GROUP_COST * max(1, -(-n * distinct[j - 1] // budget)) + n * distinct[j - 1]
            if cost < best[j]:
                best[j], cut[j] = cost, i
    runs, j = [], len(distinct)
    while j > 0:
        runs.append((cut[j], j))
        j = cut[j]
    out = []
    for i, j in reversed(runs):
        size = distinct[j - 1]
        index = np.nonzero(np.isin(sizes, distinct[i:j]))[0]
        per = max(1, budget // size)
        for start in range(0, len(index), per):
            out.append((index[start:start + per], int(size)))
    return out


class _Setup:
    """The set-up of a batch, up to the grid of the PNR interpolants."""


def setup(leaves, frequencies, S, n_directions):
    """Stages A, B (the SpinTaylor solution), C1, the bank and C2 for the
    whole batch."""
    P = n_directions
    batch = leaves["mass_1"].shape[0]
    f_min, f_max = frequencies[:, 0], frequencies[:, -1]
    if frequencies.shape[0] == 1 and batch > 1:
        f_min = mx.broadcast_to(f_min, (batch,))
        f_max = mx.broadcast_to(f_max, (batch,))
    lt = binary_tangents(leaves, P) if P else None
    st = _Setup()
    # A: the common set-up
    base, wf0, ode, error_a = _run(S.a, (leaves, f_min, f_max), (lt, None, None), batch, P)
    st.ode = ode.tree()
    st.error_a = error_a.tree()
    # B: the SpinTaylor solution
    st.solution, st.summary, summary_t = integrate(st.ode, ode.tangent_tree() if P else None, P)
    # C1: finish_setup and the points of the bank
    prec1, wf1, points, rotation, st_range = _run(S.c1, (base, wf0, st.summary), (None, None, summary_t), batch, P)
    st.rotation, st.rotation_t = rotation.tree(), rotation.tangent_tree() if P else None
    st.st_range, st.st_range_t = st_range.tree(), st_range.tangent_tree() if P else None
    st.tables = tables(st.solution, st.rotation, st.rotation_t, P)
    bank, bank_t = st_angles(st.solution, st.tables, st.st_range, st.st_range_t, st.rotation, st.rotation_t,
                             points.tree(), points.tangent_tree() if P else None, P)
    # C2: the rest of the set-up
    st.prec2, st.wf2, grid, error_c = _run(S.c2, (prec1, wf1, points, bank, st_range),
                                           (None, None, None, bank_t, None), batch, P)
    st.grid, st.grid_t = grid.tree(), grid.tangent_tree() if P else None
    st.error_c = error_c.tree()
    n_grid = mx.maximum(st.grid[:, 1], 0.0)
    st.n_grid = mx.minimum(mx.where(mx.isfinite(n_grid), n_grid, 0.0), MAX_GRID).astype(mx.int32)
    st.error = st.error_a | ~st.ode.active | st.summary.failed | st.error_c | st.summary.incomplete
    st.extra = dict(summary_t=summary_t, points=points, bank=bank, bank_t=bank_t)
    return st


def finish(st, index, size, frequencies, S, modes, n_directions):
    """The grid of the PNR interpolants of size ``size``, the interpolants
    and the waveform of the binaries ``index`` of the set-up ``st``."""
    P = n_directions
    batch = len(index)
    whole = batch == st.n_grid.shape[0] and np.array_equal(index, np.arange(batch))
    idx = mx.array(np.asarray(index, np.int32))

    def sel(x):
        return x if whole else _select(x, idx)

    if frequencies.shape[0] > 1:
        frequencies = sel(frequencies)
    n_freq = frequencies.shape[1]
    solution, tab = sel(st.solution), sel(st.tables)
    rotation, rotation_t = sel(st.rotation), sel(st.rotation_t)
    st_range, st_range_t = sel(st.st_range), sel(st.st_range_t)
    prec2, wf2 = sel(st.prec2), sel(st.wf2)
    grid, grid_t = sel(st.grid), sel(st.grid_t)
    wf2_tree, wf2_t = wf2.tree(), wf2.tangent_tree() if P else None
    n_grid = mx.minimum(sel(st.n_grid), size)
    k = mx.arange(size, dtype=mx.float32)[None, :]
    mf_grid, mf_grid_t = _call(_build.parts().grid_mf, (grid, wf2_tree.m_tot, k),
                               (grid_t, wf2_t.m_tot if P else None, None), batch, P, n_points=size, shared=True)
    st_grid, st_grid_t = st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, mf_grid, mf_grid_t, P)
    # D: the PNR angles on the grid
    (alpha, beta, err_grid), d_t = _call(S.d, (prec2, wf2, mf_grid, st_grid), (None, None, mf_grid_t, st_grid_t),
                                         batch, P, n_points=size, counts=n_grid)
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
    at_ref, inside_ref, dat_ref = spline_values(splines, wf2_tree.f_ref[:, None],
                                                wf2_t.f_ref[:, None, :] if P else None, P)
    prec_e, wf_e, extra, error_e = _run(S.e, (prec2, wf2, at_ref[:, 0], inside_ref[:, 0]),
                                        (None, None, None if dat_ref is None else dat_ref[:, 0], None), batch, P)
    # F1: the frequencies of the angles of each mode, the amplitude ratio
    (mapped, map_errors, mf_as, kappa), f1_t = _call(S.f1, (prec_e, wf_e, frequencies), (None, None, None), batch,
                                                     P, n_points=n_freq)
    n_modes = len(modes)
    k_modes, k_index = intervals(splines, mapped)
    shared = frequencies.shape[0] == 1
    if (2, 2) in modes:
        kappa_s = _smooth.smooth_amp_ratio(kappa, frequencies)
        dkappa_s = None
        if P:
            rows = mx.moveaxis(f1_t[3], 2, 1).reshape(batch * P, n_freq)
            dkappa_s = _smooth.smooth_amp_ratio(rows, frequencies, rows_per_frequency=1 if shared else P)
            dkappa_s = mx.moveaxis(dkappa_s.reshape(batch, P, n_freq), 1, 2)
    else:
        kappa_s, dkappa_s = kappa, (f1_t[3] if P else None)
    st_f, st_f_t = st_angles(solution, tab, st_range, st_range_t, rotation, rotation_t, mf_as,
                             f1_t[2] if P else None, P)
    # F2: the interpolants at the frequencies of the modes (read from their
    # table in the kernel), the angles, the modes and the twist
    gather = dict(table=splines.table, table_t=splines.dtable, index=k_index, args={})
    for m in range(n_modes):
        gather["args"].update(ends_gather(11 + 4 * m, slot=m))
    args = (prec_e, wf_e, extra, splines.grid, frequencies, mapped, k_modes, map_errors, kappa_s, st_f[0], st_f[4])
    args = args + (None,) * (4 * n_modes)
    targs = None
    if P:
        targs = (None, None, None, splines.dgrid, None, f1_t[0], None, None, dkappa_s, st_f_t[0], None)
        targs = targs + (None,) * (4 * n_modes)
    h_out, err_out = S.f2p.run(args, targs, batch=batch, n_directions=P, n_points=n_freq, gather=gather)
    h, err_f = h_out.tree(), err_out.tree()
    f2_t = (h_out.tangent_tree(),) if P else None
    error = error_grid | error_e.tree() | mx.any(err_f, axis=1)
    intermediate = dict(mf_grid=mf_grid, st_grid=st_grid, alpha=alpha, beta=beta, error_grid=error_grid,
                        splines=splines, at_ref=at_ref, prec_e=prec_e, extra=extra, error_e=error_e, mapped=mapped,
                        kappa=kappa, kappa_s=kappa_s, st_f=st_f, err_f=err_f)
    return h, error, (f2_t[0] if P else None), intermediate


#: The binaries evaluated together, without and with tangents (memory: the
#: tangents of the interpolants' values at the frequencies of a batch of 64
#: take ~0.7 GB at 8000 frequencies).
CHUNK = {False: 1024, True: 64}


def evaluate_chunked(leaves, frequencies, modes, amp_scale, n_directions=0, reference_mf=None):
    """:func:`evaluate` in chunks of :data:`CHUNK` binaries."""
    batch = leaves["mass_1"].shape[0]
    chunk = CHUNK[bool(n_directions)]
    if batch <= chunk:
        return evaluate(leaves, frequencies, modes, amp_scale, n_directions, reference_mf=reference_mf)
    outs = []
    for start in range(0, batch, chunk):
        part = {name: value[start:start + chunk] for name, value in leaves.items()}
        f = frequencies if frequencies.shape[0] == 1 else frequencies[start:start + chunk]
        out = evaluate(part, f, modes, amp_scale, n_directions, reference_mf=reference_mf)
        mx.eval([x for x in out if x is not None])
        outs.append(out)
        # (the buffers of a chunk freed for the next: MLX keeps them in its
        # cache otherwise, and the cache of large chunks slows the device)
        mx.clear_cache()
    h = mx.concatenate([o[0] for o in outs])
    error = mx.concatenate([o[1] for o in outs])
    jacobian = mx.concatenate([o[2] for o in outs]) if n_directions else None
    return h, error, jacobian


def evaluate(leaves, frequencies, modes, amp_scale, n_directions=0, keep=None, reference_mf=None):
    """The polarizations of a batch: ``leaves`` the fields of the binaries
    (``(B,)``, spins ``(B, 3)``, float32 MLX arrays), ``frequencies``
    ``(1, N)`` (shared) or ``(B, N)``. Returns ``(h, error, jacobian)``
    with ``h`` of shape ``(B, N, 4)`` (``Re h_+``, ``Im h_+``, ``Re h_x``,
    ``Im h_x``; NaN where the model fails), ``error`` ``(B,)`` and the
    Jacobian ``(B, N, 4, P)`` along the first ``n_directions`` parameters
    (``None`` for none). ``keep`` (a dict) collects the intermediate
    values (of the set-up, and of the last group of binaries);
    ``reference_mf`` the mass-scaled reference frequency, if any."""
    S = _build.stages(tuple(modes), float(amp_scale), reference_mf)
    P = n_directions
    batch = leaves["mass_1"].shape[0]
    n_freq = frequencies.shape[1]
    st = setup(leaves, frequencies, S, P)
    groups = _groups(np.asarray(st.n_grid), P)
    if len(groups) == 1:
        h, error, jacobian, intermediate = finish(st, groups[0][0], groups[0][1], frequencies, S, modes, P)
    else:
        h = mx.zeros((batch, n_freq, 4))
        error = mx.zeros((batch,), dtype=mx.bool_)
        jacobian = mx.zeros((batch, n_freq, 4, P)) if P else None
        for index, size in groups:
            hg, eg, jg, intermediate = finish(st, index, size, frequencies, S, modes, P)
            idx = mx.array(np.asarray(index, np.int32))
            h[idx] = hg
            error[idx] = eg
            if P:
                jacobian[idx] = jg
            mx.eval(h, error, *([jacobian] if P else []))
    error = error | st.error
    h = mx.where(error[:, None, None], float("nan"), h)
    if P:
        jacobian = mx.where(error[:, None, None, None], float("nan"), jacobian)
    if keep is not None:
        keep.update(ode=st.ode, error_a=st.error_a, solution=st.solution, summary=st.summary, rotation=st.rotation,
                    st_range=st.st_range, prec_c2=st.prec2, wf_c2=st.wf2, grid=st.grid, error_c=st.error_c,
                    n_grid=st.n_grid, groups=groups, **st.extra, **intermediate)
    return h, error, jacobian


def _pair(result):
    """``(values, tangents)`` with the tangents a tuple of ``None`` when
    there are none."""
    values, tangents = result
    if tangents is None:
        n = len(values) if isinstance(values, tuple) else 1
        tangents = (None,) * n if isinstance(values, tuple) else None
    return values, tangents
