# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The generated kernels of the stages of the model
(:mod:`._stages`) for a set of co-precessing modes: traced in float32 from
shapes alone (no JAX compilation), each stage at the structure of the
previous one's outputs. Built once per set of modes, strain scale and
reference convention (:func:`stages`)."""

from __future__ import annotations

import functools

import jax
import numpy as np

from .._math import Struct
from . import _stages
from ._kernels import StageKernel

__all__ = ["stages", "parts"]

F32 = np.float32


def _s(shape=(), dtype=F32):
    return jax.ShapeDtypeStruct(shape, dtype)


def summary_example():
    """The structure of :func:`._stages.solution_summary`."""
    return Struct(s1_end=_s((3,)), s2_end=_s((3,)), lnhat_end=_s((3,)), f_min_pn=_s(), omega_first=_s(),
                  omega_last=_s(), failed=_s((), np.bool_), incomplete=_s((), np.bool_))


class Built:
    """The kernels of the stages (built on first use)."""

    def __init__(self, modes, amp_scale, reference_mf=None):
        self.modes = modes
        self.amp_scale = amp_scale
        self.reference_mf = reference_mf
        self.n_modes = len(modes)

    @functools.cached_property
    def a(self):
        leaves = {name: _s((3,) if name.startswith("spin") else ()) for name in _stages.BINARY_FIELDS}
        modes, amp_scale, reference_mf = self.modes, self.amp_scale, self.reference_mf
        return StageKernel(lambda l, f0, f1: _stages.stage_a(l, f0, f1, modes, amp_scale, reference_mf),
                           (leaves, _s(), _s()), 3, "xpnr_a")

    @functools.cached_property
    def c1(self):
        base, wf0, _, _ = self.a.out_example()
        return StageKernel(_stages.stage_c1, (base, wf0, summary_example()), 3, "xpnr_c1")

    @functools.cached_property
    def c2(self):
        prec, wf, points, _, st_range = self.c1.out_example()
        bank = (_s((15,)), _s((15,)), _s((15,)), _s((15,)), _s((15,), np.bool_))
        modes = self.modes
        return StageKernel(lambda p, w, pts, v, r: _stages.stage_c2(p, w, pts, v, r, modes),
                           (prec, wf, points, bank, st_range), 5, "xpnr_c2")

    @functools.cached_property
    def d(self):
        prec, wf, _, _ = self.c2.out_example()
        st_values = (_s(), _s(), _s(), _s(), _s((), np.bool_))
        return StageKernel(_stages.grid_point, (prec, wf, _s(), st_values), 2, "xpnr_d")

    @functools.cached_property
    def e(self):
        prec, wf, _, _ = self.c2.out_example()
        modes = self.modes
        return StageKernel(lambda p, w, a, i: _stages.stage_e(p, w, a, i, modes),
                           (prec, wf, _s((len(_stages.SPLINES),)), _s((), np.bool_)), 4, "xpnr_e")

    @functools.cached_property
    def f1(self):
        prec, wf, _, _ = self.e.out_example()
        modes = self.modes
        return StageKernel(lambda p, w, f: _stages.stage_f1(p, w, f, modes), (prec, wf, _s()), 2, "xpnr_f1")

    @functools.cached_property
    def f2(self):
        prec, wf, extra, _ = self.e.out_example()
        modes, n = self.modes, self.n_modes
        examples = (prec, wf, extra, _s(), _s((n, len(_stages.SPLINES))), _s((n,), np.bool_), _s((n,), np.bool_),
                    _s(), _s(), _s((), np.bool_))
        return StageKernel(lambda p, w, e, f, v, i, me, k, sa, si: _stages.stage_f2(p, w, e, f, v, i, me, k, sa, si,
                                                                                     modes),
                           examples, 3, "xpnr_f2")


def _f2_pieces(built):
    prec, wf, extra, _ = built.e.out_example()
    modes, n = built.modes, built.n_modes
    six = _s((len(_stages.SPLINES),))
    examples = (prec, wf, extra, _s((3,)), _s(), _s((n,)), _s((n,)), _s((n,), np.bool_), _s(), _s(),
                _s((), np.bool_)) + (six,) * (4 * n)

    def function(p, w, e, g, f, x, k, me, kappa, sa, si, *pieces):
        return _stages.stage_f2_pieces(p, w, e, g, f, x, k, me, kappa, sa, si, *pieces, modes=modes)

    return StageKernel(function, examples, 4, "xpnr_f2p")


Built.f2p = functools.cached_property(_f2_pieces)
Built.f2p.__set_name__(Built, "f2p")


class Parts:
    """The kernels of the arithmetic of the parts by hand at one point
    (:mod:`._stages`), independent of the modes."""

    @functools.cached_property
    def st_newton(self):
        return StageKernel(_stages.st_newton, (_s(), _s((4,)), _s((7, 4)), _s()), 0, "xpnr_st_newton",
                           tangents=False)

    @functools.cached_property
    def st_point_angles(self):
        return StageKernel(_stages.st_point_angles,
                           (_s((3,)), _s((3,)), _s(), _s((4,)), _s((7, 4)), _s(), _s(), _s(), _s()), 2,
                           "xpnr_st_angles")

    @functools.cached_property
    def st_node_raw(self):
        return StageKernel(_stages.st_node_raw, (_s((3,)), _s((3,))), 1, "xpnr_st_node")

    @functools.cached_property
    def grid_mf(self):
        return StageKernel(lambda grid, m_tot, k: _stages.grid_frequencies(k, grid, m_tot), (_s((3,)), _s(), _s()),
                           2, "xpnr_grid_mf")

    @functools.cached_property
    def uvw(self):
        return StageKernel(_stages.uvw, (_s(), _s()), 0, "xpnr_uvw")

    @functools.cached_property
    def boole(self):
        return StageKernel(_stages.boole_interval, (_s((3,)), _s(), _s((7,)), _s((7,))), 1, "xpnr_boole")

    @functools.cached_property
    def spline_point6(self):
        six = _s((len(_stages.SPLINES),))
        return StageKernel(_stages.spline_point6, (_s((3,)), _s(), _s(), six, six, six, six), 1, "xpnr_spline")


@functools.lru_cache(maxsize=None)
def parts():
    """The :class:`Parts` kernels."""
    return Parts()


@functools.lru_cache(maxsize=None)
def stages(modes, amp_scale, reference_mf=None):
    """The :class:`Built` kernels for the co-precessing ``modes`` (a tuple),
    the scale of the strain and the mass-scaled reference frequency."""
    return Built(tuple(modes), float(amp_scale), None if reference_mf is None else float(reference_mf))
