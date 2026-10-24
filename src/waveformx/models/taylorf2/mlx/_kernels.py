# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The Metal kernels of TaylorF2 (:mod:`._tables`): the set-up generated
from :func:`..core.table`, the waveform at one frequency written by hand
(:func:`..core.evaluate`: eight terms of the phase, the leading term in
double-float arithmetic, one power for the amplitude)."""

from __future__ import annotations

import functools

from .. import core
from ._tables import TableKernels

__all__ = ["kernels"]

_CONSTANTS = f"""
constant constexpr int NB = {len(core.BASIS)};
constant constexpr int CONST_TERM = {core.CONST};
"""

#: The waveform at one frequency and its derivative along one direction
#: (the basis is that of core.BASIS).
POINT = """
struct Point {
    float m76, amp, c, sn;
    float basis[NB];
    DF b0;
};

inline void model_point(float f, const device float* T, thread Point& p) {
    float m1 = 1.0f / f;
    float lf = metal::precise::log(f);
    // (the cube root as in ..xas.core.frequency_powers)
    float t = metal::precise::exp(lf / 3.0f);
    t = t * (4.0f - t * t * t * m1) / 3.0f;
    p.m76 = m1 / metal::precise::sqrt(t);
    float t2 = t * t;
    float mt1 = t2 * m1;
    thread float* b = p.basis;
    b[0] = m1; b[1] = mt1 * mt1; b[2] = mt1; b[3] = lf; b[4] = 1.0f; b[5] = t; b[6] = t * lf; b[7] = t2;
    float rest = T[O_PHASE + CONST_TERM];
    for (int i = 0; i < NB; ++i) {
        if (i != CONST_TERM) rest = metal::fma(T[O_PHASE + i], b[i], rest);
    }
    p.b0 = df_m53(f);
    DF phi = df_add(df_mul(DF{T[O_BIG], T[O_BIG_LO]}, p.b0), DF{rest, 0.0f});
    phi = df_add(phi, DF{-T[O_BIG + 1], -T[O_BIG_LO + 1]});
    float r = reduce_phase(phi);
    p.amp = T[O_AMP] * p.m76;
    p.c = metal::precise::cos(r);
    p.sn = metal::precise::sin(r);
}

inline float4 model_tangent(thread const Point& p, const device float* T, const device float* dT) {
    float dphi = dT[(O_PHASE + CONST_TERM) * ND];
    for (int i = 0; i < NB; ++i) {
        if (i != CONST_TERM) dphi = metal::fma(dT[(O_PHASE + i) * ND], p.basis[i], dphi);
    }
    dphi = metal::fma(dT[O_BIG * ND], p.b0.hi, dphi) - dT[(O_BIG + 1) * ND];
    float damp = dT[O_AMP * ND] * p.m76;
    return polarization_tangent(p.amp, p.c, p.sn, damp, dphi, T, dT);
}
"""

#: The reference at f_ref, none at f_ref = 0 (coalescence).
ANCHOR = "float f_ref = x[7]{v}; bool anchored = f_ref > 0.0f;"


def _table(params, f_min, amp_scale):
    return core.table(params, amp_scale)


@functools.lru_cache(maxsize=None)
def kernels(amp_scale, fast=True):
    """The kernels of TaylorF2 for the scale of the strain ``amp_scale``."""
    return TableKernels("taylorf2", core, _table, POINT, ANCHOR, _CONSTANTS, amp_scale, fast)
