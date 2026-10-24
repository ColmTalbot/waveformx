# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The Metal kernels of IMRPhenomD (:mod:`...taylorf2.mlx._tables`): the
set-up generated from :func:`..core.table`, the waveform at one frequency
written by hand (:func:`..core.evaluate`: the phase and amplitude of one of
three regions each, the leading term of the inspiral phase in double-float
arithmetic)."""

from __future__ import annotations

import functools

from ...taylorf2.mlx._tables import TableKernels, kernel_name, mass_scaled
from .. import core

__all__ = ["kernels"]


def _constants():
    lines = [f"constant constexpr int NB = {len(core.BASIS)};",
             f"constant constexpr int CONST_TERM = {core.CONST};"]
    for r, terms in enumerate(core.REGION_TERMS):
        lines.append(f"constant constexpr int NTERMS{r} = {len(terms)};")
        lines.append(f"constant constexpr int TERMS{r}[NTERMS{r}] = {{{', '.join(str(k) for k in terms)}}};")
    return "\n".join(lines) + "\n"


#: The waveform at one frequency and its derivative along one direction
#: (the basis is that of core.BASIS).
POINT = """
struct Point {
    float f, t, m76;
    float basis[NB];
    int rp, ra;
    float z, at, y, poly_d, u, e, s;
    float amp, c, sn;
    DF b0;
};

// sum_k row[k stride] basis[k] over the terms of region r and the constant
inline float phenomd_phase_sum(int r, const device float* row, thread const float* b, uint stride) {
    float total = row[CONST_TERM * stride];
    if (r == 0) {
        for (int i = 0; i < NTERMS0; ++i) total = metal::fma(row[TERMS0[i] * stride], b[TERMS0[i]], total);
    } else if (r == 1) {
        for (int i = 0; i < NTERMS1; ++i) total = metal::fma(row[TERMS1[i] * stride], b[TERMS1[i]], total);
    } else {
        for (int i = 0; i < NTERMS2; ++i) total = metal::fma(row[TERMS2[i] * stride], b[TERMS2[i]], total);
    }
    return total;
}

inline void model_point(float f, const device float* T, thread Point& p) {
    p.f = f;
    float m1 = 1.0f / f;
    float lf = metal::precise::log(f);
    // (the cube root as in ..xas.core.frequency_powers)
    float t = metal::precise::exp(lf / 3.0f);
    t = t * (4.0f - t * t * t * m1) / 3.0f;
    p.t = t;
    p.m76 = m1 / metal::precise::sqrt(t);
    float t2 = t * t;
    float mt1 = t2 * m1;
    float root = metal::precise::sqrt(f);
    thread float* b = p.basis;
    b[0] = m1; b[1] = mt1 * mt1; b[2] = mt1; b[3] = lf; b[4] = 1.0f; b[5] = t; b[6] = t * lf; b[7] = t2;
    b[8] = f; b[9] = f * t; b[10] = f * t2; b[11] = f * f; b[12] = m1 * m1 * m1;
    b[13] = root * metal::precise::sqrt(root);
    int rp = f < T[O_F_PHASE_INS] ? 0 : (f >= T[O_F_PHASE_MRD] ? 2 : 1);
    int ra = f < T[O_F_AMP_INS] ? 0 : (f >= T[O_F_AMP_MRD] ? 2 : 1);
    p.rp = rp;
    p.ra = ra;
    float rest = phenomd_phase_sum(rp, T + O_PHASE + rp * NB, b, 1);
    p.z = 0.0f;
    p.at = 0.0f;
    if (rp == 2) {
        p.z = T[O_ATAN_W] * (f - T[O_ATAN_SHIFT]);
        p.at = metal::precise::atan(p.z);
        rest = metal::fma(T[O_ATAN_A], p.at, rest);
    }
    p.b0 = df_m53(f);
    DF phi = DF{rest, 0.0f};
    if (rp == 0) {
        phi = df_add(df_mul(DF{T[O_BIG], T[O_BIG_LO]}, p.b0), phi);
    }
    phi = df_add(phi, DF{-T[O_BIG + 1], -T[O_BIG_LO + 1]});
    float amp;
    if (ra == 0) {
        float poly = T[O_AMP_INS + 9];
        for (int k = 8; k >= 0; --k) poly = metal::fma(poly, t, T[O_AMP_INS + k]);
        amp = poly;
    } else if (ra == 1) {
        float y = T[O_AMP_INT_S] * f;
        float poly = T[O_AMP_INT + 4];
        float poly_d = 4.0f * T[O_AMP_INT + 4];
        for (int j = 3; j >= 0; --j) {
            poly = metal::fma(poly, y, T[O_AMP_INT + j]);
            if (j > 0) poly_d = metal::fma(poly_d, y, float(j) * T[O_AMP_INT + j]);
        }
        p.y = y;
        p.poly_d = poly_d;
        amp = poly;
    } else {
        float u = T[O_AMP_RD_S] * (f - T[O_AMP_RD_SHIFT]);
        float e = metal::precise::exp(-T[O_AMP_RD_GR] * u);
        float s = metal::fma(u, u, 1.0f);
        p.u = u;
        p.e = e;
        p.s = s;
        amp = T[O_AMP_RD_N] * e / s;
    }
    p.amp = p.m76 * amp;
    float r = reduce_phase(phi);
    p.c = metal::precise::cos(r);
    p.sn = metal::precise::sin(r);
}

inline float4 model_tangent(thread const Point& p, const device float* T, const device float* dT) {
    int rp = p.rp;
    float dphi = phenomd_phase_sum(rp, dT + (O_PHASE + rp * NB) * ND, p.basis, ND);
    if (rp == 0) {
        dphi = metal::fma(dT[O_BIG * ND], p.b0.hi, dphi);
    } else if (rp == 2) {
        float w = T[O_ATAN_W];
        float dz = dT[O_ATAN_W * ND] * (p.f - T[O_ATAN_SHIFT]) - w * dT[O_ATAN_SHIFT * ND];
        dphi += dT[O_ATAN_A * ND] * p.at + T[O_ATAN_A] * dz / metal::fma(p.z, p.z, 1.0f);
    }
    dphi -= dT[(O_BIG + 1) * ND];
    float damp;
    if (p.ra == 0) {
        float d = dT[(O_AMP_INS + 9) * ND];
        for (int k = 8; k >= 0; --k) d = metal::fma(d, p.t, dT[(O_AMP_INS + k) * ND]);
        damp = d;
    } else if (p.ra == 1) {
        float d = dT[(O_AMP_INT + 4) * ND];
        for (int j = 3; j >= 0; --j) d = metal::fma(d, p.y, dT[(O_AMP_INT + j) * ND]);
        damp = metal::fma(p.poly_d, dT[O_AMP_INT_S * ND] * p.f, d);
    } else {
        float s = T[O_AMP_RD_S];
        float du = dT[O_AMP_RD_S * ND] * (p.f - T[O_AMP_RD_SHIFT]) - s * dT[O_AMP_RD_SHIFT * ND];
        float value = T[O_AMP_RD_N] * p.e / p.s;
        damp = p.e / p.s * dT[O_AMP_RD_N * ND]
            + value * (-dT[O_AMP_RD_GR * ND] * p.u - T[O_AMP_RD_GR] * du - 2.0f * p.u * du / p.s);
    }
    return polarization_tangent(p.amp, p.c, p.sn, p.m76 * damp, dphi, T, dT);
}
"""

def anchor(reference_mf):
    """The reference at f_ref (f_min for f_ref = 0), anchored in the
    inspiral; with the mass-scaled reference from the masses."""
    if reference_mf is None:
        return ("float f_ref = x[7]{v} == 0.0f ? x[8]{v} : x[7]{v}; "
                "bool anchored = f_ref < y[O_F_PHASE_INS]{v}; DF mass_scaled = DF{{0.0f, 0.0f}};")
    inspiral = "true" if reference_mf < core.F_PHASE_INS else "false"
    return (f"float f_ref = 0.0f; bool anchored = {inspiral}; DF mass_scaled = "
            + mass_scaled(reference_mf).replace("{", "{{").replace("}", "}}") + ";")


@functools.lru_cache(maxsize=None)
def kernels(amp_scale, reference_mf=None, fast=True):
    """The kernels of IMRPhenomD for the scale of the strain ``amp_scale``
    and the reference convention (``reference_mf``)."""

    def table(params, f_min, scale):
        return core.table(params, f_min, scale, reference_mf)

    return TableKernels(kernel_name("phenomd", reference_mf), core, table, POINT, anchor(reference_mf), _constants(),
                        amp_scale, fast)
