# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The integration of the dynamics as one Metal kernel, with its Jacobian.

Written with MLX operations (:mod:`._dynamics`), the RK4 and Adams-Bashforth
steps are thousands of small operations launched one after the other,
which cost about the same whatever the size of the batch. Here the whole
integration from the reference time (the Euler step to the nearest node,
the RK4 start and the Adams-Bashforth steps, as
:func:`._dynamics.states_from_reference`) runs in one kernel.

Every binary is integrated by one SIMD group of 32 threads. Lane ``s <
11`` holds component ``s`` of the state, of the RK4 stages and of the
Adams-Bashforth history (so that a thread holds a few dozen numbers rather
than every vector, which spilled from the registers and made the kernel
twice as slow), and the lanes exchange them by shuffles: each derivative
gathers the state on every lane, splits the monomials of the fits among
the lanes (each monomial from its packed exponents, :func:`_orders`), sums
the nine fits with ``simd_sum`` and assembles the component of each lane.

The same source gives two kernels (:func:`_kernel`): on floats for the
states, and on dual numbers for their Jacobian, where a SIMD group carries
the derivative of the whole integration with respect to one of its eight
parameters ``(q, chi_a, chi_b, t_ref)`` (forward mode), so that eight groups
per binary give the Jacobian of the states, ``(B, n_ds, 11, 8)``.

:func:`states_function` is an ``mx.custom_function`` of the parameters whose
VJP (``mx.grad``, ``mx.vjp``) contracts the Jacobian with the cotangents,
so that differentiating the waveform runs the Jacobian kernel rather than
the operations of every step. MLX 0.32.4 has no forward mode through
custom kernels: ``mx.jvp`` fails, even with a custom JVP.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .._dynamics import _A, _B, _FIT_FIRST, _FIT_PARTNER, _FIT_SECOND, _IA, _IB, _RA, _RB
from .._fits import Q_FIT_OFFSET, Q_FIT_SLOPE
from ._dynamics import reference_omegas, reference_time

__all__ = ["N_PARAMETERS", "states", "jacobian", "states_and_jacobian", "states_function", "integrate"]

#: The parameters of the integration: ``q``, ``chi_a`` (3), ``chi_b`` (3)
#: and ``t_ref``.
N_PARAMETERS = 8
#: Threads of the SIMD group that integrates one binary.
SIMD = 32


def _table(name: str, values, kind: str) -> str:
    """A Metal ``constant`` array."""
    if kind == "float":
        items = ", ".join(f"{float(v)!r}f" for v in values)
    elif kind == "bool":
        items = ", ".join("true" if v else "false" for v in values)
    else:
        items = ", ".join(str(int(v)) for v in values)
    return f"constant {kind} {name}[{len(values)}] = {{{items}}};"


# dy_s/dt = A[s] y[IA[s]] r[RA[s]] + B[s] y[IB[s]] r[RB[s]] with the rotated
# fits r and y[11] = 1; the fit i is rotated with FIT_PARTNER[i]
_TABLES = "\n".join([
    _table("NR_A", _A, "float"), _table("NR_IA", _IA, "ushort"), _table("NR_RA", _RA, "ushort"),
    _table("NR_B", _B, "float"), _table("NR_IB", _IB, "ushort"), _table("NR_RB", _RB, "ushort"),
    _table("NR_FIT_PARTNER", _FIT_PARTNER, "ushort"), _table("NR_FIT_FIRST", _FIT_FIRST, "bool"),
    _table("NR_FIT_SECOND", _FIT_SECOND, "bool"),
])

# The numbers: floats, and dual numbers (a value and its derivative), with
# the same functions for both.
_NUMBERS = """
struct NrDual {
    float v;
    float d;
    NrDual() = default;
    NrDual(float value, float derivative = 0.0f) : v(value), d(derivative) {}
};

inline NrDual operator+(NrDual a, NrDual b) { return NrDual(a.v + b.v, a.d + b.d); }
inline NrDual operator-(NrDual a, NrDual b) { return NrDual(a.v - b.v, a.d - b.d); }
inline NrDual operator-(NrDual a) { return NrDual(-a.v, -a.d); }
inline NrDual operator*(NrDual a, NrDual b) { return NrDual(a.v * b.v, a.d * b.v + a.v * b.d); }
inline NrDual operator/(NrDual a, NrDual b) {
    float r = a.v / b.v;
    return NrDual(r, (a.d - r * b.d) / b.v);
}
inline NrDual operator+(NrDual a, float c) { return NrDual(a.v + c, a.d); }
inline NrDual operator+(float c, NrDual a) { return NrDual(c + a.v, a.d); }
inline NrDual operator-(NrDual a, float c) { return NrDual(a.v - c, a.d); }
inline NrDual operator-(float c, NrDual a) { return NrDual(c - a.v, -a.d); }
inline NrDual operator*(float c, NrDual a) { return NrDual(c * a.v, c * a.d); }
inline NrDual operator*(NrDual a, float c) { return NrDual(a.v * c, a.d * c); }
inline NrDual operator/(NrDual a, float c) { return NrDual(a.v / c, a.d / c); }
inline NrDual operator/(float c, NrDual b) {
    float r = c / b.v;
    return NrDual(r, -r * b.d / b.v);
}

inline float nr_v(float a) { return a; }
inline float nr_v(NrDual a) { return a.v; }
inline void nr_sincos(float a, thread float& s, thread float& c) { s = metal::precise::sincos(a, c); }
inline void nr_sincos(NrDual a, thread NrDual& s, thread NrDual& c) {
    float cv;
    float sv = metal::precise::sincos(a.v, cv);
    s = NrDual(sv, cv * a.d);
    c = NrDual(cv, -sv * a.d);
}
inline float nr_sqrt(float a) { return metal::precise::sqrt(a); }
inline NrDual nr_sqrt(NrDual a) {
    float s = metal::precise::sqrt(a.v);
    return NrDual(s, a.d / (2.0f * s));
}
inline float nr_log(float a) { return metal::precise::log(a); }
inline NrDual nr_log(NrDual a) { return NrDual(metal::precise::log(a.v), a.d / a.v); }
inline float nr_shuffle(float a, ushort lane) { return metal::simd_shuffle(a, lane); }
inline NrDual nr_shuffle(NrDual a, ushort lane) {
    return NrDual(metal::simd_shuffle(a.v, lane), metal::simd_shuffle(a.d, lane));
}
inline float nr_simd_sum(float a) { return metal::simd_sum(a); }
inline NrDual nr_simd_sum(NrDual a) {
    float2 total = metal::simd_sum(float2(a.v, a.d));
    return NrDual(total.x, total.y);
}

// the states, or their derivatives with respect to parameter k (of
// `lanes`), at `node`: component `lane` (lanes 0 to 10)
inline void nr_write(device float* out, uint b, int n_ds, int node, float y, uint lane, uint k, uint lanes) {
    if (lane < 11) out[((size_t)b * n_ds + node) * 11 + lane] = y;
}
inline void nr_write(device float* out, uint b, int n_ds, int node, NrDual y, uint lane, uint k, uint lanes) {
    if (lane < 11) out[(((size_t)b * n_ds + node) * 11 + lane) * lanes + k] = y.d;
}

// the number of t[i] < value (strict) or <= value, for i in [first, n)
inline int nr_count(const device float* t, int first, int n, float value, bool strict) {
    int lo = first;
    int hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        bool below = strict ? t[mid] < value : t[mid] <= value;
        if (below) lo = mid + 1;
        else hi = mid;
    }
    return lo - first;
}
"""

# The integration on the numbers NrNum.
_INTEGRATION = """
// |v| of three components, with a zero derivative at zero (safe_norm)
inline NrNum nr_norm3(NrNum x, NrNum y, NrNum z) {
    NrNum square = x * x + y * y + z * z;
    return nr_v(square) > 0.0f ? nr_sqrt(square) : NrNum(0.0f);
}

// PrecessingNRSur_normalize_y for the component s of the state y held by
// this lane: the quaternion (s < 4) to unit length, the spins to norm_a
// (5 <= s < 8) and norm_b (s >= 8), zero where a spin vanishes; the phase
// (s = 4) is left as it is. The sums of squares are gathered by shuffles.
inline NrNum nr_normalize(NrNum y, uint s, NrNum norm_a, NrNum norm_b) {
    NrNum square = y * y;
    NrNum q2 = ((nr_shuffle(square, 0) + nr_shuffle(square, 1)) + nr_shuffle(square, 2)) + nr_shuffle(square, 3);
    NrNum a2 = (nr_shuffle(square, 5) + nr_shuffle(square, 6)) + nr_shuffle(square, 7);
    NrNum b2 = (nr_shuffle(square, 8) + nr_shuffle(square, 9)) + nr_shuffle(square, 10);
    NrNum total = s < 4 ? q2 : (s < 8 ? a2 : b2);
    NrNum target = s < 4 ? NrNum(1.0f) : (s < 8 ? norm_a : norm_b);
    NrNum scale = nr_v(total) > 0.0f ? target / nr_sqrt(total) : NrNum(0.0f);
    return s == 4 ? y : y * scale;
}

// the quantities of the fit parameters that depend on the mass ratio only,
// computed once per binary
struct NrMass {
    NrNum q;
    NrNum inv_one_q;
    NrNum c38;
    NrNum inv_hat;
    NrNum qf;
    NrNum qf2;
    NrNum qf3;
};

inline NrMass nr_mass(NrNum q) {
    NrMass mass;
    mass.q = q;
    NrNum one_q = 1.0f + q;
    NrNum eta = q / one_q / one_q;
    mass.inv_one_q = 1.0f / one_q;
    mass.c38 = 38.0f * eta / 113.0f;
    mass.inv_hat = 1.0f / (1.0f - 76.0f * eta / 113.0f);
    mass.qf = @Q_FIT_OFFSET@ + @Q_FIT_SLOPE@ * nr_log(q);
    mass.qf2 = mass.qf * mass.qf;
    mass.qf3 = mass.qf2 * mass.qf;
    return mass;
}

// x^k for k = 0, 1, 2 (and 3 with x3)
inline NrNum nr_pick(uint k, NrNum x, NrNum x2) {
    NrNum r = k == 1u ? x : NrNum(1.0f);
    return k == 2u ? x2 : r;
}

inline NrNum nr_pick(uint k, NrNum x, NrNum x2, NrNum x3) {
    return k == 3u ? x3 : nr_pick(k, x, x2);
}

// dy_s/dt, for the component s held by this lane, at the state whose
// component s this lane holds (y_own), with the fits of the dynamics node
// `node` (PrecessingNRSur_get_time_deriv_from_index). The state is gathered
// by shuffles; the monomials of the fit parameters (each the product of the
// powers of its exponents `orders`, two bits per parameter, in the order of
// the parameters) are split among the lanes and the nine fits summed over
// them, lane i < 9 keeping fit i, rotated with its partner's; the lane then
// assembles its component (PrecessingNRSur_assemble_dydt).
inline NrNum nr_derivative(
    const device float* dynamics, const device uint* orders, int n_mono, int node,
    NrNum y_own, uint s, thread const NrMass& mass, uint lane) {
    NrNum y[11];
    for (ushort c = 0; c < 11; ++c) y[c] = nr_shuffle(y_own, c);
    NrNum sp, cp;
    nr_sincos(y[4], sp, cp);
    // the spins in the coorbital frame
    NrNum ax = y[5] * cp + y[6] * sp;
    NrNum ay = y[6] * cp - y[5] * sp;
    NrNum az = y[7];
    NrNum bx = y[8] * cp + y[9] * sp;
    NrNum by = y[9] * cp - y[8] * sp;
    NrNum bz = y[10];
    // the fit parameters (NRSur7dq4_effective_spins)
    NrNum chi_wt_avg = (mass.q * az + bz) * mass.inv_one_q;
    NrNum chi_hat = (chi_wt_avg - mass.c38 * (az + bz)) * mass.inv_hat;
    NrNum chi_asym = (az - bz) / 2.0f;
    NrNum spins[6] = {ax, ay, chi_hat, bx, by, chi_asym};
    NrNum squares[6];
    for (int j = 0; j < 6; ++j) squares[j] = spins[j] * spins[j];
    NrNum fits[9];
    for (int i = 0; i < 9; ++i) fits[i] = NrNum(0.0f);
    const device float* coefs = dynamics + (size_t)node * 9 * n_mono;
    for (int m = (int)lane; m < n_mono; m += @SIMD@) {
        uint o = orders[m];
        NrNum mono = nr_pick(o & 3u, mass.qf, mass.qf2, mass.qf3);
        for (int j = 0; j < 6; ++j) mono = mono * nr_pick((o >> (2 * j + 2)) & 3u, spins[j], squares[j]);
        for (int i = 0; i < 9; ++i) fits[i] = fits[i] + coefs[i * n_mono + m] * mono;
    }
    NrNum fit = NrNum(0.0f);
    for (uint i = 0; i < 9; ++i) {
        NrNum total = nr_simd_sum(fits[i]);
        fit = lane == i ? total : fit;
    }
    // the coorbital components rotated to the coprecessing frame
    uint f = metal::min(lane, 8u);
    NrNum partner = nr_shuffle(fit, NR_FIT_PARTNER[f]);
    NrNum rotated = NR_FIT_FIRST[f] ? fit * cp - partner * sp : (NR_FIT_SECOND[f] ? fit * cp + partner * sp : fit);
    // A y[IA] r[RA] + B y[IB] r[RB], with y[11] = 1
    NrNum ya = NR_IA[s] == 11 ? NrNum(1.0f) : nr_shuffle(y_own, NR_IA[s]);
    NrNum yb = NR_IB[s] == 11 ? NrNum(1.0f) : nr_shuffle(y_own, NR_IB[s]);
    return (NR_A[s] * ya) * nr_shuffle(rotated, NR_RA[s]) + (NR_B[s] * yb) * nr_shuffle(rotated, NR_RB[s]);
}

// PrecessingNRSur_get_time_deriv: the cubic through the derivatives at the
// four nodes around t, for component s
inline NrNum nr_derivative_at_time(
    const device float* t_ds, int n_ds, const device float* dynamics, const device uint* orders, int n_mono,
    NrNum t, NrNum y, uint s, thread const NrMass& mass, uint lane) {
    int i0 = metal::clamp(nr_count(t_ds, 1, n_ds, nr_v(t), true) - 1, 0, n_ds - 4);
    NrNum d0 = nr_derivative(dynamics, orders, n_mono, i0, y, s, mass, lane);
    NrNum d1 = nr_derivative(dynamics, orders, n_mono, i0 + 1, y, s, mass, lane);
    NrNum d2 = nr_derivative(dynamics, orders, n_mono, i0 + 2, y, s, mass, lane);
    NrNum d3 = nr_derivative(dynamics, orders, n_mono, i0 + 3, y, s, mass, lane);
    float x0 = t_ds[i0], x1 = t_ds[i0 + 1], x2 = t_ds[i0 + 2], x3 = t_ds[i0 + 3];
    // Newton's divided differences (gsl_poly_dd_init, gsl_poly_dd_eval)
    NrNum dd1 = (d1 - d0) / (x1 - x0);
    NrNum dd2 = (d2 - d1) / (x2 - x1);
    NrNum dd3 = (d3 - d2) / (x3 - x2);
    dd3 = (dd3 - dd2) / (x3 - x1);
    dd2 = (dd2 - dd1) / (x2 - x0);
    dd3 = (dd3 - dd2) / (x3 - x0);
    NrNum value = dd3;
    value = dd2 + (t - x2) * value;
    value = dd1 + (t - x1) * value;
    return d0 + (t - x0) * value;
}
"""

_SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint lane = gid % @SIMD@;
    uint group = gid / @SIMD@;
    uint b = group / LANES;
    uint k = group % LANES;
    // the component of the state held by this lane (lanes 11 to 31 repeat
    // the last)
    uint s = metal::min(lane, 10u);
    const int n = N_DS;
    const int n_mono = N_MONO;

    // the parameters (with the derivative with respect to parameter k)
    const device float* p = params + @N_PARAMETERS@ * b;
    NrNum q = NR_SEED(p[0], k == 0);
    NrNum chi[6];
    for (int i = 0; i < 6; ++i) chi[i] = NR_SEED(p[1 + i], k == (uint)(1 + i));
    NrNum t_ref = NR_SEED(p[7], k == 7);
    NrNum norm_a = nr_norm3(chi[0], chi[1], chi[2]);
    NrNum norm_b = nr_norm3(chi[3], chi[4], chi[5]);
    NrMass mass = nr_mass(q);
    // the reference state: identity quaternion, zero orbital phase, the spins
    NrNum y_ref = s == 0 ? NrNum(1.0f) : (s < 5 ? NrNum(0.0f) : chi[metal::max((int)s - 5, 0)]);

    // PrecessingNRSur_initialize_at_dynamics_node: an Euler step from t_ref
    // to the nearest node i0
    int imin = metal::clamp(nr_count(t_ds, 0, n, nr_v(t_ref), false) - 1, 0, n - 2);
    float t_min = t_ds[imin];
    float t_max = t_ds[imin + 1];
    bool lower = metal::fabs(nr_v(t_ref) - t_min) < metal::fabs(t_max - nr_v(t_ref));
    int i0 = lower ? imin : imin + 1;
    NrNum dt = lower ? t_min - t_ref : t_max - t_ref;
    NrNum dydt = nr_derivative_at_time(t_ds, n, dynamics, orders, n_mono, t_ref, y_ref, s, mass, lane);
    NrNum y = nr_normalize(y_ref + dt * dydt, s, norm_a, norm_b);

    // three RK4 steps from i0 (PrecessingNRSur_initialize_RK4 and
    // PrecessingNRSur_initialize_RK4_with_half_nodes)
    bool half_nodes = i0 == 0;
    int sign = i0 > 2 ? -1 : 1;
    NrNum states4[4];
    NrNum derivs4[4];
    states4[0] = y;
    for (int step = 0; step < 3; ++step) {
        int node = i0 + sign * step;
        int following = node + sign;
        float hs = t_ds[following] - t_ds[node];
        NrNum t_half = NrNum(0.5f * (t_ds[node] + t_ds[following]));
        NrNum k1 = nr_derivative(dynamics, orders, n_mono, node, y, s, mass, lane);
        NrNum mid = y + (0.5f * hs) * k1;
        NrNum k2 = half_nodes ? nr_derivative(dynamics, orders, n_mono, n + step, mid, s, mass, lane)
                              : nr_derivative_at_time(t_ds, n, dynamics, orders, n_mono, t_half, mid, s, mass, lane);
        mid = y + (0.5f * hs) * k2;
        NrNum k3 = half_nodes ? nr_derivative(dynamics, orders, n_mono, n + step, mid, s, mass, lane)
                              : nr_derivative_at_time(t_ds, n, dynamics, orders, n_mono, t_half, mid, s, mass, lane);
        mid = y + hs * k3;
        NrNum k4 = nr_derivative(dynamics, orders, n_mono, following, mid, s, mass, lane);
        y = nr_normalize(y + (hs / 6.0f) * (((k1 + 2.0f * k2) + 2.0f * k3) + k4), s, norm_a, norm_b);
        states4[step + 1] = y;
        derivs4[step] = k1;
    }
    derivs4[3] = nr_derivative(dynamics, orders, n_mono, i0 + 3 * sign, y, s, mass, lane);
    int i_start = sign > 0 ? i0 : i0 - 3;
    if (sign < 0) {
        // in increasing order of the nodes
        NrNum a = states4[0]; states4[0] = states4[3]; states4[3] = a;
        a = states4[1]; states4[1] = states4[2]; states4[2] = a;
        a = derivs4[0]; derivs4[0] = derivs4[3]; derivs4[3] = a;
        a = derivs4[1]; derivs4[1] = derivs4[2]; derivs4[2] = a;
    }
    for (int j = 0; j < 4; ++j) nr_write(out, b, n, i_start + j, states4[j], lane, k, LANES);

    // PrecessingNRSur_integrate_AB4: backwards from i_start to node 0, then
    // forwards from i_start + 3 to the last node, with the derivatives at
    // the four nodes before the next step of each direction
    NrNum hf0 = derivs4[0], hf1 = derivs4[1], hf2 = derivs4[2], hf3 = derivs4[3];
    NrNum hb0 = hf0, hb1 = hf1, hb2 = hf2, hb3 = hf3;
    y = states4[0];
    for (int j = 0; j < n - 4; ++j) {
        bool back = j < i_start;
        int node = back ? i_start - 1 - j : j + 4;
        if (j == i_start) y = states4[3];
        const device float* w = (back ? ab4_backward : ab4_forward) + 4 * node;
        NrNum h0 = back ? hb0 : hf0;
        NrNum h1 = back ? hb1 : hf1;
        NrNum h2 = back ? hb2 : hf2;
        NrNum h3 = back ? hb3 : hf3;
        y = nr_normalize(y + (((w[0] * h0 + w[1] * h1) + w[2] * h2) + w[3] * h3), s, norm_a, norm_b);
        NrNum derivative = nr_derivative(dynamics, orders, n_mono, node, y, s, mass, lane);
        if (back) {
            hb3 = hb2;
            hb2 = hb1;
            hb1 = hb0;
            hb0 = derivative;
        } else {
            hf0 = hf1;
            hf1 = hf2;
            hf2 = hf3;
            hf3 = derivative;
        }
        nr_write(out, b, n, node, y, lane, k, LANES);
    }
"""


def _kernel(number: str, seed: str):
    """The integration on the numbers ``number`` (``float`` or ``NrDual``),
    the parameters made numbers by the macro ``NR_SEED(x, is_k)``."""
    substitutions = {
        "@SIMD@": str(SIMD),
        "@N_PARAMETERS@": str(N_PARAMETERS),
        "@Q_FIT_OFFSET@": f"{Q_FIT_OFFSET!r}f",
        "@Q_FIT_SLOPE@": f"{Q_FIT_SLOPE!r}f",
        "NrNum": number,
    }
    header = _NUMBERS + _TABLES + f"\n#define NR_SEED(x, is_k) {seed}\n" + _INTEGRATION
    source = _SOURCE
    for key, value in substitutions.items():
        header, source = header.replace(key, value), source.replace(key, value)
    return mx.fast.metal_kernel(
        name=f"nrsur7dq4_dynamics_{number.lower()}",
        input_names=["params", "t_ds", "dynamics", "orders", "ab4_forward", "ab4_backward"],
        output_names=["out"],
        source=source,
        header=header,
    )


_STATES_KERNEL = _kernel("float", "(x)")
_JACOBIAN_KERNEL = _kernel("NrDual", "NrDual((x), (is_k) ? 1.0f : 0.0f)")

_ORDERS = {}


def _orders(data):
    """The exponents of the seven fit parameters in each monomial of the
    dynamics fits, two bits each in the order of the parameters
    (``uint32``, ``(n_monomials,)``), from their factors (indices ``4 j +
    k`` into the table of powers, padded with 0)."""
    key = id(data)
    if key not in _ORDERS:
        factors = np.array(data.monomials).astype(np.int64)
        packed = np.zeros(factors.shape[0], dtype=np.uint32)
        for column in factors.T:
            nonzero = column > 0
            packed[nonzero] |= ((column[nonzero] % 4) << (2 * (column[nonzero] // 4))).astype(np.uint32)
        _ORDERS[key] = (data, mx.array(packed))
    return _ORDERS[key][1]


def _launch(kernel, data, params, lanes: int, shape):
    size = int(params.shape[0])
    threads = size * lanes * SIMD
    (out,) = kernel(
        inputs=[params.astype(mx.float32), data.t_ds, data.dynamics, _orders(data), data.ab4_forward, data.ab4_backward],
        template=[("N_DS", data.n_ds), ("N_MONO", int(data.monomials.shape[0])), ("LANES", lanes)],
        grid=(threads, 1, 1),
        threadgroup=(min(threads, 256), 1, 1),
        output_shapes=[(size, data.n_ds, *shape)],
        output_dtypes=[mx.float32],
    )
    return out


def states(data, params):
    """The states at the dynamics nodes (``(B, n_ds, 11)``) from the
    parameters ``(q, chi_a, chi_b, t_ref)`` of each binary (``params``,
    ``(B, 8)``)."""
    return _launch(_STATES_KERNEL, data, params, 1, (11,))


def jacobian(data, params):
    """The derivatives of :func:`states` with respect to the parameters
    (``(B, n_ds, 11, 8)``)."""
    return _launch(_JACOBIAN_KERNEL, data, params, N_PARAMETERS, (11, N_PARAMETERS))


def states_and_jacobian(data, params):
    """:func:`states` and :func:`jacobian`."""
    return states(data, params), jacobian(data, params)


def _single(value):
    return value[0] if isinstance(value, (list, tuple)) else value


_FUNCTIONS = {}


def states_function(data):
    """The function ``params -> states`` (:func:`states`) of ``data``,
    differentiable: an ``mx.custom_function`` whose VJP contracts the
    Jacobian."""
    key = id(data)
    if key not in _FUNCTIONS:

        @mx.custom_function
        def function(params):
            return states(data, params)

        @function.vjp
        def function_vjp(primals, cotangent, output):
            return (mx.sum(jacobian(data, _single(primals)) * _single(cotangent)[..., None], axis=(1, 2)),)

        _FUNCTIONS[key] = (data, function)
    return _FUNCTIONS[key][1]


def integrate(data, q, chi_a, chi_b, omega_ref, omegas=None):
    """:func:`._dynamics.integrate` with the integration in the kernel: the
    states at every dynamics node (``(B, n_ds, 11)``) and whether each
    evaluation is valid."""
    if omegas is None:
        omegas = reference_omegas(data, q, chi_a, chi_b)
    t_ref, valid = reference_time(data, omegas, omega_ref)
    params = mx.concatenate([q[:, None], chi_a, chi_b, t_ref[:, None]], axis=1)
    return states_function(data)(params), valid
