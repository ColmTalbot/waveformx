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

Every binary is integrated by one SIMD group of 32 threads, which share
the matrix-vector product of the fits of each step (the monomials are
split among them and the nine fits summed with ``simd_sum``) and repeat the
rest of the step in lockstep. The arithmetic is on dual numbers: a SIMD
group carries the derivative of the whole integration with respect to one
of its eight parameters ``(q, chi_a, chi_b, t_ref)`` (forward mode), so
that eight groups per binary give the Jacobian of the states, ``(B, n_ds,
11, 8)``. The value alone takes one group per binary.

:func:`states_function` is an ``mx.custom_function`` of the parameters whose
VJP (``mx.grad``, ``mx.vjp``) contracts the Jacobian with the cotangents,
so that differentiating the waveform runs the kernel once more with the
eight groups, rather than the operations of every step. MLX 0.32.4 has no
forward mode through custom kernels: ``mx.jvp`` fails, even with a custom
JVP.
"""

from __future__ import annotations

import mlx.core as mx

from .._fits import Q_FIT_OFFSET, Q_FIT_SLOPE
from ._dynamics import reference_omegas, reference_time

__all__ = ["N_PARAMETERS", "states_and_jacobian", "states_function", "integrate"]

#: The parameters of the integration: ``q``, ``chi_a`` (3), ``chi_b`` (3)
#: and ``t_ref``.
N_PARAMETERS = 8
#: Threads of the SIMD group that integrates one binary.
SIMD = 32

_HEADER = f"""
struct NrDual {{
    float v;
    float d;
}};

inline NrDual nr_const(float v) {{ return NrDual{{v, 0.0f}}; }}
inline NrDual operator+(NrDual a, NrDual b) {{ return NrDual{{a.v + b.v, a.d + b.d}}; }}
inline NrDual operator-(NrDual a, NrDual b) {{ return NrDual{{a.v - b.v, a.d - b.d}}; }}
inline NrDual operator-(NrDual a) {{ return NrDual{{-a.v, -a.d}}; }}
inline NrDual operator*(NrDual a, NrDual b) {{ return NrDual{{a.v * b.v, a.d * b.v + a.v * b.d}}; }}
inline NrDual operator/(NrDual a, NrDual b) {{
    float r = a.v / b.v;
    return NrDual{{r, (a.d - r * b.d) / b.v}};
}}
inline NrDual operator+(NrDual a, float c) {{ return NrDual{{a.v + c, a.d}}; }}
inline NrDual operator+(float c, NrDual a) {{ return NrDual{{c + a.v, a.d}}; }}
inline NrDual operator-(NrDual a, float c) {{ return NrDual{{a.v - c, a.d}}; }}
inline NrDual operator-(float c, NrDual a) {{ return NrDual{{c - a.v, -a.d}}; }}
inline NrDual operator*(float c, NrDual a) {{ return NrDual{{c * a.v, c * a.d}}; }}
inline NrDual operator*(NrDual a, float c) {{ return NrDual{{a.v * c, a.d * c}}; }}
inline NrDual operator/(NrDual a, float c) {{ return NrDual{{a.v / c, a.d / c}}; }}
inline NrDual operator/(float c, NrDual b) {{
    float r = c / b.v;
    return NrDual{{r, -r * b.d / b.v}};
}}
inline NrDual nr_sin(NrDual a) {{ return NrDual{{metal::precise::sin(a.v), metal::precise::cos(a.v) * a.d}}; }}
inline NrDual nr_cos(NrDual a) {{ return NrDual{{metal::precise::cos(a.v), -metal::precise::sin(a.v) * a.d}}; }}
inline NrDual nr_sqrt(NrDual a) {{
    float s = metal::precise::sqrt(a.v);
    return NrDual{{s, a.d / (2.0f * s)}};
}}
inline NrDual nr_log(NrDual a) {{ return NrDual{{metal::precise::log(a.v), a.d / a.v}}; }}

// |v| of three components, with a zero derivative at zero (safe_norm)
inline NrDual nr_norm3(NrDual x, NrDual y, NrDual z) {{
    NrDual square = x * x + y * y + z * z;
    return square.v > 0.0f ? nr_sqrt(square) : nr_const(0.0f);
}}

// PrecessingNRSur_normalize_y: the quaternion to unit length, the spins to
// norm_a and norm_b (zero where a spin vanishes)
inline void nr_normalize(thread NrDual* y, NrDual norm_a, NrDual norm_b) {{
    NrDual square = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
    NrDual scale = square.v > 0.0f ? 1.0f / nr_sqrt(square) : nr_const(0.0f);
    for (int i = 0; i < 4; ++i) y[i] = y[i] * scale;
    square = y[5] * y[5] + y[6] * y[6] + y[7] * y[7];
    scale = square.v > 0.0f ? norm_a / nr_sqrt(square) : nr_const(0.0f);
    for (int i = 5; i < 8; ++i) y[i] = y[i] * scale;
    square = y[8] * y[8] + y[9] * y[9] + y[10] * y[10];
    scale = square.v > 0.0f ? norm_b / nr_sqrt(square) : nr_const(0.0f);
    for (int i = 8; i < 11; ++i) y[i] = y[i] * scale;
}}

// dy/dt at the state y with the fits of the dynamics node `node`
// (PrecessingNRSur_get_time_deriv_from_index): the monomials of the fit
// parameters, split among the threads of the SIMD group, the nine fits
// summed over the group, and PrecessingNRSur_assemble_dydt
inline void nr_derivative(
    const device float* dynamics, const device int* factors, int n_mono, int node,
    thread const NrDual* y, NrDual q, thread const NrDual* q_powers, uint lane, thread NrDual* out) {{
    NrDual sp = nr_sin(y[4]);
    NrDual cp = nr_cos(y[4]);
    // the spins in the coorbital frame
    NrDual ax = y[5] * cp + y[6] * sp;
    NrDual ay = y[6] * cp - y[5] * sp;
    NrDual az = y[7];
    NrDual bx = y[8] * cp + y[9] * sp;
    NrDual by = y[9] * cp - y[8] * sp;
    NrDual bz = y[10];
    // the fit parameters (NRSur7dq4_effective_spins)
    NrDual one_q = 1.0f + q;
    NrDual eta = q / one_q / one_q;
    NrDual chi_wt_avg = (q * az + bz) / one_q;
    NrDual chi_hat = (chi_wt_avg - (38.0f * eta / 113.0f) * (az + bz)) / (1.0f - 76.0f * eta / 113.0f);
    NrDual chi_asym = (az - bz) / 2.0f;
    NrDual spins[6] = {{ax, ay, chi_hat, bx, by, chi_asym}};
    NrDual table[28];
    for (int k = 0; k < 4; ++k) table[k] = q_powers[k];
    for (int j = 0; j < 6; ++j) {{
        table[4 + 4 * j] = nr_const(1.0f);
        table[5 + 4 * j] = spins[j];
        table[6 + 4 * j] = spins[j] * spins[j];
        table[7 + 4 * j] = nr_const(0.0f);
    }}
    NrDual fits[9];
    for (int i = 0; i < 9; ++i) fits[i] = nr_const(0.0f);
    const device float* coefs = dynamics + (size_t)node * 9 * n_mono;
    for (int m = (int)lane; m < n_mono; m += {SIMD}) {{
        const device int* f = factors + 5 * m;
        NrDual mono = table[f[0]] * table[f[1]] * table[f[2]] * table[f[3]] * table[f[4]];
        for (int i = 0; i < 9; ++i) fits[i] = fits[i] + coefs[i * n_mono + m] * mono;
    }}
    for (int i = 0; i < 9; ++i) {{
        fits[i].v = metal::simd_sum(fits[i].v);
        fits[i].d = metal::simd_sum(fits[i].d);
    }}
    // the coorbital components rotated to the coprecessing frame
    NrDual r1 = fits[1] * cp - fits[2] * sp;
    NrDual r2 = fits[2] * cp + fits[1] * sp;
    NrDual r3 = fits[3] * cp - fits[4] * sp;
    NrDual r4 = fits[4] * cp + fits[3] * sp;
    NrDual r6 = fits[6] * cp - fits[7] * sp;
    NrDual r7 = fits[7] * cp + fits[6] * sp;
    // the quaternion derivative q Omega / 2
    out[0] = (-0.5f * y[1]) * r1 + (-0.5f * y[2]) * r2;
    out[1] = (-0.5f * y[3]) * r2 + (0.5f * y[0]) * r1;
    out[2] = (0.5f * y[3]) * r1 + (0.5f * y[0]) * r2;
    out[3] = (0.5f * y[1]) * r2 + (-0.5f * y[2]) * r1;
    out[4] = fits[0];
    out[5] = r3;
    out[6] = r4;
    out[7] = fits[5];
    out[8] = r6;
    out[9] = r7;
    out[10] = fits[8];
}}

// PrecessingNRSur_get_time_deriv: the cubic through the derivatives at the
// four nodes around t
inline void nr_derivative_at_time(
    const device float* t_ds, int n_ds, const device float* dynamics, const device int* factors, int n_mono,
    NrDual t, thread const NrDual* y, NrDual q, thread const NrDual* q_powers, uint lane, thread NrDual* out) {{
    int i1 = 0;
    for (int i = 1; i < n_ds; ++i) i1 += t_ds[i] < t.v ? 1 : 0;
    int i0 = metal::clamp(i1 - 1, 0, n_ds - 4);
    NrDual d[44];
    float x[4];
    for (int j = 0; j < 4; ++j) {{
        nr_derivative(dynamics, factors, n_mono, i0 + j, y, q, q_powers, lane, d + 11 * j);
        x[j] = t_ds[i0 + j];
    }}
    // Newton's divided differences (gsl_poly_dd_init, gsl_poly_dd_eval)
    for (int s = 0; s < 11; ++s) {{
        NrDual dd0 = d[s];
        NrDual dd1 = (d[11 + s] - d[s]) / (x[1] - x[0]);
        NrDual dd2 = (d[22 + s] - d[11 + s]) / (x[2] - x[1]);
        NrDual dd3 = (d[33 + s] - d[22 + s]) / (x[3] - x[2]);
        dd3 = (dd3 - dd2) / (x[3] - x[1]);
        dd2 = (dd2 - dd1) / (x[2] - x[0]);
        dd3 = (dd3 - dd2) / (x[3] - x[0]);
        NrDual value = dd3;
        value = dd2 + (t - x[2]) * value;
        value = dd1 + (t - x[1]) * value;
        out[s] = dd0 + (t - x[0]) * value;
    }}
}}

inline void nr_write(
    device float* states, device float* jacobian, uint b, int n_ds, int node, thread const NrDual* y,
    uint lane, uint k, uint lanes) {{
    if (lane != 0) return;
    size_t base = ((size_t)b * n_ds + node) * 11;
    for (int s = 0; s < 11; ++s) {{
        if (k == 0) states[base + s] = y[s].v;
        jacobian[(base + s) * lanes + k] = y[s].d;
    }}
}}
"""

_SOURCE = f"""
    uint gid = thread_position_in_grid.x;
    uint lane = gid % {SIMD};
    uint group = gid / {SIMD};
    uint b = group / LANES;
    uint k = group % LANES;
    const int n = N_DS;
    const int n_mono = N_MONO;

    // the parameters, with the derivative with respect to parameter k
    const device float* p = params + {N_PARAMETERS} * b;
    NrDual q = NrDual{{p[0], k == 0 ? 1.0f : 0.0f}};
    NrDual y_ref[11];
    y_ref[0] = nr_const(1.0f);
    for (int i = 1; i < 5; ++i) y_ref[i] = nr_const(0.0f);
    for (int i = 0; i < 6; ++i) y_ref[5 + i] = NrDual{{p[1 + i], k == (uint)(1 + i) ? 1.0f : 0.0f}};
    NrDual t_ref = NrDual{{p[7], k == 7 ? 1.0f : 0.0f}};
    NrDual norm_a = nr_norm3(y_ref[5], y_ref[6], y_ref[7]);
    NrDual norm_b = nr_norm3(y_ref[8], y_ref[9], y_ref[10]);
    NrDual qf = {Q_FIT_OFFSET!r}f + {Q_FIT_SLOPE!r}f * nr_log(q);
    NrDual q_powers[4] = {{nr_const(1.0f), qf, qf * qf, qf * qf * qf}};

    // PrecessingNRSur_initialize_at_dynamics_node: an Euler step from t_ref
    // to the nearest node i0
    int imin = -1;
    for (int i = 0; i < n; ++i) imin += t_ds[i] <= t_ref.v ? 1 : 0;
    imin = metal::clamp(imin, 0, n - 2);
    float t_min = t_ds[imin];
    float t_max = t_ds[imin + 1];
    bool lower = metal::fabs(t_ref.v - t_min) < metal::fabs(t_max - t_ref.v);
    int i0 = lower ? imin : imin + 1;
    NrDual dt = lower ? t_min - t_ref : t_max - t_ref;
    NrDual dydt[11];
    nr_derivative_at_time(t_ds, n, dynamics, factors, n_mono, t_ref, y_ref, q, q_powers, lane, dydt);
    NrDual y[11];
    for (int s = 0; s < 11; ++s) y[s] = y_ref[s] + dt * dydt[s];
    nr_normalize(y, norm_a, norm_b);

    // three RK4 steps from i0 (PrecessingNRSur_initialize_RK4 and
    // PrecessingNRSur_initialize_RK4_with_half_nodes)
    bool half_nodes = i0 == 0;
    int sign = i0 > 2 ? -1 : 1;
    NrDual states4[44];
    NrDual derivs4[44];
    for (int s = 0; s < 11; ++s) states4[s] = y[s];
    NrDual k1[11], k2[11], k3[11], k4[11], mid[11];
    for (int step = 0; step < 3; ++step) {{
        int node = i0 + sign * step;
        int following = node + sign;
        float hs = t_ds[following] - t_ds[node];
        NrDual t_half = nr_const(0.5f * (t_ds[node] + t_ds[following]));
        nr_derivative(dynamics, factors, n_mono, node, y, q, q_powers, lane, k1);
        for (int s = 0; s < 11; ++s) mid[s] = y[s] + (0.5f * hs) * k1[s];
        if (half_nodes) nr_derivative(dynamics, factors, n_mono, n + step, mid, q, q_powers, lane, k2);
        else nr_derivative_at_time(t_ds, n, dynamics, factors, n_mono, t_half, mid, q, q_powers, lane, k2);
        for (int s = 0; s < 11; ++s) mid[s] = y[s] + (0.5f * hs) * k2[s];
        if (half_nodes) nr_derivative(dynamics, factors, n_mono, n + step, mid, q, q_powers, lane, k3);
        else nr_derivative_at_time(t_ds, n, dynamics, factors, n_mono, t_half, mid, q, q_powers, lane, k3);
        for (int s = 0; s < 11; ++s) mid[s] = y[s] + hs * k3[s];
        nr_derivative(dynamics, factors, n_mono, following, mid, q, q_powers, lane, k4);
        for (int s = 0; s < 11; ++s) y[s] = y[s] + (hs / 6.0f) * (((k1[s] + 2.0f * k2[s]) + 2.0f * k3[s]) + k4[s]);
        nr_normalize(y, norm_a, norm_b);
        for (int s = 0; s < 11; ++s) {{
            states4[11 * (step + 1) + s] = y[s];
            derivs4[11 * step + s] = k1[s];
        }}
    }}
    nr_derivative(dynamics, factors, n_mono, i0 + 3 * sign, y, q, q_powers, lane, derivs4 + 33);
    int i_start = sign > 0 ? i0 : i0 - 3;
    if (sign < 0) {{
        // in increasing order of the nodes
        for (int j = 0; j < 2; ++j) {{
            for (int s = 0; s < 11; ++s) {{
                NrDual a = states4[11 * j + s];
                states4[11 * j + s] = states4[11 * (3 - j) + s];
                states4[11 * (3 - j) + s] = a;
                a = derivs4[11 * j + s];
                derivs4[11 * j + s] = derivs4[11 * (3 - j) + s];
                derivs4[11 * (3 - j) + s] = a;
            }}
        }}
    }}
    for (int j = 0; j < 4; ++j) nr_write(states, jacobian, b, n, i_start + j, states4 + 11 * j, lane, k, LANES);

    // PrecessingNRSur_integrate_AB4: backwards from i_start to node 0, then
    // forwards from i_start + 3 to the last node, with the derivatives at
    // the four nodes before the next step of each direction
    NrDual h_forward[44];
    NrDual h_backward[44];
    for (int i = 0; i < 44; ++i) {{
        h_forward[i] = derivs4[i];
        h_backward[i] = derivs4[i];
    }}
    for (int s = 0; s < 11; ++s) y[s] = states4[s];
    for (int j = 0; j < n - 4; ++j) {{
        bool back = j < i_start;
        int node = back ? i_start - 1 - j : j + 4;
        if (j == i_start) {{
            for (int s = 0; s < 11; ++s) y[s] = states4[33 + s];
        }}
        const device float* w = (back ? ab4_backward : ab4_forward) + 4 * node;
        thread NrDual* h = back ? h_backward : h_forward;
        for (int s = 0; s < 11; ++s) {{
            y[s] = y[s] + (((w[0] * h[s] + w[1] * h[11 + s]) + w[2] * h[22 + s]) + w[3] * h[33 + s]);
        }}
        nr_normalize(y, norm_a, norm_b);
        nr_derivative(dynamics, factors, n_mono, node, y, q, q_powers, lane, k1);
        if (back) {{
            for (int i = 43; i >= 11; --i) h[i] = h[i - 11];
            for (int s = 0; s < 11; ++s) h[s] = k1[s];
        }} else {{
            for (int i = 0; i < 33; ++i) h[i] = h[i + 11];
            for (int s = 0; s < 11; ++s) h[33 + s] = k1[s];
        }}
        nr_write(states, jacobian, b, n, node, y, lane, k, LANES);
    }}
"""

_KERNEL = mx.fast.metal_kernel(
    name="nrsur7dq4_dynamics",
    input_names=["params", "t_ds", "dynamics", "factors", "ab4_forward", "ab4_backward"],
    output_names=["states", "jacobian"],
    source=_SOURCE,
    header=_HEADER,
)


def states_and_jacobian(data, params, lanes: int = N_PARAMETERS):
    """The states at the dynamics nodes (``(B, n_ds, 11)``) and their
    derivatives with respect to the first ``lanes`` parameters (``(B, n_ds,
    11, lanes)``) from the parameters ``(q, chi_a, chi_b, t_ref)`` of each
    binary (``params``, ``(B, 8)``)."""
    size = int(params.shape[0])
    n = data.n_ds
    threads = size * lanes * SIMD
    return _KERNEL(
        inputs=[params.astype(mx.float32), data.t_ds, data.dynamics, data.monomials, data.ab4_forward, data.ab4_backward],
        template=[("N_DS", n), ("N_MONO", int(data.monomials.shape[0])), ("LANES", lanes)],
        grid=(threads, 1, 1),
        threadgroup=(min(threads, 256), 1, 1),
        output_shapes=[(size, n, 11), (size, n, 11, lanes)],
        output_dtypes=[mx.float32, mx.float32],
    )


def _single(value):
    return value[0] if isinstance(value, (list, tuple)) else value


_FUNCTIONS = {}


def states_function(data):
    """The function ``params -> states`` (see :func:`states_and_jacobian`) of
    ``data``, differentiable: an ``mx.custom_function`` whose VJP contracts
    the Jacobian."""
    key = id(data)
    if key not in _FUNCTIONS:

        @mx.custom_function
        def states(params):
            return states_and_jacobian(data, params, 1)[0]

        @states.vjp
        def states_vjp(primals, cotangent, output):
            jacobian = states_and_jacobian(data, _single(primals))[1]
            return (mx.sum(jacobian * _single(cotangent)[..., None], axis=(1, 2)),)

        _FUNCTIONS[key] = (data, states)
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
