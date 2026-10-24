# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The pieces of the lean SpinTaylor solution (:mod:`..spintaylor.lean`)
that the integration kernel (:mod:`._spintaylor`) calls, as JAX functions
of flat float arrays, and their Metal translations (:func:`device_source`):
each is traced in float32, lowered (:mod:`._program`) and written as a
Metal function on floats and, from its ``jax.jvp``, on dual numbers.

The functions take and return float arrays only (counts and flags as
floats); the coefficients of the equations (:func:`..spintaylor.dynamics.coefficients`)
travel as one vector ``c`` (their array leaves).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .._generated import dop853
from .._math import C_SI, MTSUN_SI, PI, g_si, lean as lean_mode, mass_unit, precision
from ..spintaylor import dynamics, trajectory
from ..spintaylor import lean as st_lean
from ..spintaylor.angles import _du_per_step
from . import _metal
from ._program import lower
from ._stages import combine, partition

__all__ = ["SETUP_FIELDS", "N_INPUTS", "N_ROW", "device_source", "c_size"]

#: The inputs of the integration: ``m1``, ``m2`` (solar masses), ``chi1``,
#: ``chi2`` (3 each), ``f_ref``, ``f_low``, ``f_cut``.
N_INPUTS = 11
#: The scalars :func:`setup` returns besides ``c``, ``y_ref`` and ``k_ref``.
SETUP_FIELDS = ("beyond", "omega_ref", "omega_low", "failed", "du_step", "n_back", "du_back", "dt_fine", "msec",
                "pi_gm")
#: The values of a step kept for the angles: ``y[1:5]``, the dense output of
#: those components (7 x 4), the time at the start of the step, the step
#: and ``M omega`` at its end.
N_ROW = 4 + 7 * 4 + 3
#: Of those, the ones with derivatives (all but ``M omega`` at the end).
N_ROW_TANGENT = N_ROW - 1


def _unpack_inputs(x):
    return x[0], x[1], [x[2], x[3], x[4]], [x[5], x[6], x[7]], x[8], x[9], x[10]


def _coefficients(x):
    m1, m2, _, _, f_ref, _, f_cut = _unpack_inputs(x)
    return dynamics.coefficients(m1, m2, f_ref, f_cut)


@functools.lru_cache(maxsize=None)
def _c_meta():
    holder = {}

    def f(x):
        with precision(np.float32), lean_mode(True):
            values, holder["meta"] = partition(_coefficients(x))
        return values

    jax.eval_shape(f, jax.ShapeDtypeStruct((N_INPUTS,), np.float32))
    return holder["meta"]


def c_size():
    """The length of the vector of coefficients."""
    return sum(_c_meta()[2])


def _c(c_vec):
    return combine([c_vec[i] for i in range(c_size())], _c_meta())


def setup(x):
    """:func:`..spintaylor.lean.solution` before its loop: ``(c, y_ref,
    k_ref, info)`` with ``info`` the :data:`SETUP_FIELDS`."""
    m1, m2, chi1, chi2, f_ref, f_low, f_cut = _unpack_inputs(x)
    m_tot = m1 + m2
    msec = m_tot / mass_unit() * MTSUN_SI
    pi_gm = PI * m_tot * (g_si() / C_SI) / (C_SI * C_SI)
    dt_fine = 0.5 / f_cut / msec
    y_ref = st_lean._initial_state(m1, m2, chi1, chi2, f_ref, msec)
    c = dynamics.coefficients(m1, m2, f_ref, f_cut)
    k_ref = st_lean._rhs(c, y_ref)
    beyond = ~(st_lean.margin(c, y_ref) > 0.0)
    omega_ref = y_ref[1]
    omega_low = PI * msec * (f_low - 0.5)
    failed = ~(omega_low > 0.0) | ~(omega_low < omega_ref)
    omega_low = jnp.where(failed, 0.5 * omega_ref, omega_low)
    du_step = _du_per_step(m1, m2)
    cover = jnp.where(beyond, 1.25, 1.03)
    range_back = jax.lax.stop_gradient((1.0 / omega_low - 1.0 / omega_ref) * cover)
    n_back = jnp.clip(jnp.ceil(range_back / du_step), 1, st_lean.CAPACITY // 2)
    du_back = range_back / n_back
    f32 = jnp.float32
    info = jnp.stack([beyond.astype(f32), omega_ref, omega_low, failed.astype(f32), du_step, n_back, du_back, dt_fine,
                      msec, pi_gm])
    return jnp.stack(partition(c)[0]), y_ref, k_ref, info


def step_in_time(du, omega, domega):
    """The step in time that changes ``u = 1 / (M omega)`` by ``du``."""
    return trajectory._step_in_time(du[0], omega[0], domega[0])[None]


def step(c_vec, y, k0, h):
    """One DOP853 step of ``h`` from ``y`` (``k0`` its derivative):
    ``(y_new, k_new, dense, margin)`` with the dense output of the whole
    state (7 x 14) and :func:`..spintaylor.lean.margin` at the end."""
    c = _c(c_vec)
    k, y_new = trajectory._stages_looped(c, y, h[0], k0)
    dense = trajectory._dense_looped(y, h[0], k, y_new)
    return y_new, k[dop853.N_STAGES], dense, st_lean.margin(c, y_new)[None]


def rhs(c_vec, y):
    """The derivative of the state (``dynamics.derivatives``)."""
    return st_lean._rhs(_c(c_vec), y)


def going(c_vec, y):
    """:func:`..spintaylor.lean.going`, ``-1`` for NaN (as ``_end``)."""
    g = st_lean.going(_c(c_vec), y)
    return jnp.where(jnp.isnan(g), -1.0, g)[None]


def margin(c_vec, y):
    return st_lean.margin(_c(c_vec), y)[None]


def dense_eval(dense, y0, x):
    """The dense output at ``x`` in the step."""
    return trajectory._dense_eval(dense, y0, x[0])


def finish(last, omega_first, x):
    """The spins and LNhat at the end of the inspiral, ``f_min_pn`` and
    ``M omega`` at the end (``solution``'s last lines)."""
    m1, m2 = x[0], x[1]
    m_tot = m1 + m2
    pi_gm = PI * m_tot * (g_si() / C_SI) / (C_SI * C_SI)
    norm1 = (m1 / m_tot) ** 2
    norm2 = (m2 / m_tot) ** 2
    s1 = jnp.stack([last[5] / norm1, last[6] / norm1, last[7] / norm1])
    s2 = jnp.stack([last[8] / norm2, last[9] / norm2, last[10] / norm2])
    f_min_pn = jnp.cbrt(omega_first[0]) ** 3 / pi_gm
    return jnp.concatenate([s1, s2, last[2:5], jnp.stack([f_min_pn, last[1]])])


def _shapes():
    n_c = c_size()
    n_y = dynamics.NUM_VARIABLES
    s = lambda *shape: jax.ShapeDtypeStruct(shape, np.float32)  # noqa: E731
    return {
        "setup": (setup, [s(N_INPUTS)]),
        "step_in_time": (step_in_time, [s(1), s(1), s(1)]),
        "rhs": (rhs, [s(n_c), s(n_y)]),
        "step": (step, [s(n_c), s(n_y), s(n_y), s(1)]),
        "going": (going, [s(n_c), s(n_y)]),
        "margin": (margin, [s(n_c), s(n_y)]),
        "dense_eval": (dense_eval, [s(trajectory.N_DENSE, n_y), s(n_y), s(1)]),
        "finish": (finish, [s(n_y), s(1), s(N_INPUTS)]),
    }


def _traced(function, avals, tangent):
    def flat(*args):
        with precision(np.float32), lean_mode(True):
            out = function(*args)
        return list(out) if isinstance(out, tuple) else [out]

    if not tangent:
        return lower(jax.make_jaxpr(flat)(*avals))

    def jvp(*args):
        primals, tangents = args[:len(avals)], args[len(avals):]
        out, t_out = jax.jvp(flat, list(primals), list(tangents))
        return list(out) + list(t_out)

    return lower(jax.make_jaxpr(jvp)(*avals, *avals))


@functools.lru_cache(maxsize=None)
def device_source(dual, names=None):
    """The Metal functions ``st_<name>`` of the pieces, on floats or (with
    ``dual``) on :data:`._metal.DUAL` numbers, and the number of
    instructions of each."""
    parts, sizes = [], {}
    for name, (function, avals) in _shapes().items():
        if names is not None and name not in names:
            continue
        program = _traced(function, avals, dual)
        parts.append(_metal.function_source(program, f"st_{name}", n_primal=len(avals) if dual else None))
        sizes[name] = len(program)
    return "\n".join(parts), sizes
