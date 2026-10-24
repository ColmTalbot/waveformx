# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR (:mod:`..lean`) cut into stages, in JAX.

The MLX port runs the model's own JAX code for everything that is a
function of one binary's scalars, or of one binary's scalars and one point:
each such stage is traced to a jaxpr once (:mod:`._translate`) and evaluated
with MLX operations, for a batch of binaries and many points at once by
broadcasting. What does not fit that form is written for MLX by hand: the
integration of the precession equations (a loop), the angles read from its
dense output (searches in tables), the splines of the PNR interpolants and
the moving average of the antisymmetric mode. This module holds the stages
as plain JAX functions of pytrees, and :func:`reference` composes them with
the JAX versions of the hand-written parts: it reproduces
:func:`waveformjax.models.xpnr.lean.polarizations` (the tests check it to
rounding), and its intermediate values are the reference of the MLX port.

The stages, in order (``prec`` and ``wf`` are the model's precession and
waveform structs):

=====  ====================================================================
``A``  the binary to the common set-up and the inputs of the integration
``B``  (by hand) the integration: :func:`..spintaylor.lean.solution`
``C1`` ``finish_setup`` and the 15 points of the set-up's SpinTaylor angles
``B2`` (by hand) the SpinTaylor angles at points (:func:`st_angles`)
``C2`` the rest of the set-up of version 330, up to the grid of the PNR
       interpolants
``D``  per grid point: the PNR ``alpha`` and ``beta`` (``uniform_angles``)
``S``  (by hand) the interpolants: :func:`..pnr._geometric_interpolants`
``E``  the offsets at the reference frequency, the remapped J frame and the
       coefficients of the modes and of the antisymmetric mode
``F1`` per frequency: the frequencies at which each mode reads the angles,
       and the amplitude ratio of the antisymmetric mode
``F2`` per frequency: the angles, the modes, the twist and the
       polarizations
=====  ====================================================================
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .. import hm, model, pnr
from .. import pnr_angles as pa
from .._math import (
    MTSUN_SI, PI, Struct, fma, hz_to_mf, lean as lean_mode, mf_to_hz, powers, precision, real, safe_div, safe_sqrt,
    tiny,
)
from ..amp22 import amp_ansatz
from ..antisymmetric import _kappa_function, antisymmetric_coefficients, smooth_amp_ratio
from ..internals import errors as wf_errors
from ..phase22 import phase_xas
from ..spintaylor import angles as st_angles
from ..spintaylor import lean as st_lean
from ..xas import h22, phif_ref

__all__ = ["BANK_GROUPS", "stage_a", "stage_c1", "stage_c2", "grid_point", "stage_e", "stage_f1", "stage_f2",
           "spline_tail", "grid_frequencies", "reference", "binary_leaves", "BINARY_FIELDS"]

#: The fields of a :class:`~waveformjax.Binary`, in order.
BINARY_FIELDS = ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination", "phi_ref", "f_ref")
#: The groups of points at which the set-up reads the SpinTaylor angles
#: (:func:`..pnr.st_bank`), and their sizes.
BANK_GROUPS = (("euler", 4), ("alpha", 4), ("beta", 3), ("antisymmetric", 4))
#: The interpolants of the smooth PNR angles (:func:`..pnr._geometric_interpolants`).
SPLINES = ("u", "v", "w", "plus", "minus", "unwrapped")


def binary_leaves(binary):
    """The fields of a binary as a dict."""
    return {name: getattr(binary, name) for name in BINARY_FIELDS}


def _binary(leaves):
    from ...._binary import Binary

    return Binary(**{name: leaves[name] for name in BINARY_FIELDS})


# ---------------------------------------------------------------------------
# A: the common set-up
# ---------------------------------------------------------------------------


def stage_a(leaves, f_min, f_max, modes, amp_scale):
    """``model._decide`` of the lean model up to the integration: returns
    ``(base, wf0, ode, error)`` with ``ode`` the inputs of
    :func:`..spintaylor.lean.solution` (``m1``, ``m2`` in solar masses, the
    spins ``chi1``, ``chi2``, ``f_ref``, ``f_low``, ``f_cut``, whether the
    integration is needed and whether its start is valid)."""
    pnr.clear_cache()
    binary = _binary(leaves)
    wf, m1, m2, chi1, chi2 = model.setup_waveform(binary, 0.0, f_min, f_max, amp_scale)
    wf.hm_release = 122019
    error = wf_errors(wf)
    error = error | ~((wf.f_ref >= f_min) & (wf.f_ref <= f_max))
    mode_range = pnr.mode_range(modes)
    base, wf0 = pnr.common_setup(wf, m1, m2, chi1, chi2, mode_range, True, True, True, True, False, True, mode_range)
    error = error | base.error_spin | (wf0.f_max_prime <= wf0.f_min)
    needs_st = base.pnr_on & ~base.nnlo_from_start
    flow_safe = jnp.where(base.st_flow_ok, base.st_flow_lean, wf0.f_min)
    ode = Struct(m1=m1, m2=m2, chi1=jnp.stack(base.chi1), chi2=jnp.stack(base.chi2), f_ref=wf0.f_ref, f_low=flow_safe,
                 f_cut=base.st_fcut, active=needs_st, flow_ok=base.st_flow_ok)
    return base, wf0, ode, error


def solution_summary(sol, flow_ok):
    """What the scalar set-up reads of the SpinTaylor solution: the spins
    and LNhat at the end of the inspiral, the start of the solution and its
    range in ``M omega``, and its failure flags."""
    return Struct(
        s1_end=jnp.stack(sol.s1_end), s2_end=jnp.stack(sol.s2_end), lnhat_end=jnp.stack(sol.lnhat_end),
        f_min_pn=sol.f_min_pn, omega_first=sol.omega_first, omega_last=sol.omega_last,
        failed=sol.failed | ~flow_ok, incomplete=sol.incomplete,
    )


def _st_stub(summary):
    """A stand-in for the SpinTaylor solution in ``prec.st``: its scalars,
    and markers for the functions that look for the arrays (which the
    stages never read)."""
    dummy = jnp.zeros((1,), real())
    return Struct(lean=jnp.zeros((), real()), y=dummy, dense=dummy, h=dummy, f_min_pn=summary.f_min_pn,
                  lnhat_end=list(summary.lnhat_end), s1_end=list(summary.s1_end), s2_end=list(summary.s2_end))


def _st_range(prec, wf, summary):
    """The scale from ``M omega`` to ``Mf`` and the range of the solution
    in ``Mf`` (:func:`..spintaylor.lean.context`)."""
    mf_scale = MTSUN_SI * wf.m_tot / prec.pi_gm
    first_mf = jnp.cbrt(summary.omega_first) ** 3 * mf_scale
    last_mf = jnp.cbrt(summary.omega_last) ** 3 * mf_scale
    return mf_scale, first_mf, last_mf


# ---------------------------------------------------------------------------
# C1, C2: the set-up of version 330
# ---------------------------------------------------------------------------


def stage_c1(base, wf0, summary):
    """``finish_setup`` and :func:`..pnr.st_bank` up to the evaluation of
    the SpinTaylor angles: returns ``(prec, wf, points, rotation,
    st_range)`` with the 15 points (``Mf``) of :data:`BANK_GROUPS`, the J
    frame of the angles and ``(mf_scale, first_mf, last_mf)``."""
    pnr.clear_cache()
    prec, wf = pnr.finish_setup(base, wf0, 330, _st_stub(summary))
    prec.st_rotation = pnr._rotation(prec)
    mf_scale, first_mf, last_mf = _st_range(prec, wf, summary)
    context = Struct(lean=jnp.zeros((), real()), first_mf=first_mf, last_mf=last_mf)
    fmax_inspiral, euler = pnr._euler_points(prec, wf, context)
    ap = pnr.precompute_alpha_coefficients(prec, wf, 330)
    bp = pa.beta_connection_frequencies(pnr.precompute_beta_coefficients(prec, wf, 330), pnr._soft(prec))
    _, _, alpha = pnr._alpha_points(prec, wf, ap, 330)
    _, beta = pnr._st_beta_points(prec, bp, 0.98 * fmax_inspiral)
    _, antisymmetric = pnr.antisymmetric_points(wf)
    prec.st_bank_params = Struct(alpha=ap, beta=bp)
    points = jnp.concatenate([euler, alpha, beta, antisymmetric])
    return prec, wf, points, jnp.stack(prec.st_rotation), jnp.stack([mf_scale, first_mf, last_mf])


def _with_bank(prec, points, values, st_range):
    """``prec`` with the SpinTaylor angles at the bank's points
    (:func:`..pnr.st_bank`) and the range of the solution."""
    _, first_mf, last_mf = st_range[0], st_range[1], st_range[2]
    prec.st_context = Struct(lean=jnp.zeros((), real()), first_mf=first_mf, last_mf=last_mf)
    bank, start = Struct(), 0
    for name, n in BANK_GROUPS:
        setattr(bank, name, Struct(mf=points[start:start + n], values=tuple(v[start:start + n] for v in values)))
        start += n
    prec.st_bank = bank
    return prec


def stage_c2(prec, wf, points, values, st_range, modes):
    """The rest of ``model._lean_setup`` with the SpinTaylor angles at the
    bank's points ``values`` (``(alpha, dalpha, cosbeta, dcosbeta,
    inside)``, each of shape ``(15,)``): returns ``(prec, wf, grid,
    error)`` with ``grid = (i_start, n, delta_f)`` of the PNR interpolants."""
    pnr.clear_cache()
    prec = _with_bank(prec, points, values, st_range)
    prec = pnr.initialize_euler_angles(prec, wf)
    prec = pnr.populate_structs(prec, wf, 330)
    grid = pnr.interpolation_grid(prec, wf, modes, 330)
    f0 = grid[0] * grid[2]
    error = prec.error_structs | grid[3] | prec.error_euler | (prec.fmin_integration > f0)
    return prec, wf, jnp.stack([grid[0], grid[1], grid[2]]), error


# ---------------------------------------------------------------------------
# D: the PNR angles on the grid, per point
# ---------------------------------------------------------------------------


def grid_frequencies(k, grid, m_tot):
    """``Mf`` of the points ``k`` (real) of the grid of the PNR
    interpolants (the last point repeated beyond ``n``)."""
    i_start, n, h = grid[0], grid[1], grid[2]
    return hz_to_mf((i_start + jnp.minimum(k, n - 1.0)) * h, m_tot)


def grid_point(prec, wf, mf, st_values):
    """``uniform_angles`` of the smooth angles at one point ``mf`` with
    the SpinTaylor angles ``st_values`` there: ``(alpha, beta, error)``."""
    pnr.clear_cache()
    return pnr.uniform_angles(prec, wf, mf, 330, tuple(st_values), True)


# ---------------------------------------------------------------------------
# E: offsets, remapped J frame and the coefficients of the modes
# ---------------------------------------------------------------------------


def spline_tail(values):
    """``(alpha, beta, gamma)`` of :func:`..pnr.spline_angles` from the
    values of the interpolants (:data:`SPLINES`) at a point."""
    u, v, w, plus, minus, unwrapped = values
    a = st_angles.small_atan2(v, u)
    b = st_angles._atan2(safe_sqrt(u * u + v * v, tiny()), w)
    a_u = unwrapped + pnr._wrap(a - unwrapped)
    weight = pa._ramp(w / safe_sqrt(u * u + v * v + w * w, tiny()), pnr.HEMISPHERE_SOFT)
    gamma = fma(weight, plus - a_u, (1.0 - weight) * (minus + a_u))
    return a, b, gamma


def stage_e(prec, wf, at_ref, inside_ref, modes):
    """``_prepare_pnr`` after the interpolants (``at_ref``: the values of
    :data:`SPLINES` at ``f_ref``; ``inside_ref``: whether ``f_ref`` lies on
    the grid) and the frequency-independent parts of ``_angle_data`` and
    ``_twist_polarizations``. Returns ``(prec, wf, extra, error)``."""
    pnr.clear_cache()
    alpha_ref, beta_ref, gamma_ref = spline_tail(list(at_ref))
    error = ~inside_ref | ~jnp.isfinite(beta_ref)
    prec.alpha_offset = alpha_ref
    prec.epsilon_offset = -gamma_ref - prec.epsilon0
    prec = pnr.remap_theta_jsf(prec, wf, beta_ref)
    extra = Struct()
    if (2, 2) in modes:
        co = antisymmetric_coefficients(prec, wf, 330)
        error = error | co.error
        extra.co = co
        amp22, phase22, linb = co.amp22, co.phase22, co.linb
    else:
        from ..xas import xas_setup

        xas = xas_setup(wf)
        amp22, phase22, linb = xas.amp, xas.phase, xas.linb
    wf_out = Struct(**{name: getattr(wf, name) for name in model.WF_FIELDS})
    wf_out.hm_release = jnp.asarray(122019)
    wf_out.exact_derivatives = jnp.zeros((), real())
    xas = Struct(amp=amp22, phase=phase22, linb=linb, lina=0.0, inveta=1.0 / wf_out.eta,
                 amp_prefactor=wf_out.amp0 * wf_out.amp_norm)
    higher = [mode for mode in modes if mode != (2, 2)]
    extra.coefficients = list(hm.modes_coefficients(wf_out, phase22, higher))
    extra.xas = xas
    extra.phif = phif_ref(wf_out, xas)
    extra.ylms = {ell: model.spherical_harmonics(prec.theta_jn, ell) for ell in sorted({e for e, _ in modes})}
    extra.wf_out = wf_out
    return prec, wf, extra, error


# ---------------------------------------------------------------------------
# F1, F2: per frequency
# ---------------------------------------------------------------------------


def stage_f1(prec, wf, f, modes):
    """At one frequency ``f`` (Hz): the frequencies (Hz) at which each mode
    reads the PNR angles and whether the map failed, ``Mf`` of the
    antisymmetric mode and its amplitude ratio before the moving average."""
    pnr.clear_cache()
    mf = wf.m_sec * f
    mapped, errors = [], []
    for ell, emm in modes:
        if (ell, emm) == (2, 2):
            mf_mapped, map_error = mf, jnp.zeros((), bool)
        else:
            mf_rd_lm = pnr.ringdown_frequency(ell, emm, wf)
            low, high = pa.linear_frequency_map_transition_frequencies(emm, wf.f_ring, mf_rd_lm, prec.pnr_hm_mflow,
                                                                       prec.pnr_hm_mfhigh)
            mf_mapped, map_error = pnr.frequency_map(prec, mf, ell, emm, low, high, wf.f_ring, mf_rd_lm)
        mapped.append(mf_to_hz(mf_mapped, wf.m_tot))
        errors.append(map_error)
    mf_as = hz_to_mf(f, wf.m_tot)
    if (2, 2) in modes:
        kappa_at, kappa_rd = _kappa_function(prec, wf)
        below = mf_as < wf.f_ring
        kappa = jnp.where(below, kappa_at(jnp.cbrt(PI * jnp.where(below, mf_as, wf.f_ring))), kappa_rd)
    else:
        kappa = jnp.zeros((), real())
    return jnp.stack(mapped), jnp.stack(errors), mf_as, kappa


def stage_f2(prec, wf, extra, f, spline_values, inside, map_errors, kappa, st_alpha, st_inside, modes):
    """At one frequency ``f`` (Hz): ``(h_+, h_x)`` as ``(re, im)`` pairs
    and whether the point makes the waveform fail. ``spline_values``
    (``(n_modes, 6)``) and ``inside`` are the interpolants at the mapped
    frequencies of the modes, ``kappa`` the smoothed amplitude ratio and
    ``st_alpha``, ``st_inside`` the SpinTaylor ``alpha`` at ``Mf`` of the
    antisymmetric mode."""
    pnr.clear_cache()
    wf_out = extra.wf_out
    mf = wf.m_sec * f
    keep = mf <= wf.f_max_prime * wf.m_sec
    vanishing_odd = (wf.q == 1.0) & (wf.chi1l == wf.chi2l)
    error = jnp.zeros((), bool)
    angles = []
    for index, (ell, emm) in enumerate(modes):
        a, b, g = spline_tail([spline_values[index, j] for j in range(len(SPLINES))])
        err = map_errors[index] | ~inside[index]
        if emm % 2:
            err = err & ~vanishing_odd
        error = error | (err & keep)
        angles.append((a - prec.alpha_offset, jnp.cos(b), -g - prec.epsilon_offset))
    h_as = None
    if (2, 2) in modes:
        co = extra.co
        mf_as = hz_to_mf(f, wf.m_tot)
        nan = jnp.full_like(st_alpha, jnp.nan)
        pnr._st_spline_put(prec, wf, mf_as, (st_alpha, nan, nan, nan, st_inside))
        alpha_pnr, err_alpha = pnr.region_alpha_at(prec, wf, mf_as, co.alpha_params, 330)
        used = mf_as < co.mf_t
        error = error | (err_alpha & used & (mf_as <= wf.f_max_prime * wf.m_sec))
        p = powers(mf)
        phi = phase_xas(mf, p, co.phase22) * co.inveta
        phi = phi + (fma(co.linb, mf, co.lina) + co.phiref22)
        a = amp_ansatz(mf, p, co.amp22)
        phi_as = jnp.where(mf < co.mf_t, fma(co.a0, mf, phi / 2.0 + alpha_pnr) + co.phi_a0, phi + co.phi_b0)
        amp = co.amp_prefactor * p.m_seven_sixths * (kappa * a)
        h_as = amp * jnp.exp(1j * (co.phase_offset + phi_as))
    coefficients = dict(zip([m for m in modes if m != (2, 2)], extra.coefficients))
    hp = jnp.zeros((), complex)
    hc = jnp.zeros((), complex)
    mf_modes = wf_out.m_sec * f
    for index, (ell, emm) in enumerate(modes):
        if (ell, emm) == (2, 2):
            h = h22(mf_modes, wf_out, extra.xas, extra.phif, mf_cut=jnp.inf)
        else:
            h = hm.mode_h(coefficients[(ell, emm)], mf_modes)
        twist = model.twist_factors(*angles[index], emm)
        p_, c_ = model.twist_up(ell, emm, twist, extra.ylms[ell], h, 1.0)
        if (ell, emm) == (2, 2):
            pa_, ca_ = model.twist_up(ell, emm, twist, extra.ylms[ell], h_as, -1.0)
            p_ = p_ + pa_
            c_ = c_ + ca_
        if emm % 2:
            p_ = jnp.where(vanishing_odd, 0.0, p_)
            c_ = jnp.where(vanishing_odd, 0.0, c_)
        hp = hp + jnp.where(keep, p_, 0.0)
        hc = hc + jnp.where(keep, c_, 0.0)
    hp, hc = model.rotate_polarizations(prec.zeta_polarization, hp, hc)
    return jnp.stack([jnp.real(hp), jnp.imag(hp), jnp.real(hc), jnp.imag(hc)]), error


# ---------------------------------------------------------------------------
# the stages composed in JAX
# ---------------------------------------------------------------------------


def partition(tree):
    """The array leaves of a pytree and a hashable description of the rest
    (its structure and its Python leaves), as ``model._partition``."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    dynamic = tuple(isinstance(leaf, (jax.Array, np.ndarray, np.generic)) for leaf in leaves)
    values = [leaf for leaf, d in zip(leaves, dynamic) if d]
    static = tuple(None if d else leaf for leaf, d in zip(leaves, dynamic))
    return values, (treedef, static, dynamic)


def combine(values, meta):
    """The pytree of :func:`partition` from its array leaves."""
    treedef, static, dynamic = meta
    values = iter(values)
    return jax.tree_util.tree_unflatten(treedef, [next(values) if d else s for s, d in zip(static, dynamic)])


class Staged:
    """``function(*pytrees)`` under ``jax.jit`` in a given precision, with
    the Python leaves of its arguments and results kept static (the
    structs of the model carry Python flags)."""

    def __init__(self, function, dtype):
        self.function = function
        self.dtype = np.dtype(dtype)
        self.cache = {}

    def __call__(self, *args):
        parts = [partition(a) for a in args]
        metas = tuple(meta for _, meta in parts)
        if metas not in self.cache:
            holder = {}

            def flat(*values):
                trees = [combine(v, meta) for v, meta in zip(values, metas)]
                with precision(self.dtype), lean_mode(True):
                    out, holder["meta"] = partition(self.function(*trees))
                return out

            self.cache[metas] = (jax.jit(flat), holder)
        jitted, holder = self.cache[metas]
        out = jitted(*[v for v, _ in parts])
        return combine(out, holder["meta"])


@functools.lru_cache(maxsize=None)
def _compiled(dtype_name, modes, amp_scale):
    """The stages and the parts by hand as jitted JAX functions (in the
    precision ``dtype_name``)."""
    dtype = np.dtype(dtype_name)

    def under(function):
        return Staged(function, dtype)

    def a(leaves, f_min, f_max):
        return stage_a(leaves, f_min, f_max, modes, amp_scale)

    def b(ode):
        sol = st_lean.solution(ode.m1, ode.m2, list(ode.chi1), list(ode.chi2), ode.f_ref, ode.f_low, ode.f_cut,
                               ode.active)
        return sol, solution_summary(sol, ode.flow_ok)

    def b2(sol, rotation, st_range, mf):
        mf_scale = st_range[0]
        ctx = st_lean.context(sol, tuple(rotation), mf_scale)
        return st_lean.angles(sol, tuple(rotation), mf, mf_scale, ctx)

    def c2(prec, wf, points, values, st_range):
        return stage_c2(prec, wf, points, values, st_range, modes)

    def d(prec, wf, mf, st_values):
        return jax.vmap(grid_point, in_axes=(None, None, 0, 0))(prec, wf, mf, st_values)

    def s(alpha, beta, grid):
        i_start, n, h = grid[0], grid[1], grid[2]
        return pnr._geometric_interpolants(alpha, beta, i_start, n, h, n, jnp.asarray(False))

    def spline_values(splines, f):
        out = [pnr._spline_at(getattr(splines, name), getattr(splines, "m_" + name), splines.n, splines.x0,
                              splines.h, f)[0] for name in SPLINES]
        inside = (f >= splines.x0) & (f <= splines.x0 + (splines.n - 1.0) * splines.h)
        return jnp.stack(out, axis=-1), inside

    def e(prec, wf, at_ref, inside_ref):
        return stage_e(prec, wf, at_ref, inside_ref, modes)

    def f1(prec, wf, f):
        return jax.vmap(stage_f1, in_axes=(None, None, 0, None))(prec, wf, f, modes)

    def f2(prec, wf, extra, f, values, inside, map_errors, kappa, st_alpha, st_inside):
        return jax.vmap(stage_f2, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, None))(
            prec, wf, extra, f, values, inside, map_errors, kappa, st_alpha, st_inside, modes)

    def smooth(kappa, f):
        return smooth_amp_ratio(kappa, f)

    return Struct(a=under(a), b=under(b), c1=under(stage_c1), b2=under(b2), c2=under(c2), d=under(d), s=under(s),
                  spline_values=under(spline_values), e=under(e), f1=under(f1), f2=under(f2), smooth=under(smooth))


def reference(binary, frequencies, mode_array=None, scaled_strain=True, keep=False):
    """The lean polarizations from the stages composed in JAX (eagerly,
    stage by stage, with the sizes of the binary), in the precision of the
    binary: ``(h_+, h_x)``, and with ``keep`` also a dict of the
    intermediate values of every stage."""
    from ...._options import STRAIN_SCALE
    from .._math import precision_of

    dtype = precision_of(binary)
    modes = model.coprecessing_modes(mode_array)
    st = _compiled(dtype.name, modes, STRAIN_SCALE if scaled_strain else 1.0)
    leaves = {name: jnp.asarray(value, dtype) for name, value in binary_leaves(binary).items()}
    f = jnp.asarray(frequencies, dtype)
    out = {}
    base, wf0, ode, error_a = st.a(leaves, f[0], f[-1])
    sol, summary = st.b(ode)
    prec, wf, points, rotation, st_range = st.c1(base, wf0, summary)
    bank = st.b2(sol, rotation, st_range, points)
    prec, wf, grid, error_c = st.c2(prec, wf, points, bank, st_range)
    n = int(grid[1])
    k = jnp.arange(n, dtype=dtype)
    mf_grid = grid_frequencies(k, grid, wf.m_tot)
    st_grid = st.b2(sol, rotation, st_range, mf_grid)
    alpha, beta, err_grid = st.d(prec, wf, mf_grid, st_grid)
    capacity = pnr.PNR_GRID_CAPACITY
    pad = capacity - n
    splines = st.s(jnp.pad(alpha, (0, pad)), jnp.pad(beta, (0, pad)), grid)
    at_ref, inside_ref = st.spline_values(splines, wf.f_ref)
    prec, wf, extra, error_e = st.e(prec, wf, at_ref, inside_ref)
    mapped, map_errors, mf_as, kappa = st.f1(prec, wf, f)
    values, inside = st.spline_values(splines, mapped)
    kappa_s = st.smooth(kappa, f) if (2, 2) in modes else kappa
    st_f = st.b2(sol, rotation, st_range, mf_as)
    h, err_f = st.f2(prec, wf, extra, f, values, inside, map_errors, kappa_s, st_f[0], st_f[4])
    error = (error_a | ~ode.active | summary.failed | error_c | summary.incomplete | jnp.any(err_grid)
             | error_e | jnp.any(err_f))
    nan = jnp.nan
    hp = jnp.where(error, nan, h[:, 0] + 1j * h[:, 1])
    hc = jnp.where(error, nan, h[:, 2] + 1j * h[:, 3])
    if keep:
        out.update(base=base, wf0=wf0, ode=ode, error_a=error_a, sol=sol, summary=summary, points=points,
                   rotation=rotation, st_range=st_range, bank=bank, prec_c2=prec, grid=grid, error_c=error_c,
                   mf_grid=mf_grid, st_grid=st_grid, alpha=alpha, beta=beta, err_grid=err_grid, splines=splines,
                   at_ref=at_ref, inside_ref=inside_ref, extra=extra, error_e=error_e, mapped=mapped,
                   map_errors=map_errors, mf_as=mf_as, kappa=kappa, kappa_s=kappa_s, values=values, inside=inside,
                   st_f=st_f, h=h, err_f=err_f, error=error)
        return (hp, hc), out
    return hp, hc
