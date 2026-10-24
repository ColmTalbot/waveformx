# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The amplitude of the IMRPhenomXHM higher modes in the original release,
``PhenomXHMReleaseVersion = 122019`` (arXiv:2001.10914), as used by
IMRPhenomXPNR: ``IMRPhenomXHM_GetAmplitudeCoefficients`` and the amplitude
ansatzes of ``LALSimIMRPhenomXHM_internals.c``, after waveformrs's
``imrphenomx/hm_amp_122019.rs``.

The release uses the 122018 parameter-space fits, three inspiral
collocation points on top of the re-expanded PN amplitude (or the TaylorT4
based amplitude of the (2,1)), an intermediate amplitude that is the inverse
of a polynomial whose order is lowered when collocation points are vetoed or
when it crosses zero, two intermediate regions for extreme mass ratios
(``AmpEMR``) and the original ringdown ansatz. Every discrete choice of LAL
is a traced boolean selecting among values computed for all cases; the
zero-crossing tests (:mod:`.roots`) see the coefficients through
``jax.lax.stop_gradient``.

Accuracy: the intermediate polynomial is fixed by the derivative of the
inspiral amplitude (and of the rotated ringdown of the (3,2)), which LAL
takes by finite differences with a step of 1e-9 (:mod:`.finite_differences`).
Rounding differences in the ansatz become ~1e-7 in the derivative and ~1e-8
in the intermediate amplitude, the level at which different builds of LAL
differ; elsewhere the amplitude agrees with LAL to ~1e-13.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .._generated import hm_amp_122019 as gen
from .._generated import hm_fits_122018 as fits
from .._math import Struct, fma, powers
from . import roots
from .finite_differences import value_and_nd_derivative
from .inspiral import pn21_coefficients, pn_amp_rescaled, pn_amplitude_coefficients

__all__ = ["amplitude_coefficients", "amplitude", "choose_polynomial", "finish_amplitude"]

def select(condition, a, b):
    """``a`` where ``condition`` else ``b``, leaf by leaf."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(condition, x, y), a, b)


# the conditions fixing the intermediate polynomial of each reconstruction:
# its value at f1, f2, f3, f4 ("v") and its derivative at f1, f4 ("d")
_CONDITIONS = {
    101: (("v", 1), ("v", 4)),
    102: (("v", 1), ("v", 4), ("d", 4)),
    1032: (("v", 1), ("v", 4), ("d", 1), ("d", 4)),
    104: (("v", 1), ("v", 2), ("v", 4), ("d", 1), ("d", 4)),
    1042: (("v", 1), ("v", 3), ("v", 4), ("d", 1), ("d", 4)),
    105: (("v", 1), ("v", 2), ("v", 3), ("v", 4), ("d", 1), ("d", 4)),
}
_INPUTS = ("d1", "d4", "v1", "v2", "v3", "v4", "f1", "f2", "f3", "f4")


def _make_polynomial_coefficients(version):
    """``delta0`` ... ``delta5`` of reconstruction ``version``: the solution
    of its collocation conditions ``M(f) delta = (v, d)`` (LAL writes out
    the closed forms of the solution (waveformrs's ``hm_poly_122019.rs``);
    the same to rounding, which one small linear solve compiles to much less
    code than). The polynomial is solved for in powers of ``f / f4``, which
    keeps the system well conditioned."""
    conditions = _CONDITIONS[version]
    n = len(conditions)

    def coefficients(x):
        values = dict(zip(_INPUTS, x))
        scale = values["f4"]
        rows = []
        rhs = []
        for kind, i in conditions:
            g = values[f"f{i}"] / scale
            powers_of_g = [jnp.ones_like(g)]
            for _ in range(1, n):
                powers_of_g.append(powers_of_g[-1] * g)
            if kind == "v":
                rows.append(powers_of_g)
                rhs.append(values[f"v{i}"])
            else:
                rows.append([jnp.zeros_like(g)] + [k * powers_of_g[k - 1] for k in range(1, n)])
                rhs.append(values[f"d{i}"] * scale)
        matrix = jnp.stack([jnp.stack(row) for row in rows])
        scaled = jnp.linalg.solve(matrix, jnp.stack(rhs))
        delta = scaled / scale ** jnp.arange(n)
        return jnp.concatenate([delta, jnp.zeros(6 - n)])

    return coefficients


_POLYNOMIAL_COEFFICIENTS = {version: _make_polynomial_coefficients(version) for version in _CONDITIONS}


def polynomial_coefficients(inputs, version):
    """The coefficients ``delta0`` ... ``delta5`` of the intermediate
    polynomial of reconstruction ``version`` (101, 102, 1032, 104, 1042,
    105) for the collocation data ``inputs`` (``v1`` ... ``v4``, ``d1``,
    ``d4``, ``f1`` ... ``f4``), ``IMRPhenomXHM_Intermediate_Amp_delta*``."""
    x = jnp.stack([jnp.asarray(inputs[name] if isinstance(inputs, dict) else getattr(inputs, name), jnp.float64)
                   for name in _INPUTS])
    delta = _POLYNOMIAL_COEFFICIENTS[version](x)
    return [delta[k] for k in range(6)]


def amplitude_fcut_insp(hm, wf):
    """``IMRPhenomXHM_Amplitude_fcutInsp`` (frequencies version 122018): the
    end of the inspiral region of the amplitude."""
    f_meco = hm.f_meco_lm
    emm = float(hm.emm)
    eta = wf.eta
    chi1 = wf.chi1l
    fcut_emr = 1.25 * emm * (
        fma(0.000316707064291237, chi1 * chi1, fma(-0.0000858396080377194, chi1, 0.011671068725758493))
        * fma(6.2873167352395125, eta, 0.8447212540381764)
    ) / fma(-0.9977728883419751, chi1, 1.2857082764038923)
    chieff = wf.chi_eff
    f_isco = wf.f_isco * emm * 0.5
    df = jnp.abs(f_isco - f_meco)
    tag = hm.mode_tag
    if tag == 21:
        factor = fma(-(5.0 / 6.0 * chieff), chieff, fma(-0.235, chieff, 0.75))
        return jnp.where(eta < 0.023795359904818562, fcut_emr, fma(factor, df, f_meco))
    if tag == 33:
        factor = fma(-(5.0 / 6.0), chieff, fma(-0.235, chieff, 0.75))
        return jnp.where(eta < 0.04535147392290249, fcut_emr, fma(factor, df, f_meco))
    if tag == 32:
        fcut = fma(fma(-0.235, jnp.abs(chieff), 0.75), df, f_meco)
        return jnp.where(eta < 0.04535147392290249, fcut_emr, fcut * hm.f_ring / wf.f_ring)
    return jnp.where(eta < 0.04535147392290249, fcut_emr, fma(fma(-0.235, chieff, 0.75), df, f_meco))


def amplitude_fcut_rd(hm, wf):
    """``IMRPhenomXHM_Amplitude_fcutRD`` (version 122018): the start of the
    ringdown region of the amplitude."""
    fring = hm.f_ring
    fdamp = hm.f_damp
    eta = wf.eta
    chi1 = wf.chi1l
    tag = hm.mode_tag
    if tag == 21:
        return 0.75 * fring
    if tag == 33:
        return 0.95 * fring
    if tag == 32:
        f_rd22 = wf.f_ring
        c, r = 0.5, 5.0
        ecr = jnp.exp(jnp.asarray(c * r))
        erc = jnp.exp(r * chi1)
        # a smooth step between fRING (negative spins) and fRD22 (positive spins)
        fcut = jnp.where(eta < 0.0453515, fma(fring, ecr, f_rd22 * erc) / (ecr + erc) - fdamp, f_rd22)
        corner = (0.02126654064272212 < eta) & (eta < 0.12244897959183673) & (chi1 > 0.95)
        return jnp.where(corner, fma(-2.0, fdamp, fring), fcut)
    return 0.9 * fring


def wavy_points(p1, p2, p3):
    """``WavyPoints``: whether three consecutive values are not monotonic."""
    return ((p1 > p2) & (p2 < p3)) | ((p1 < p2) & (p2 > p3))


def inspiral_rescaled(p, amp):
    """``IMRPhenomXHM_Inspiral_Amp_Ansatz`` with ``InspRescaleFactor = 1``:
    PN plus pseudo-PN terms, rescaled by the leading order of the 22."""
    pn = pn_amp_rescaled(p, amp.pn, amp.amp_norm, amp.pn21, amp.use_famp_pn)
    rho = amp.rho
    insp = fma(p.seven_thirds / amp.fcut_insp_seven_thirds, rho[0], pn)
    insp = fma(p.eight_thirds / amp.fcut_insp_eight_thirds, rho[1], insp)
    return fma(p.three / amp.fcut_insp_three, rho[2], insp)


def inspiral_ansatz(p, amp):
    """``IMRPhenomXHM_Inspiral_Amp_Ansatz``: the inspiral amplitude (strain
    without ``amp0``)."""
    return inspiral_rescaled(p, amp) * (p.m_seven_sixths * amp.amp_norm)


def rd_amp_ansatz(mf, amp):
    """``IMRPhenomXHM_RD_Amp_Ansatz`` (version 0): the ringdown amplitude
    (the spheroidal one for the (3,2)) rescaled by the leading order of the
    22."""
    rd = amp.rd
    return gen.rd_amp_ansatz0(mf, amp.f_ring, amp.f_damp, rd[0], rd[1], rd[2], rd[3])


def rd_amp_dansatz(mf, amp):
    """``IMRPhenomXHM_RD_Amp_DAnsatz`` (version 0)."""
    rd = amp.rd
    return gen.rd_amp_dansatz0(mf, amp.f_ring, amp.f_damp, rd[0], rd[1], rd[2], rd[3])


def _part(p, i):
    """The ``i``-th entry of a struct of arrays."""
    return jax.tree_util.tree_map(lambda x: x[i], p)


def _rho(version, iv, p_cut, p, p_orig):
    """The pseudo-PN coefficients ``rho1``, ``rho2``, ``rho3`` for
    ``IMRPhenomXHMInspiralAmpVersion = version`` collocation points (the
    other versions are evaluated on the original, distinct frequencies so
    that they stay finite)."""
    iv1, iv2, iv3 = iv
    p1, p2, p3 = p
    o1, o2, o3 = p_orig
    rho3 = [
        gen.rho1_3(iv1, iv2, iv3, o1, o1, o2, o3),
        gen.rho2_3(iv1, iv2, iv3, o1, o1, o2, o3),
        gen.rho3_3(iv1, iv2, iv3, o1, o1, o2, o3),
    ]
    s1 = select(version == 2, p1, o1)
    s2 = select(version == 2, p2, o2)
    rho2 = [gen.rho1_2(iv1, iv2, p_cut, s1, s2), gen.rho2_2(iv1, iv2, p_cut, s1, s2), 0.0]
    rho1 = [gen.rho1_1(iv1, p_cut, p1), 0.0, 0.0]
    return [
        jnp.where(version == 3, rho3[k], jnp.where(version == 2, rho2[k], jnp.where(version == 1, rho1[k], 0.0)))
        for k in range(3)
    ]


def amplitude_coefficients(wf, hm, rotate=None, defer=False):
    """``IMRPhenomXHM_GetAmplitudeCoefficients`` for the 122019 release.

    ``hm`` is :func:`~.variables.hm_waveform_struct`; for the (3,2) mode
    ``rotate(mf, rd)`` is the spherical ringdown (``SpheroidalToSpherical``,
    ``(re, im)`` rescaled by the leading order of the 22) for the ringdown
    coefficients ``rd``, at the frequencies ``mf``. With ``defer`` the
    intermediate polynomial is left to :func:`choose_polynomial` and
    :func:`finish_amplitude`."""
    tag = hm.mode_tag
    mode_int = hm.mode_int
    amp_norm = wf.amp_norm
    q, eta, chi1l, chi2l = wf.q, wf.eta, wf.chi1l, wf.chi2l
    if tag in (44, 33):
        use_insp_rd = (q > 7.0) & (chi1l > 0.95)
    else:
        use_insp_rd = jnp.asarray(False)

    f_in = amplitude_fcut_insp(hm, wf)
    f_im = amplitude_fcut_rd(hm, wf)
    df = f_im - f_in
    freqs_inter = [f_in + df / 3.0, f_in + df * 2.0 / 3.0]

    # ------------------------------ inspiral ------------------------------
    pn = lax.optimization_barrier(pn_amplitude_coefficients(wf, hm))
    pn21 = lax.optimization_barrier(pn21_coefficients(wf)) if tag == 21 else None
    use_famp_pn = jnp.asarray(hm.use_famp_pn)
    amp = Struct(
        amp_norm=amp_norm, pn=pn, pn21=pn21, use_famp_pn=use_famp_pn, f_ring=hm.f_ring, f_damp=hm.f_damp,
    )

    fcut = f_in
    p123 = powers(jnp.stack([fcut, 0.75 * fcut, 0.5 * fcut]))
    p1, p2, p3 = _part(p123, 0), _part(p123, 1), _part(p123, 2)
    p_cut = p1
    pn_insp = pn_amp_rescaled(p123, pn, amp_norm, pn21, use_famp_pn)
    colloc = [jnp.abs(fit(wf)) for fit in fits.inspiral_fits(mode_int)]
    iv1 = colloc[0] - pn_insp[0]
    iv2 = colloc[1] - pn_insp[1]
    iv3 = colloc[2] - pn_insp[2]

    # vetoes of the collocation points outside the calibration region
    version = jnp.asarray(3)
    if tag == 21:
        veto = hm.inspiral_amp_veto
        threshold = 0.2 / hm.amp_norm
        c = veto & (colloc[0] < threshold * p1.seven_sixths)
        iv1 = jnp.where(c, 0.0, iv1)
        version = jnp.where(c, 2, version)
        c = veto & (colloc[1] < threshold * p2.seven_sixths)
        iv2 = jnp.where(c, 0.0, iv2)
        version = jnp.where(c, version - 1, version)
        c = veto & (colloc[2] < threshold * p3.seven_sixths)
        iv3 = jnp.where(c, 0.0, iv3)
        version = jnp.where(c, version - 1, version)
    if tag == 32:
        # only the PN amplitude in this corner
        version = jnp.where((q > 2.5) & (chi1l < -0.9) & (chi2l < -0.9), 0, version)
        c = (q > 2.5) & (chi1l < -0.6) & (chi2l > 0.0) & (iv1 != 0.0)
        version = jnp.where(c, version - 1, version)
        iv1 = jnp.where(c, 0.0, iv1)
    if tag == 33:
        c = (1.2 > q) & (q > 1.0) & (chi1l < -0.1) & (chi2l > 0.0) & (iv1 != 0.0)
        version = jnp.where(c, version - 1, version)
        iv1 = jnp.where(c, 0.0, iv1)
    if tag in (21, 32):
        c = (version == 3) & wavy_points(
            colloc[0] * p1.m_seven_sixths, colloc[1] * p2.m_seven_sixths, colloc[2] * p3.m_seven_sixths
        )
        iv2 = jnp.where(c, 0.0, iv2)
        version = jnp.where(c, version - 1, version)
    # rename the remaining collocation points and frequencies
    p_orig = (p1, p2, p3)
    c = iv2 == 0.0
    iv2 = jnp.where(c, iv3, iv2)
    iv3 = jnp.where(c, 0.0, iv3)
    p2 = select(c, p3, p2)
    c = iv1 == 0.0
    iv1 = jnp.where(c, iv2, iv1)
    p1 = select(c, p2, p1)
    p_cut = select(c, p1, p_cut)
    iv2 = jnp.where(c, iv3, iv2)
    iv3 = jnp.where(c, 0.0, iv3)
    p2 = select(c, p3, p2)
    # PN only: the inspiral ends at the MECO frequency of the mode
    p_cut = select(version == 0, powers(hm.f_meco_lm), p_cut)
    f_amp_match_in = p_cut.itself
    amp.inspiral_amp_version = version
    amp.rho = _rho(version, (iv1, iv2, iv3), p_cut, (p1, p2, p3), p_orig)
    amp.fcut_insp_seven_thirds = p_cut.seven_thirds
    amp.fcut_insp_eight_thirds = p_cut.eight_thirds
    amp.fcut_insp_three = p_cut.three

    # the inspiral at its end and its derivative (by finite differences)
    f1 = f_amp_match_in
    exact = "exact_derivatives" in wf
    insp_f1, d1 = value_and_nd_derivative(f1, lambda x: inspiral_rescaled(powers(x), amp), exact)

    # ------------------------------ ringdown ------------------------------
    alambda, lambda_, sigma = fits.ringdown_fits(mode_int)
    amp.rd = [jnp.abs(alambda(wf)), lambda_(wf), sigma(wf), 1.0 / 12.0]
    if tag in (44, 33):
        # the ringdown at fAmpMatchIM is 0.9 times the inspiral at its end
        ringdown = rd_amp_ansatz(f_im, amp)
        rd0 = jnp.where(use_insp_rd, 0.9 * jnp.abs(insp_f1 / (ringdown / amp.rd[0])), amp.rd[0])
        amp.rd = [rd0] + amp.rd[1:]
    amp.use_insp_ansatz_ringdown = use_insp_rd

    # ---------------------------- intermediate ----------------------------
    int1, int2, int0, dint0 = fits.intermediate_fits(mode_int)
    f2 = freqs_inter[0]
    f3 = freqs_inter[1]
    f4 = f_im
    p4 = powers(f4)
    if hm.mixing_on:
        def rotated(x):
            re, im = rotate(x, amp.rd)
            return jnp.hypot(re, im)

        rd_f4, d4 = value_and_nd_derivative(f4, rotated, exact)
    else:
        rd_f4 = rd_amp_ansatz(f4, amp)
        d4 = rd_amp_dansatz(f4, amp)
    # derivatives of the inverse of the full amplitudes
    d1 = 7.0 / 6.0 * jnp.power(f1, 1.0 / 6.0) / insp_f1 - jnp.power(f1, 7.0 / 6.0) * d1 / (insp_f1 * insp_f1)
    d4 = 7.0 / 6.0 * jnp.power(f4, 1.0 / 6.0) / rd_f4 - jnp.power(f4, 7.0 / 6.0) * d4 / (rd_f4 * rd_f4)

    # by default a fifth order polynomial
    version = jnp.where(use_insp_rd, 101, 105)
    v1 = p_cut.m_seven_sixths * insp_f1
    v2 = jnp.where(use_insp_rd, 1.0, jnp.abs(int1(wf)))
    v3 = jnp.where(use_insp_rd, 1.0, jnp.abs(int2(wf)))
    v4 = p4.m_seven_sixths * rd_f4
    # the reconstruction is for the inverse of the amplitude
    v1 = 1.0 / v1
    v2 = 1.0 / v2
    v3 = 1.0 / v3
    v4 = 1.0 / v4

    # AmpEMR: a fourth order polynomial from fcutInsp to F0 (value and
    # derivative from fits) and the usual reconstruction from F0
    f0 = f1 + (f2 - f1) / 3.0
    v0 = int0(wf)
    d0 = dint0(wf)
    d0 = 7.0 / 6.0 / (v0 * f0) - d0 / (v0 * v0 * jnp.power(f0, 7.0 / 6.0))
    v0 = 1.0 / v0
    first = Struct(d1=d1, d4=d0, v1=v1, v2=v2, v3=v3, v4=v0, f1=f1, f2=f2, f3=f3, f4=f0)
    alpha = polynomial_coefficients(first, 104)[:5]

    # otherwise vetoes of the intermediate collocation points
    nv2, nv3, nf3, nversion = v2, v3, f3, version
    if tag == 21:
        on = hm.intermediate_amp_veto & (nversion == 105)
        threshold = 0.2 / amp_norm
        a = on & (1.0 / nv2 < threshold)
        ab = a & (1.0 / nv3 < threshold)
        b = on & ~a & (1.0 / nv3 < threshold)
        nv2 = jnp.where(a, 1.0, nv2)
        nv3 = jnp.where(ab | b, 1.0, nv3)
        nversion = jnp.where(ab, 1032, jnp.where(a | b, 1042, nversion))
    if tag in (21, 32):
        # IMRPhenomXHM_Ringdown_Amplitude_Veto
        threshold = (0.1 if tag == 32 else 0.01) / hm.amp_norm
        c = 1.0 / v4 < threshold
        nv2 = jnp.where(c, 1.0, nv2)
        nv3 = jnp.where(c, 1.0, nv3)
        nversion = jnp.where(c, 101 if tag == 32 else 1032, nversion)
    if tag == 21:
        c = (nversion == 105) & wavy_points(nv2, nv3, v4)
        nv3 = jnp.where(c, nv2, nv3)
        nf3 = jnp.where(c, f2, nf3)
        nv2 = jnp.where(c, 1.0, nv2)
        nversion = jnp.where(c, 1042, nversion)
    corner = (q > 40.0) & (chi1l > 0.9) & (nv2 != 1.0) & (nv3 != 1.0)
    if tag == 32:
        corner = corner | ((nversion != 101) & (((q > 2.5) & (chi1l < -0.6) & (chi2l > 0.0))
                                                | ((chi1l < -0.9) & (chi2l < -0.9))))
    if tag == 21:
        corner = corner | ((eta < 0.23) & (chi1l > 0.7) & (chi2l < -0.5))
    nv2 = jnp.where(corner, 1.0, nv2)
    nv3 = jnp.where(corner, 1.0, nv3)
    nversion = jnp.where(corner, 1032, nversion)

    emr = hm.amp_emr
    f1 = jnp.where(emr, f0, f1)
    v1 = jnp.where(emr, v0, v1)
    d1 = jnp.where(emr, d0, d1)
    v2 = jnp.where(emr, v2, nv2)
    v3 = jnp.where(emr, v3, nv3)
    f3 = jnp.where(emr, f3, nf3)
    version = jnp.where(emr, version, nversion)
    amp.f_amp_match_int12 = jnp.where(emr, f0, 0.0)
    amp.alpha = [jnp.where(emr, a, 1.0) for a in alpha] + [0.0]
    amp.amp_emr = emr

    # the reconstruction expects the point with a value at F3
    c = v3 == 1.0
    v3 = jnp.where(c, v2, v3)
    f3 = jnp.where(c, f2, f3)
    v2 = jnp.where(c, 1.0, v2)
    inputs = Struct(d1=d1, d4=d4, v1=v1, v2=v2, v3=v3, v4=v4, f1=f1, f2=f2, f3=f3, f4=f4)
    amp.intermediate_amp_version = version
    amp.f_amp_match_in = f_amp_match_in
    amp.f_amp_match_im = f_im
    amp.poly_inputs = inputs
    # what choose_polynomial needs (computed later for several modes at once
    # with defer)
    amp.choice = Struct(x=jnp.stack([jnp.asarray(getattr(inputs, name), jnp.float64) for name in _INPUTS]),
                        f3_105=freqs_inter[1], fstart=f_amp_match_in, fend=f_im, version=version)
    if not defer:
        amp = finish_amplitude(amp, choose_polynomial(amp.choice))
    return amp


def choose_polynomial(choice):
    """The intermediate polynomial of ``choice`` (``amp.choice`` of
    :func:`amplitude_coefficients`): ``(delta, order, error_poly_roots)``,
    LAL's ``ChoosePolOrder``. The same code for every mode (``jax.vmap``
    computes it for several at once)."""
    inputs = Struct(**dict(zip(_INPUTS, [choice.x[k] for k in range(len(_INPUTS))])))
    version = choice.version
    candidates = {
        # the fifth order needs four distinct frequencies (it is only used
        # without vetoes, when F3 is the original one)
        105: polynomial_coefficients(inputs.replace(f3=choice.f3_105), 105),
        104: polynomial_coefficients(inputs, 1042),
        103: polynomial_coefficients(inputs, 1032),
        102: polynomial_coefficients(inputs, 102),
        101: polynomial_coefficients(inputs, 101),
    }

    # ChoosePolOrder: lower the order until the polynomial does not cross
    # zero in [fAmpMatchIN, fAmpMatchIM]
    sg = lax.stop_gradient
    fstart, fend = sg(choice.fstart), sg(choice.fend)
    cz5, ok5 = roots.cross_zero_p5(sg(jnp.stack(candidates[105])), fstart, fend)
    cz4 = roots.cross_zero_p4(sg(jnp.stack(candidates[104])), fstart, fend)
    cz3 = roots.cross_zero_p3(sg(jnp.stack(candidates[103])), fstart, fend)
    cz2 = roots.cross_zero_p2(sg(jnp.stack(candidates[102])), fstart, fend)
    from_102 = jnp.where(cz2, 101, 102)
    from_103 = jnp.where(cz3, from_102, 103)
    from_104 = jnp.where(cz4, from_103, 104)
    from_105 = jnp.where(cz5, from_104, 105)
    order = jnp.where(
        version == 105, from_105,
        jnp.where(version == 1042, from_104, jnp.where(version == 1032, from_103, jnp.where(version == 102, from_102, 101))),
    )
    delta = [candidates[101][k] for k in range(6)]
    for o in (102, 103, 104, 105):
        delta = [jnp.where(order == o, candidates[o][k], delta[k]) for k in range(6)]
    return jnp.stack(delta), order, (version == 105) & ~ok5


def finish_amplitude(amp, chosen):
    """``amp`` with the polynomial chosen by :func:`choose_polynomial`."""
    delta, order, error = chosen
    amp = amp.replace(delta=[delta[k] for k in range(6)], inter_amp_pol_order=order, error_poly_roots=error)
    del amp._fields["choice"]
    return amp


def _polynomial(f, a):
    """``a0 + a1 f + ... + a5 f^5`` as LAL's fifth order (lower orders have
    zero leading coefficients, which leaves the value unchanged)."""
    two = f * f
    three = two * f
    four = two * two
    five = three * two
    result = fma(a[1], f, a[0])
    result = fma(a[2], two, result)
    result = fma(a[3], three, result)
    result = fma(a[4], four, result)
    return fma(a[5], five, result)


def amplitude(mf, p, amp, hm, rotation=None):
    """The amplitude of the mode for the 122019 release
    (``IMRPhenomXHM_Amplitude_noModeMixing`` / ``_ModeMixing``), strain
    without ``amp0``: the inspiral below ``fAmpMatchIN``, the ringdown from
    ``fAmpMatchIM`` (``rotation = (re, im)``, the spherical (3,2) ringdown,
    for the mode with mixing) and the inverse of the intermediate polynomial
    in between (two polynomials with ``AmpEMR``). Negative amplitudes are
    kept, as in LAL."""
    ins = inspiral_ansatz(p, amp)
    if hm.mixing_on:
        rd = jnp.hypot(rotation[0], rotation[1])
    else:
        rd = rd_amp_ansatz(mf, amp)
    rd = rd * (p.m_seven_sixths * amp.amp_norm)
    # the polynomial on its own region only (it may vanish outside)
    f_int = jnp.clip(mf, amp.f_amp_match_in, amp.f_amp_match_im)
    first = amp.amp_emr & (mf < amp.f_amp_match_int12)
    coefficients = [jnp.where(first, a, d) for a, d in zip(amp.alpha, amp.delta)]
    int_ = amp.amp_norm / _polynomial(f_int, coefficients)
    return jnp.where(mf < amp.f_amp_match_in, ins, jnp.where(mf >= amp.f_amp_match_im, rd, int_))
