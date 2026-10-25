# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The parameter-space fits and the model-independent helpers of the PNR
angles of IMRPhenomXPNR against LALSimulation 6.2.0: the PhenomXCP
deviations through SWIG, everything else through the internal functions
that ``liblalsimulation`` exports (ctypes).

Functions taking LAL's parameter structs get them as arrays of doubles
(``IMRPhenomX_PNR_alpha_parameters`` and ``IMRPhenomX_PNR_beta_parameters``
are plain doubles). The few that read a field of the waveform or precession
structs get buffers filled with the value (or with the value at the probed
offset of the field), which needs no struct layout.
"""

import ctypes
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformx.models.xpnr import pnr_angles as pa
from waveformx.models.xpnr._generated import pnr_fits as pf

D = ctypes.c_double
BETA_FIELDS = ["b0", "b1", "b2", "b3", "b4", "b5", "mf_beta_lower", "mf_beta_upper", "beta_lower",
               "beta_upper", "derivative_beta_lower", "derivative_beta_upper", "beta_rescale_1",
               "beta_rescale_2"]
ALPHA_FIELDS = ["a1", "a2", "a3", "a4", "mf_alpha_lower", "mf_alpha_upper", "alpha_lower", "alpha_upper",
                "derivative_alpha_lower", "derivative_alpha_upper"]
BetaStruct = D * 18
AlphaStruct = D * 15
# doubles in the buffers standing in for the waveform and precession structs
NSLOT = 1 << 17

@pytest.fixture(scope="module")
def lib(lal):
    """liblalsimulation, for its exported internal functions."""
    for name in ("liblalsimulation.dylib", "liblalsimulation.so"):
        path = Path(sys.prefix) / "lib" / name
        if path.is_file():
            return ctypes.CDLL(str(path))
    pytest.skip("liblalsimulation not found")


def cfunc(lib, name, argtypes, restype=D):
    function = getattr(lib, name)
    function.argtypes = argtypes
    function.restype = restype
    return function


def doubles(lib, name, nargs):
    """A LAL function of doubles, vectorised with numpy."""
    return np.vectorize(cfunc(lib, name, [D] * nargs))


def beta_struct(values):
    return BetaStruct(*([float(values.get(k, 0.0)) for k in BETA_FIELDS] + [0.0] * 4))


def alpha_struct(values, interp):
    return AlphaStruct(*([float(values.get(k, 0.0)) for k in ALPHA_FIELDS] + [float(x) for x in interp] + [0.0]))


def beta_coefficients(eta, chi, costheta, costhetaf):
    """``IMRPhenomX_PNR_precompute_beta_coefficients`` (versions 223 and
    102) from the fits."""
    b4 = pf.beta_b4(eta, chi, costheta)
    b2 = pf.beta_b2(eta, chi, costheta)
    return dict(
        b0=jnp.arccos(costhetaf) - pf.beta_b0(eta, chi, costheta),
        b1=pf.beta_b1(eta, chi, costheta),
        b2=b2,
        b3=b2 * pf.beta_b3(eta, chi, costheta),
        b4=jnp.where(b4 <= 175.0, 175.0, b4),
        b5=pf.beta_b5(eta, chi, costheta),
    )


def realistic_beta(n=1500, seed=2):
    """MR beta coefficients over the parameter space of the fits."""
    rng = np.random.default_rng(seed)
    eta, chi, ct, ctf = (rng.uniform(0.04, 0.25, n), rng.uniform(0.0, 1.0, n), rng.uniform(-1.0, 1.0, n),
                         rng.uniform(-0.2, 1.0, n))
    return {k: np.asarray(v) for k, v in beta_coefficients(eta, chi, ct, ctf).items()}


def lal_connection(lib, values):
    """LAL's inflection point, its three roots and the connection
    frequencies, one binary at a time."""
    single = cfunc(lib, "IMRPhenomX_PNR_single_inflection_point", [ctypes.POINTER(BetaStruct)])
    three = cfunc(lib, "IMRPhenomX_PNR_three_inflection_points", [ctypes.POINTER(BetaStruct)],
                  ctypes.POINTER(D))
    connection = cfunc(lib, "IMRPhenomX_PNR_BetaConnectionFrequencies", [ctypes.POINTER(BetaStruct)],
                       ctypes.c_int)
    n = len(values["b1"])
    inflection, lower, upper, roots = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 3), complex)
    for i in range(n):
        p = beta_struct({k: v[i] for k, v in values.items()})
        inflection[i] = single(ctypes.byref(p))
        r = three(ctypes.byref(p))
        roots[i] = [complex(r[2 * j], r[2 * j + 1]) for j in range(3)]
        assert connection(ctypes.byref(p)) == 0
        lower[i], upper[i] = p[6], p[7]
    return inflection, roots, lower, upper


@jax.jit
def connection(p):
    """Our inflection point, the real parts of the three roots of the
    inflection cubic and the connection frequencies."""
    out = pa.beta_connection_frequencies(p)
    roots = jnp.stack([r[0] for r in pa._three_roots(*pa.inflection_cubic(p))], -1)
    return (pa._single_inflection_point(p, pa.BETA_SOFT_WIDTH), roots, out.mf_beta_lower, out.mf_beta_upper,
            out.beta_freeze)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mu1", "mu2", "mu3", "nu4", "nu5", "nu6", "zeta1", "zeta2"])
def test_xcp_deviations(lal, name):
    """``XLALSimIMRPhenomXCP_*_l2m2(theta_LS, eta, a1)`` over a grid."""
    _, lalsim = lal
    theta, eta, a1 = np.meshgrid(np.linspace(0.0, np.pi, 9), np.linspace(0.02, 0.25, 7), np.linspace(0.0, 1.0, 6))
    theta, eta, a1 = theta.ravel(), eta.ravel(), a1.ravel()
    reference = np.array([getattr(lalsim, f"SimIMRPhenomXCP_{name.upper()}_l2m2")(*x) for x in zip(theta, eta, a1)])
    value = jax.jit(getattr(pf, f"{name}_l2m2"))(theta, eta, a1)
    error, _ = compare(value, reference)
    assert error < 1e-13, error


FITS = [("alpha_a1", "alpha_A1"), ("alpha_a2", "alpha_A2"), ("alpha_a3", "alpha_A3"), ("alpha_a4", "alpha_A4"),
        ("beta_b0", "beta_B0"), ("beta_b1", "beta_B1"), ("beta_b2", "beta_B2"), ("beta_b3", "beta_B3"),
        ("beta_b4", "beta_B4"), ("beta_b5", "beta_B5"), ("beta_bf", "beta_Bf")]


@pytest.mark.parametrize("name, lal_name", FITS)
def test_angle_coefficient_fits(lib, name, lal_name):
    """``IMRPhenomX_PNR_*_coefficient(eta, chi, costheta)``: LAL sums the
    80 terms with fused multiply-adds and ``pow``; the sums cancel near the
    zeros of the polynomials, where the pointwise relative error grows
    (~1e-10), so the error is measured against the largest value."""
    eta, chi, ct = np.meshgrid(np.linspace(0.02, 0.25, 12), np.linspace(0.0, 1.0, 11), np.linspace(-1.0, 1.0, 13))
    eta, chi, ct = eta.ravel(), chi.ravel(), ct.ravel()
    reference = doubles(lib, f"IMRPhenomX_PNR_{lal_name}_coefficient", 3)(eta, chi, ct)
    value = jax.jit(getattr(pf, name))(eta, chi, ct)
    error, _ = compare(value, reference)
    assert error < 1e-12, error


def test_evaluate_coefficient_array(lib):
    """``IMRPhenomX_PNR_evaluate_coefficient_array`` with a table of its own."""
    table = (np.arange(80.0).reshape(4, 4, 5) - 40.0) / 7.0
    table[1, 2, 3] = 0.0
    function = cfunc(lib, "IMRPhenomX_PNR_evaluate_coefficient_array", [ctypes.POINTER(D)] + [D] * 3)
    flat = np.ascontiguousarray(table)
    for eta, chi, ct in [(0.1, 0.3, -0.4), (0.25, 0.9, 0.99), (0.05, 0.0, 0.0)]:
        reference = function(flat.ctypes.data_as(ctypes.POINTER(D)), eta, chi, ct)
        value = pf.evaluate_coefficient_array(table.tolist(), eta, chi, ct)
        assert float(value) == pytest.approx(reference, rel=1e-13)


# ---------------------------------------------------------------------------
# alpha
# ---------------------------------------------------------------------------


def connection_arguments(n=500, seed=3):
    rng = np.random.default_rng(seed)
    mf1 = rng.uniform(0.005, 0.1, n)
    mf2 = mf1 + rng.uniform(0.001, 0.1, n)
    return (mf1, mf2, *rng.normal(size=(4, n)))


@pytest.mark.parametrize("k", range(4))
def test_alpha_interpolation(lib, k):
    args = connection_arguments()
    reference = doubles(lib, f"IMRPhenomX_PNR_alpha_interpolation_{k}", 6)(*args)
    value = jax.jit(getattr(pa, f"alpha_interpolation_{k}"))(*args)
    error, _ = compare(value, reference)
    assert error < 1e-14, error
    # the Mf = 100 sentinel: LAL divides by zero (+-inf, or NaN which its
    # caller replaces by 0; the coefficients are not used then)
    lal_value = doubles(lib, f"IMRPhenomX_PNR_alpha_interpolation_{k}", 6)(100.0, 100.0, 0.3, 0.3, 1.0, -2.0)
    assert not np.isfinite(lal_value)
    assert float(getattr(pa, f"alpha_interpolation_{k}")(100.0, 100.0, 0.3, 0.3, 1.0, -2.0)) == 0.0


def test_mr_and_intermediate_alpha(lib):
    mr = cfunc(lib, "IMRPhenomX_PNR_MR_alpha_expression", [D, ctypes.POINTER(AlphaStruct)])
    intermediate = cfunc(lib, "IMRPhenomX_PNR_intermediate_alpha_expression", [D, ctypes.POINTER(AlphaStruct)])
    rng = np.random.default_rng(4)
    mfs = np.geomspace(1e-3, 0.3, 200)
    for eta, chi, ct in [(0.2, 0.5, 0.3), (0.1, 0.8, -0.6), (0.24, 0.2, 0.9), (0.06, 0.95, 0.0)]:
        values = dict(a1=abs(float(pf.alpha_a1(eta, chi, ct))), a2=float(pf.alpha_a2(eta, chi, ct)),
                      a3=abs(float(pf.alpha_a3(eta, chi, ct))), a4=float(pf.alpha_a4(eta, chi, ct)))
        interp = rng.normal(size=4)
        struct = alpha_struct(values, interp)
        p = pa.alpha_params(**values, alpha_interp=tuple(interp))
        reference = np.array([mr(mf, ctypes.byref(struct)) for mf in mfs])
        error, _ = compare(jax.jit(pa.mr_alpha)(mfs, p), reference)
        assert error < 1e-14, error
        reference = np.array([intermediate(mf, ctypes.byref(struct)) for mf in mfs])
        error, _ = compare(jax.jit(pa.intermediate_alpha)(mfs, p), reference)
        assert error < 1e-14, error


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def test_mr_beta_expressions(lib):
    """The MR ``beta``, its derivatives and the rescaling of the inspiral
    ``beta`` for coefficients from the fits."""
    values = realistic_beta(n=40, seed=5)
    rng = np.random.default_rng(5)
    values["beta_rescale_1"] = rng.normal(size=40)
    values["beta_rescale_2"] = rng.normal(size=40)
    mfs = np.geomspace(1e-3, 0.3, 60)
    for name, lal_name in [("mr_beta", "MR_beta"), ("mr_dbeta", "MR_dbeta"), ("mr_ddbeta", "MR_ddbeta"),
                           ("mr_dddbeta", "MR_dddbeta"), ("rescale_beta", "rescale_beta")]:
        function = cfunc(lib, f"IMRPhenomX_PNR_{lal_name}_expression", [D, ctypes.POINTER(BetaStruct)])
        ours = jax.jit(getattr(pa, name))(mfs[None, :], pa.beta_params(**{k: v[:, None] for k, v in values.items()}))
        for i in range(40):
            struct = beta_struct({k: v[i] for k, v in values.items()})
            reference = np.array([function(mf, ctypes.byref(struct)) for mf in mfs])
            error, _ = compare(ours[i], reference)
            # (LAL's fused multiply-adds, which XLA contracts differently: the
            # third derivative cancels most)
            assert error < 1e-10, (name, i, error)


@pytest.mark.parametrize("name, lal_name, nargs", [
    ("beta_rescaling_1", "IMRPhenomX_PNR_beta_rescaling_1", 5),
    ("beta_rescaling_2", "IMRPhenomX_PNR_beta_rescaling_2", 5),
    ("st_beta_coeffs_0", "IMRPhenomX_ST_PNR_beta_coeffs_0", 6),
    ("st_beta_coeffs_1", "IMRPhenomX_ST_PNR_beta_coeffs_1", 6),
    ("st_beta_coeffs_2", "IMRPhenomX_ST_PNR_beta_coeffs_2", 6),
    ("st_beta_coeffs_3", "IMRPhenomX_ST_PNR_beta_coeffs_3", 6),
])
def test_beta_connection_coefficients(lib, name, lal_name, nargs):
    args = connection_arguments(seed=6)
    args = args if nargs == 6 else args[:1] + args[2:]
    reference = doubles(lib, lal_name, nargs)(*args)
    value = jax.jit(getattr(pa, name))(*args)
    error, _ = compare(value, reference)
    assert error < 1e-14, error


def test_beta_connection_cubic():
    """``beta_connection`` (static in LAL): the cubic of ``beta_interp``."""
    p = pa.beta_params(beta_interp=(0.3, -2.0, 15.0, -40.0))
    mf = np.linspace(0.0, 0.1, 11)
    expected = 0.3 - 2.0 * mf + 15.0 * mf ** 2 - 40.0 * mf ** 3
    assert np.allclose(pa.beta_connection(mf, p), expected, rtol=1e-14, atol=1e-16)


def test_arctan_window(lib):
    rng = np.random.default_rng(7)
    beta = np.concatenate([rng.uniform(-0.3, 0.0101, 300), rng.uniform(0.0, np.pi, 300),
                           rng.uniform(np.pi - 0.0101, np.pi + 0.3, 300), [0.01, np.pi - 0.01, 0.0, np.pi]])
    reference = doubles(lib, "IMRPhenomX_PNR_arctan_window", 1)(beta)
    value = jax.jit(pa.arctan_window)(beta)
    # (two ulps at pi: the power is formed as a ratio, which stays within
    # the range of float32, rather than as LAL's quotient of 500th powers)
    assert np.max(np.abs(np.asarray(value) - reference)) <= 9e-16


def test_attach_mr_beta(lib):
    attach = cfunc(lib, "IMRPhenomX_PNR_AttachMRBeta", [ctypes.POINTER(BetaStruct)], ctypes.c_uint32)
    cases = [dict(b0=0.5, mf_beta_lower=0.02, beta_upper=0.6),  # attached
             dict(b0=0.5, mf_beta_lower=0.008, beta_upper=0.6),  # too low
             dict(b0=0.5, mf_beta_lower=100.0, beta_upper=0.6),  # no connection
             dict(b0=0.5, mf_beta_lower=0.02, beta_upper=0.0),  # negative beta
             dict(b0=0.01, mf_beta_lower=0.02, beta_upper=0.6),  # beta_upper > 5 (B0 + 0.1)
             dict(b0=0.02, mf_beta_lower=0.009, beta_upper=0.6)]
    for case in cases:
        reference = attach(ctypes.byref(beta_struct(case)))
        assert bool(pa.attach_mr_beta(pa.beta_params(**case))) == bool(reference), case
    batch = pa.beta_params(**{k: jnp.array([c.get(k, 0.0) for c in cases])
                              for k in ["b0", "mf_beta_lower", "beta_upper"]})
    assert jax.jit(pa.attach_mr_beta)(batch).tolist() == [True, False, False, False, False, True]


def test_inflection_and_connection_frequencies(lib):
    """``IMRPhenomX_PNR_single_inflection_point``,
    ``IMRPhenomX_PNR_three_inflection_points`` and
    ``IMRPhenomX_PNR_BetaConnectionFrequencies`` over the parameter space of
    the fits.

    The inflection points come from ``acos`` of values near 1 and from a
    cancellation of terms up to ~1e3 times larger than the roots: they
    agree to ~1e-10 in absolute terms, except where the trigonometric
    formula loses the roots (a small cubic term), where the roots are
    bracketed instead and LAL's choice jumps. LAL's choices between
    candidate frequencies are blended across narrow bands, and beta is held
    from its minimum with a weight (``beta_freeze``) rather than not at
    all: away from those bands, the frequencies are LAL's."""
    values = realistic_beta()
    inflection, roots, lower, upper = lal_connection(lib, values)
    ours = connection(pa.beta_params(**values))
    our_inflection, our_roots, our_lower, our_upper, freeze = (np.asarray(x) for x in ours)
    assert np.all(np.isfinite(inflection))
    assert np.mean(np.abs(our_inflection - inflection) < 1e-9) > 0.95
    assert np.max(np.abs(our_roots - roots.real) / np.maximum(1.0, np.abs(roots.real))) < 1e-9
    # the lower frequency: LAL's sentinel where ours is (LAL's also where
    # its maximum lies far away, which ours blends), its value to rounding
    # except in the bands and the blend (87% of these binaries agree)
    assert np.all(lower[our_lower == 100.0] == 100.0)
    close = np.abs(our_lower - lower) / lower < 1e-9
    assert close.mean() > 0.8, close.mean()
    # the upper frequency: LAL's where beta is held fully, LAL's 100 where
    # it is not held at all
    held = freeze >= 1.0
    assert np.any(held) and np.any(freeze <= 0.0)
    assert np.all(np.abs(our_upper[held] - upper[held]) / upper[held] < 1e-12)
    assert np.all(upper[freeze <= 0.0] == 100.0)


# special cases: the cubic term of the inflection cubic vanishes (b2 = 2 b3 b5
# exactly); a = b = 0 (then c = 0 too: LAL gives -d / c = NaN); an upper
# connection frequency at infinity (100); a negative lower one (no MR beta).
# (The inflection cubic of the MR beta has three real roots: LAL's NaN from
# complex roots of the trigonometric formula only arises from rounding at
# double roots, see test_three_roots_complex.)
SPECIAL = {
    "two_roots": dict(b0=0.5, b1=2e-3, b2=-3e-4, b3=3e-4, b4=400.0, b5=-0.5),
    "two_roots_b": dict(b0=0.7202664308540649, b1=0.0003291959672871742, b2=0.00025630891981592255,
                        b3=-0.00012815445990796128, b4=4096.0, b5=-1.0),
    "linear": dict(b0=0.5, b1=65.0 / 256.0, b2=1.0, b3=1.0, b4=256.0, b5=0.5),
    "asymptote": dict(b0=0.7630824918673582, b1=1.0491999669117931e-07, b2=0.11519418378015227,
                      b3=-0.28204504840930245, b4=406.4400930158092, b5=0.007748859966057177),
    "negative_lower": dict(b0=0.18347764898885877, b1=-2.070152326130502e-06, b2=-0.00015865898280466014,
                           b3=-2.3431798840978237e-05, b4=2394.7953798522994, b5=0.0008978495694061265),
}


@pytest.mark.parametrize("name", SPECIAL)
def test_inflection_special_cases(lib, name):
    """Finite where LAL's are NaN (``a = b = 0``); LAL's inflection point
    elsewhere. Where the maximum of beta lies at infinity (a vanishing cubic
    term), the lower connection frequency is the blend of LAL's turnover
    rather than LAL's 100 (no connection)."""
    values = {k: np.array([v]) for k, v in SPECIAL[name].items()}
    inflection, _, lower, upper = lal_connection(lib, values)
    our_inflection, _, our_lower, our_upper, freeze = (np.asarray(x)[0] for x in connection(pa.beta_params(**values)))
    assert np.isfinite(our_inflection) and np.isfinite(our_lower) and np.isfinite(our_upper)
    if name == "linear":
        assert np.isnan(inflection[0]) and lower[0] == upper[0] == 100.0
    else:
        np.testing.assert_allclose(our_inflection, inflection, rtol=1e-12, atol=0)
    if name in ("asymptote", "negative_lower"):
        np.testing.assert_allclose(our_lower, lower, rtol=1e-11)
    if name == "two_roots":
        two = cfunc(lib, "IMRPhenomX_PNR_two_inflection_points", [ctypes.POINTER(BetaStruct)], ctypes.POINTER(D))
        r = two(ctypes.byref(beta_struct(SPECIAL[name])))
        reference = np.array([complex(r[0], r[1]), complex(r[2], r[3])])
        ours = pa._two_roots(*pa.inflection_cubic(pa.beta_params(**SPECIAL[name]))[1:])
        np.testing.assert_allclose(np.array([complex(float(x), float(y)) for x, y in ours]), reference, rtol=1e-13)
    elif name == "negative_lower":
        assert lower[0] == our_lower == 100.0


def test_three_roots_complex():
    """The trigonometric formula for a cubic with complex roots (``x^3 - 3x
    + 4``: ``acos(-2)``, where LAL's are NaN): finite, with finite
    gradients; for three real roots, the roots."""

    def roots(a, b, c, d):
        return tuple(v[0] for v in pa._three_roots(a, b, c, d))

    assert all(np.isfinite(float(v)) for v in roots(1.0, 0.0, -3.0, 4.0))
    grads = jax.grad(lambda *x: sum(roots(*x)), argnums=(0, 1, 2, 3))(1.0, 0.0, -3.0, 4.0)
    assert np.all(np.isfinite(grads))
    expected = sorted(np.roots([1.0, 0.0, -3.0, 1.0]).real)
    np.testing.assert_allclose(sorted(float(v) for v in roots(1.0, 0.0, -3.0, 1.0)), expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# inspiral beta
# ---------------------------------------------------------------------------


def test_chi_calc(lib):
    rng = np.random.default_rng(8)
    args = (rng.uniform(0.5, 0.95, 300), *rng.uniform(0.1, 2.0, (3, 300)), rng.normal(size=300),
            rng.uniform(0.0, np.pi, 300))
    reference = doubles(lib, "IMRPhenomX_PNR_chi_calc", 6)(*args)
    error, _ = compare(jax.jit(pa.chi_calc)(*args), reference)
    assert error < 1e-15, error


def test_pn_waveform_beta(lib):
    rng = np.random.default_rng(9)
    m1 = rng.uniform(0.5, 0.95, 500)
    args = (rng.uniform(1e-4, 0.1, 500), rng.uniform(0.0, 0.9 * np.pi, 500), m1, 1.0 - m1, rng.uniform(0.0, 1.0, 500),
            rng.uniform(-1.0, 1.0, 500))
    reference = doubles(lib, "IMRPhenomX_PNR_PNWaveformBeta", 6)(*args)
    error, _ = compare(jax.jit(pa.pn_waveform_beta)(*args), reference)
    assert error < 1e-14, error
    # the tolerance of atan2: no precession
    assert float(pa.pn_waveform_beta(0.01, 0.0, 0.6, 0.4, 0.0, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# windows, frequency maps, gamma
# ---------------------------------------------------------------------------


def test_windows(lib):
    """``IMRPhenomX_PNR_AnglesWindow(pWF, pPrec)`` and
    ``IMRPhenomX_PNR_CoprecWindow(pWF)`` read ``pWF->q`` and
    ``pPrec->chi_singleSpin``: buffers filled with the values stand in for
    the structs."""
    angles = cfunc(lib, "IMRPhenomX_PNR_AnglesWindow", [ctypes.c_void_p, ctypes.c_void_p])
    coprec = cfunc(lib, "IMRPhenomX_PNR_CoprecWindow", [ctypes.c_void_p])
    qs = np.array([1.0, 8.5, 8.6, 10.0, 11.9, 12.0, 15.0, 19.9, 20.0, 25.0])
    chis = np.array([0.1, 0.85, 0.9, 1.1, 1.2, 1.5])
    for q in qs:
        wf = np.full(NSLOT, q)
        assert float(pa.coprec_window(q)) == pytest.approx(coprec(wf.ctypes.data), rel=1e-15, abs=1e-16)
        for chi in chis:
            prec = np.full(NSLOT, chi)
            reference = angles(wf.ctypes.data, prec.ctypes.data)
            assert float(pa.angles_window(q, chi)) == pytest.approx(reference, rel=1e-15, abs=1e-16)
    assert np.array_equal(jax.jit(pa.angles_window)(qs[:, None], chis[None]),
                          np.vectorize(lambda q, c: float(pa.angles_window(q, c)))(qs[:, None], chis[None]))


@pytest.mark.parametrize("ell, emm", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)])
def test_linear_frequency_map(lib, ell, emm):
    function = cfunc(lib, "IMRPhenomX_PNR_LinearFrequencyMap", [D] * 7 + [ctypes.c_uint32])
    slope = doubles(lib, "IMRPhenomX_PNR_LinearFrequencySlope", 5)
    rng = np.random.default_rng(10)
    n = 400
    lower = rng.uniform(0.005, 0.05, n)
    upper = lower + rng.uniform(0.0, 0.1, n)
    rd22 = rng.uniform(0.05, 0.12, n)
    rdlm = rd22 * rng.uniform(0.5, 2.0, n)
    mf = rng.uniform(0.0, 0.25, n)
    for inspiral in (False, True):
        reference = np.array([function(*x, float(ell), float(emm), *y, int(inspiral))
                              for x, y in zip(mf[:, None], np.stack([lower, upper, rd22, rdlm], 1))])
        value, error = jax.jit(pa.linear_frequency_map, static_argnums=(1, 2))(mf, ell, emm, lower, upper, rd22, rdlm,
                                                                               inspiral)
        assert not np.any(error)
        assert compare(value, reference)[0] < 1e-15
        # traced mode indices and flag
        value_traced, error = jax.jit(pa.linear_frequency_map)(mf, ell, emm, lower, upper, rd22, rdlm, inspiral)
        assert not np.any(error)
        assert compare(value_traced, reference)[0] < 1e-15
    reference = slope(float(emm), lower, upper, rd22, rdlm)
    error, _ = compare(pa.linear_frequency_slope(emm, lower, upper, rd22, rdlm), reference)
    assert error < 1e-14


def test_linear_frequency_map_failure(lib, lal):
    """LAL fails (returns XLAL_FAILURE) for ``Mf_lower > Mf_upper`` when the
    transition is needed; here the value is NaN and the error flag set."""
    lal_module, _ = lal
    function = cfunc(lib, "IMRPhenomX_PNR_LinearFrequencyMap", [D] * 7 + [ctypes.c_uint32])
    args = (0.05, 2.0, 1.0, 0.04, 0.03, 0.08, 0.06)
    assert function(*args, 0) == -1.0
    if hasattr(lal_module, "ClobberErrno"):
        lal_module.ClobberErrno()
    value, error = pa.linear_frequency_map(*args, False)
    assert bool(error) and np.isnan(value)
    # not needed below Mf_lower, for the (2,2) mode or with the inspiral scaling
    for mf, ell, emm, inspiral in [(0.035, 2, 1, False), (0.05, 2, 2, False), (0.05, 2, 1, True)]:
        value, error = pa.linear_frequency_map(mf, ell, emm, 0.04, 0.03, 0.08, 0.06, inspiral)
        assert not bool(error)
        assert float(value) == function(mf, float(ell), float(emm), 0.04, 0.03, 0.08, 0.06, int(inspiral))


def test_linear_frequency_map_transition_frequencies(lib):
    """``IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies`` reads
    ``pPrec->PNR_HM_Mflow`` and ``pPrec->PNR_HM_Mfhigh``, whose offsets are
    probed with a buffer of distinct values."""
    function = cfunc(lib, "IMRPhenomX_PNR_LinearFrequencyMapTransitionFrequencies",
                     [ctypes.POINTER(D), ctypes.POINTER(D), D, D, D, ctypes.c_void_p], ctypes.c_int)
    low, high = D(), D()
    probe = np.arange(1.0, NSLOT + 1.0)
    assert function(ctypes.byref(low), ctypes.byref(high), 2.0, 0.1, 0.1, probe.ctypes.data) == 0
    slot_low, slot_high = round(low.value / 0.65) - 1, round(high.value / 1.1) - 1
    assert probe[slot_low] * 0.65 == low.value and probe[slot_high] * 1.1 == high.value
    cases = [(2, 0.08, 0.09, 0.03, 0.06),  # no fallback
             (1, 0.08, 0.05, 0.03, 0.06),  # (2,1)-like, Mf_high above half
             (1, 0.08, 0.03, 0.03, 0.04),  # Mf_high below half of PNR_HM_Mfhigh
             (1, 0.08, 0.001, 0.03, 0.06),  # Mf_high negative
             (3, 0.08, 0.12, 0.03, 0.06), (4, 0.08, 0.15, 0.025, 0.07)]
    for emm, rd22, rdlm, mflow, mfhigh in cases:
        prec = np.zeros(NSLOT)
        prec[slot_low], prec[slot_high] = mflow, mfhigh
        assert function(ctypes.byref(low), ctypes.byref(high), float(emm), rd22, rdlm, prec.ctypes.data) == 0
        ours = pa.linear_frequency_map_transition_frequencies(emm, rd22, rdlm, mflow, mfhigh)
        assert (float(ours[0]), float(ours[1])) == (low.value, high.value)


def test_small_helpers():
    x = np.linspace(-1.0, 1.0, 11)
    np.testing.assert_allclose(pa.sqrt_one_minus_square(x), np.sqrt(1.0 - x * x), rtol=1e-15, atol=1e-16)
    p = pa.beta_params(b1=1.0)
    assert p.beta_interp == (0.0, 0.0, 0.0, 0.0) and p.b1 == 1.0
    with pytest.raises(TypeError):
        pa.alpha_params(b1=1.0)


# ---------------------------------------------------------------------------
# jit, vmap, grad
# ---------------------------------------------------------------------------


def test_jit_vmap_grad():
    """Batches of binaries through the fits and the connection of ``beta``,
    with finite gradients everywhere (sentinels and NaN inflection points
    included) that agree with finite differences where the selected branch
    is smooth."""

    def frequencies(x):
        eta, chi, ct, ctf = x
        p = beta_connection_frequencies_from_fits(eta, chi, ct, ctf)
        return jnp.stack([p.mf_beta_lower, p.mf_beta_upper, pa._single_inflection_point(p, pa.BETA_SOFT_WIDTH)])

    def beta_connection_frequencies_from_fits(eta, chi, ct, ctf):
        return pa.beta_connection_frequencies(pa.beta_params(**beta_coefficients(eta, chi, ct, ctf)))

    rng = np.random.default_rng(12)
    x = jnp.stack([rng.uniform(0.04, 0.25, 300), rng.uniform(0.05, 1.0, 300), rng.uniform(-0.99, 0.99, 300),
                   rng.uniform(-0.2, 0.99, 300)], 1)
    values = jax.jit(jax.vmap(frequencies))(x)
    jacobians = jax.jit(jax.vmap(jax.jacfwd(frequencies)))(x)
    gradients = jax.jit(jax.vmap(jax.grad(lambda x: frequencies(x).sum())))(x)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jacobians)) and np.all(np.isfinite(gradients))
    np.testing.assert_allclose(gradients, jacobians.sum(1), rtol=1e-8, atol=1e-12)
    # the sentinel, where it is reached, has no gradient
    assert np.all(jacobians[np.asarray(values[:, 0]) == 100.0, 0] == 0.0)

    # finite differences
    f = jax.jit(frequencies)
    for x0 in [jnp.array([0.2, 0.5, 0.3, 0.8]), jnp.array([0.1, 0.7, -0.4, 0.6]), jnp.array([0.16, 0.3, 0.8, 0.95])]:
        jacobian = np.asarray(jax.jacfwd(frequencies)(x0))
        for j in range(4):
            h = 1e-6 * np.eye(4)[j]
            fd = (np.asarray(f(x0 + h)) - np.asarray(f(x0 - h))) / 2e-6
            np.testing.assert_allclose(jacobian[:, j], fd, rtol=1e-4, atol=1e-7)

    # special cases: finite gradients through the stand-ins (but for an
    # exactly vanishing cubic term, b2 = 2 b3 b5, where the blend of the
    # turnover far from the maximum has NaN derivatives, as at the tag
    # lal-faithful)
    finite_cases = [case for name, case in SPECIAL.items() if not name.startswith("two_roots")]
    special = pa.beta_params(**{k: jnp.array([case[k] for case in finite_cases]) for k in SPECIAL["linear"]})
    grads = jax.jit(jax.vmap(jax.grad(lambda p: pa.beta_connection_frequencies(p).mf_beta_lower
                                      + pa._single_inflection_point(p, pa.BETA_SOFT_WIDTH))))(special)
    assert all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(grads))

    # the fits and the other helpers
    eta, chi, ct = x[:, 0], x[:, 1], x[:, 2]
    for name, _ in FITS:
        g = jax.jit(jax.vmap(jax.grad(getattr(pf, name), argnums=(0, 1, 2))))(eta, chi, ct)
        assert all(np.all(np.isfinite(v)) for v in g), name
    for name in ["mu1", "zeta2"]:
        g = jax.vmap(jax.grad(getattr(pf, f"{name}_l2m2"), argnums=(0, 1, 2)))(jnp.arccos(ct), eta, chi)
        assert all(np.all(np.isfinite(v)) for v in g)
    beta = jnp.array([-0.1, 0.0, 0.005, 0.01, 0.5, 1.5707963267948966, 3.0, np.pi - 0.005, np.pi + 0.1])
    g = jax.vmap(jax.grad(pa.arctan_window))(beta)
    assert np.all(np.isfinite(g)) and np.all(np.asarray(g)[4:7] == 1.0)
    g = jax.vmap(jax.grad(pa.pn_waveform_beta, argnums=(0, 1, 2, 3, 4, 5)))(
        jnp.full(9, 0.02), beta[4:].repeat(2)[:9] * 0.5, jnp.full(9, 0.6), jnp.full(9, 0.4), jnp.full(9, 0.5),
        jnp.linspace(-0.9, 0.9, 9))
    assert all(np.all(np.isfinite(v)) for v in g)
    mf = jnp.linspace(0.0, 0.2, 21)
    for lower, upper in [(0.04, 0.08), (0.04, 0.04), (0.08, 0.04)]:
        g = jax.vmap(jax.grad(lambda m, lo, hi: jnp.nan_to_num(pa.linear_frequency_map(m, 3, 3, lo, hi, 0.09, 0.14,
                                                                                       False)[0]),
                              argnums=(0, 1, 2)), in_axes=(0, None, None))(mf, lower, upper)
        assert all(np.all(np.isfinite(v)) for v in g)
    q = jnp.linspace(1.0, 25.0, 49)
    assert np.all(np.isfinite(jax.vmap(jax.grad(pa.coprec_window))(q)))
    assert np.all(np.isfinite(jax.vmap(jax.grad(pa.angles_window, argnums=(0, 1)))(q, jnp.linspace(0.0, 1.5, 49))))
    args = connection_arguments(n=20)
    # with the Mf = 100 sentinel appended
    args = [jnp.append(a, 100.0) for a in args[:2]] + [jnp.append(a, a[0]) for a in args[2:]]
    for k in range(4):
        g = jax.vmap(jax.grad(getattr(pa, f"alpha_interpolation_{k}"), argnums=tuple(range(6))))(*args)
        assert all(np.all(np.isfinite(v)) for v in g)
