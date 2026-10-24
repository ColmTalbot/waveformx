# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The MSA (``PhenomXPrecVersion`` 223) and NNLO (102) Euler angles and the
precession set-up of IMRPhenomXP/XPNR against LALSimulation 6.2.0
(``SimIMRPhenomXPMSAAngles``, ``SimIMRPhenomXPPNAngles``,
``SimIMRPhenomXPCalculateModelParametersFromSourceFrame``) and GSL 2.8.

The angle functions of LAL use the global ``powers_of_lalpi`` (in the 3PN
orbital angular momentum, which sets the J frame and the offsets) that only
the IMRPhenomX waveform drivers initialise: the tests call IMRPhenomXAS once
first.

The MSA angles are ill-conditioned in parts of the parameter space: LAL's own
output changes, for a change of the inputs in the last bit, by ~1e-9 in
``cos(beta)`` at equal masses, ~5e-8 in ``alpha`` and ``gamma`` for (nearly)
aligned spins (the degenerate roots of the spin-magnitude cubic, whose
precession is driven by LAL's ``1e-9`` perturbation of the roots) and ~6e-2
for nearly aligned spins with in-plane components ~1e-6 on both bodies. The
tolerances of those cases follow these sensitivities.
"""

import ctypes
import ctypes.util
import os
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformx.models.xpnr import msa, nnlo, precession

Case = namedtuple("Case", "m1 m2 s1 s2 iota fmin fref tol_msa tol_pn", defaults=(3e-12, 1e-13))

# (alpha, gamma, cos(beta)) tolerances of the MSA angles where they are
# ill-conditioned (see the module docstring)
_ALIGNED = (3e-7, 3e-7, 1e-13)
_EQUAL_MASSES = (1e-13, 1e-13, 1e-8)

CASES = {
    # precessing_cases of waveformrs's scripts/generate_reference_data.py
    "generic": Case(36.0, 29.0, [0.4, 0.2, 0.3], [-0.2, 0.3, -0.1], 0.7, 20.0, 20.0),
    "q4": Case(60.0, 15.0, [0.6, -0.3, 0.5], [0.1, 0.1, 0.0], 2.3, 15.0, 20.0),
    "q8": Case(80.0, 10.0, [-0.7, 0.4, -0.2], [0.0, 0.5, 0.2], 1.0, 10.0, 25.0),
    "q1": Case(30.0, 30.0, [0.5, 0.5, 0.5], [0.5, -0.5, -0.5], 1.6, 20.0, 30.0, _EQUAL_MASSES),
    "light": Case(12.0, 6.0, [0.2, 0.7, -0.1], [0.3, -0.3, 0.6], 0.3, 20.0, 20.0),
    "heavy": Case(150.0, 100.0, [0.1, 0.8, 0.4], [0.2, 0.0, -0.8], 2.9, 10.0, 15.0),
    "q10_one_spin": Case(40.0, 4.0, [0.8, 0.0, 0.5], [0.0, 0.0, 0.0], 1.2, 15.0, 20.0),
    "aligned": Case(25.0, 20.0, [0.0, 0.0, 0.5], [0.0, 0.0, -0.3], 0.9, 20.0, 20.0, _ALIGNED),
    # msa_cases: the MSA system fails (|Omega_z5| > 1000)
    "msa_error": Case(5.223612204949848, 3.7902144714569808,
                      [0.28093860818235766, 0.48620119720341687, -0.09813236836152804],
                      [-0.02328864377636709, -0.684967014257048, -0.6656211288407142], 0.7, 20.0, 59.33970557086933,
                      (1e-10, 1e-10, 1e-13)),
    # XPNR edge binaries
    "inplane_5e-8": Case(30.0, 20.0, [5e-8, 0.0, 0.5], [0.0, 0.0, -0.3], 0.33, 20.0, 25.0, _ALIGNED),
    "inplane_5e-7": Case(30.0, 20.0, [5e-7, 0.0, 0.5], [0.0, 0.0, -0.3], 0.46, 20.0, 20.0, _ALIGNED),
    "inplane_2e-6": Case(30.0, 20.0, [2e-6, 0.0, 0.5], [0.0, 0.0, -0.3], 0.59, 20.0, 20.0, _ALIGNED),
    "q9": Case(90.0, 10.0, [0.5, 0.2, 0.1], [0.1, 0.0, 0.2], 0.72, 15.0, 15.0),
    "q11": Case(110.0, 10.0, [0.5, 0.2, 0.1], [0.1, 0.0, 0.2], 0.85, 15.0, 20.0),
    "q3_high_spin": Case(30.0, 10.0, [0.9, 0.1, 0.3], [0.1, 0.0, 0.2], 0.98, 20.0, 20.0),
    "q13": Case(130.0, 10.0, [0.5, 0.2, 0.1], [0.1, 0.0, 0.2], 1.11, 10.0, 10.0),
    "q19": Case(190.0, 10.0, [0.7, 0.2, -0.1], [0.1, 0.4, 0.2], 1.24, 10.0, 12.0, 1e-11),
    "q6_spin_098": Case(30.0, 5.0, [0.98, 0.1, 0.1], [0.5, 0.0, 0.2], 1.37, 20.0, 20.0),
    "equal_masses_and_spins": Case(30.0, 30.0, [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], 1.5, 20.0, 20.0,
                                   _EQUAL_MASSES),
    "st_fallback_1": Case(250.0, 30.0, [0.7, 0.3, -0.4], [0.0, -0.5, 0.5], 1.63, 10.0, 25.0),
    "st_fallback_2": Case(180.0, 65.0, [0.2, 0.1, -0.1], [-0.1, -0.2, 0.1], 1.76, 15.5, 15.5),
    "pnr_msa_error": Case(324.97464539595285, 156.1317450178247,
                          [0.033473504803287535, 0.3586421803050725, -0.8463397636474128],
                          [-0.0181461364021327, -0.7837686724310844, 0.22548210669736396], 1.89, 20.0, 20.0,
                          (1e-10, 1e-10, 1e-12)),
    "low_inplane": Case(65.39462569822626, 5.48546881140544,
                        [0.015777294748968645, -0.01129185185344695, 0.0015599118587599989],
                        [0.26347849219372566, 0.5361809956686221, -0.26720621117427656], 2.02, 30.0,
                        52.60747784129517, (3e-10, 3e-10, 3e-12)),
    "no_mr_beta": Case(238.87854968975887, 13.840439770449478,
                       [-0.006661588927623341, 0.012223592277567882, -0.02867622012464532],
                       [0.21532597836655717, -0.9596190530452179, 0.06428272454610019], 2.15, 30.0,
                       36.54489151780514, (3e-10, 3e-10, 3e-12)),
    "no_beta_connection": Case(141.98456902640373, 34.571276664666726,
                               [0.5558143552228454, -0.08580576236737993, -0.781422167551453],
                               [-0.4951057605866491, 0.15954331438844666, -0.5788066659662768], 2.28, 10.0,
                               34.6828483981249),
    "single_spin": Case(44.776077533636894, 14.83112049039989,
                        [-0.05546660880373749, -0.06115453893757115, 0.36144221300916063], [0.0, 0.0, 0.0],
                        2.41, 30.0, 30.0),
    "long_inspiral": Case(5.0, 3.0, [0.4, -0.3, 0.2], [0.2, 0.3, -0.5], 2.54, 20.0, 20.0),
    "anti_aligned": Case(40.0, 10.0, [0.1, 0.1, -0.98], [0.0, 0.2, -0.9], 2.67, 20.0, 30.0, (1e-10, 1e-10, 1e-11)),
    "high_f_ref": Case(36.0, 29.0, [0.4, 0.2, 0.3], [-0.2, 0.3, -0.1], 2.8, 20.0, 200.0),
    "lighter_first": Case(20.0, 45.0, [0.3, 0.4, -0.2], [-0.5, 0.1, 0.6], 2.93, 15.0, 22.0),
    # nearly aligned spins with in-plane components on both bodies: LAL's
    # alpha and gamma change by ~6e-2 for changes of the inputs in the last bit
    "nearly_aligned_two_spins": Case(30.0, 20.0, [1e-6, 1e-6, 0.5], [1e-6, -2e-6, -0.3], 0.4, 20.0, 20.0,
                                     (1e-1, 1e-1, 3e-8)),
    # face-on, f_ref = 0 (f_min)
    "face_on": Case(36.0, 29.0, [0.4, 0.2, 0.3], [-0.2, 0.3, -0.1], 0.0, 20.0, 0.0),
}

N_FREQUENCIES = 512


def _initialise_phenomx_globals(lalsim):
    """``powers_of_lalpi`` is only initialised by the IMRPhenomX waveform
    drivers: without a waveform call the 3PN orbital angular momentum of the
    angle functions misses its pi^2 term."""
    import lal

    lalsim.SimInspiralChooseFDWaveform(
        30 * lal.MSUN_SI, 30 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, 1e6 * lal.PC_SI,
        0, 0, 0, 0, 0, 1.0, 20.0, 64.0, 20.0, None, lalsim.IMRPhenomXAS,
    )


@pytest.fixture(scope="module")
def lal_globals(lal):
    _initialise_phenomx_globals(lal[1])
    return lal


def _frequencies(case):
    f_max = min(0.3 / ((case.m1 + case.m2) * 4.925490947641267e-6), 4096.0)
    return np.geomspace(case.fmin, f_max, N_FREQUENCIES)


def _lal_angles(lal, lalsim, which, case, mprime):
    frequencies = _frequencies(case)
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, 223 if which == "msa" else 102)
    sequence = lal.CreateREAL8Sequence(len(frequencies))
    sequence.data = frequencies
    func = lalsim.SimIMRPhenomXPMSAAngles if which == "msa" else lalsim.SimIMRPhenomXPPNAngles
    alpha, gamma, cos_beta = func(sequence, case.m1 * lal.MSUN_SI, case.m2 * lal.MSUN_SI, *case.s1, *case.s2,
                                  case.iota, case.fref, mprime, params)
    return frequencies, (alpha.data, gamma.data, cos_beta.data)


_MSA = jax.jit(precession.msa_angles, static_argnums=(7,))
_PN = jax.jit(precession.pn_angles, static_argnums=(7,))


@pytest.mark.parametrize("name", list(CASES))
def test_msa_angles_match_lalsimulation(lal_globals, name):
    lal_module, lalsim = lal_globals
    case = CASES[name]
    tolerance = case.tol_msa if isinstance(case.tol_msa, tuple) else (case.tol_msa,) * 3
    for mprime in (1, 2, 3, 4):
        frequencies, reference = _lal_angles(lal_module, lalsim, "msa", case, mprime)
        alpha, gamma, cos_beta, prec = _MSA(frequencies, case.m1, case.m2, case.s1, case.s2, case.iota, case.fref,
                                            mprime)
        assert bool(prec.msa_error) == name.endswith("msa_error")
        for label, value, ref, tol in zip(("alpha", "gamma", "cos_beta"), (alpha, gamma, cos_beta), reference,
                                          tolerance):
            assert np.all(np.isfinite(value))
            error, _ = compare(np.asarray(value), ref)
            assert error < tol, (label, mprime, error)


@pytest.mark.parametrize("name", list(CASES))
def test_pn_angles_match_lalsimulation(lal_globals, name):
    lal_module, lalsim = lal_globals
    case = CASES[name]
    for mprime in (1, 2, 3, 4):
        frequencies, reference = _lal_angles(lal_module, lalsim, "pn", case, mprime)
        out = _PN(frequencies, case.m1, case.m2, case.s1, case.s2, case.iota, case.fref, mprime)
        for label, value, ref in zip(("alpha", "gamma", "cos_beta"), out[:3], reference):
            error, _ = compare(np.asarray(value), ref)
            assert error < case.tol_pn, (label, mprime, error)


@pytest.mark.parametrize("version", [102, 223])
@pytest.mark.parametrize("name", ["generic", "q1", "aligned", "lighter_first", "msa_error", "single_spin",
                                  "inplane_5e-7", "face_on"])
def test_frame_matches_lalsimulation(lal_globals, name, version):
    """``thetaJN``, ``alpha0``, ``phi_aligned = -phiJ_Sf``, ``zeta`` and
    ``chi_p`` of ``SimIMRPhenomXPCalculateModelParametersFromSourceFrame``
    (with ``phiRef != 0``, which the angle functions do not exercise)."""
    lal_module, lalsim = lal_globals
    case = CASES[name]
    f_ref = case.fref if case.fref > 0 else 25.0
    phi_ref = 1.234
    params = lal_module.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, version)
    reference = lalsim.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        case.m1 * lal_module.MSUN_SI, case.m2 * lal_module.MSUN_SI, f_ref, phi_ref, case.iota, *case.s1, *case.s2,
        params)
    _, _, chi_p, theta_jn, alpha0, phi_aligned, zeta = reference

    @jax.jit
    def frame(m1, m2, chi1, chi2, inclination):
        m1_si, m2_si, chi1, chi2 = precession.check_masses_and_spins(
            m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, chi1, chi2)
        from waveformx.models.xpnr.internals import waveform_struct

        wf = waveform_struct(m1_si, m2_si, chi1[2], chi2[2], 0.125, f_ref, phi_ref, 30.0, 1024.0,
                             1e6 * lal_module.PC_SI, inclination)
        prec, _ = precession.precession_struct(wf, m1_si, m2_si, chi1, chi2, version)
        return prec.chi_p, prec.theta_jn, prec.alpha0, -prec.phi_j_sf, prec.zeta_polarization

    values = frame(case.m1, case.m2, case.s1, case.s2, case.iota)
    for label, value, expected in zip(("chi_p", "thetaJN", "alpha0", "phi_aligned", "zeta"), values,
                                      (chi_p, theta_jn, alpha0, phi_aligned, zeta)):
        difference = abs(float(value) - expected)
        if label in ("alpha0", "zeta"):
            difference = min(difference, abs(difference - 2 * np.pi))
        # thetaJN and alpha0 inherit the conditioning of acos(J0z / |J0|) for
        # nearly aligned spins
        assert difference < 1e-12 * max(1.0, abs(expected)), (label, float(value), expected)


def test_final_spin_and_atan2tol_match_lalsimulation(lal):
    _, lalsim = lal
    for eta, chi1l, chi2l, chi_perp in [(0.25, 0.3, -0.2, 0.5), (0.2, -0.9, 0.4, 0.1), (0.09, 0.0, 0.0, 0.0),
                                        (0.16, 0.95, 0.9, 0.3), (0.05, -0.5, 0.2, 0.9)]:
        assert float(precession.precessing_final_spin_2017(eta, chi1l, chi2l, chi_perp)) == pytest.approx(
            lalsim.SimIMRPhenomXPrecessingFinalSpin2017(eta, chi1l, chi2l, chi_perp), rel=1e-14, abs=1e-15)
    for a, b in [(1e-16, -1e-16), (0.3, -0.2), (-2e-15, 1e-16), (0.0, -1.0)]:
        assert float(precession.atan2_tol(a, b, 1e-15)) == lalsim.SimIMRPhenomXatan2tol(a, b, 1e-15)
    msun = 1.988409870698050731911960804878414216e30
    for f, m1, m2, s1, s2 in [(20.0, 36.0, 29.0, 0.3, -0.1), (10.0, 250.0, 30.0, -0.4, 0.5),
                              (52.6, 65.4, 5.5, 0.0016, -0.27), (15.0, 1.4, 1.3, 0.0, 0.0)]:
        assert float(precession.chirp_time_bound(f, m1 * msun, m2 * msun, s1, s2)) == pytest.approx(
            lalsim.SimInspiralChirpTimeBound(f, m1 * msun, m2 * msun, s1, s2), rel=1e-14)


def _gsl():
    """GSL 2.8 of the lalsimulation environment via ctypes (or None)."""
    candidates = [Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "libgsl.dylib",
                  Path(__file__).resolve().parents[0].parent / "lib" / "libgsl.dylib"]
    import sys

    candidates.append(Path(sys.prefix) / "lib" / "libgsl.dylib")
    candidates.append(Path(sys.prefix) / "lib" / "libgsl.so")
    found = ctypes.util.find_library("gsl")
    for path in [str(c) for c in candidates if c.is_file()] + ([found] if found else []):
        try:
            gsl = ctypes.CDLL(path)
        except OSError:
            continue
        gsl.gsl_set_error_handler_off()
        for name in ("gsl_sf_ellint_F", "gsl_sf_ellint_Kcomp", "gsl_sf_ellint_RF"):
            getattr(gsl, name).restype = ctypes.c_double
        gsl.gsl_sf_ellint_F.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint]
        gsl.gsl_sf_ellint_Kcomp.argtypes = [ctypes.c_double, ctypes.c_uint]
        gsl.gsl_sf_ellint_RF.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_uint]
        gsl.gsl_sf_elljac_e.argtypes = [ctypes.c_double, ctypes.c_double] + [ctypes.POINTER(ctypes.c_double)] * 3
        return gsl
    return None


def test_special_functions_match_gsl():
    gsl = _gsl()
    if gsl is None:
        pytest.skip("libgsl not found")
    u = np.array([0.3, -1.7, 4.2, 12.0, 0.8, 100.0, 1e3, 0.4, -3.0, 2.0])
    m = np.array([0.2, 0.9, 0.5, 0.01, 0.999, 0.7, 1 - 1e-12, 0.0, 1e-17, -0.4])
    sn = np.asarray(jax.jit(msa.elljac_sn)(u, m))
    for ui, mi, value in zip(u, m, sn):
        out = [ctypes.c_double() for _ in range(3)]
        gsl.gsl_sf_elljac_e(ui, mi, *(ctypes.byref(o) for o in out))
        assert value == pytest.approx(out[0].value, rel=1e-14, abs=1e-15)
    phi = np.array([0.1, 0.5, 1.2, np.pi / 2, -0.7, -np.pi / 2, 0.0, 3.0])
    k = np.array([0.3, 0.9, 0.99, 0.5, 0.999999, 0.2, 0.7, 0.4])
    values = np.asarray(jax.jit(msa.ellint_f)(phi, k))
    for p, kk, value in zip(phi, k, values):
        assert value == pytest.approx(gsl.gsl_sf_ellint_F(p, kk, 0), rel=1e-14, abs=1e-15)
    for kk in [0.0, 0.3, 0.9, 0.99999, 1 - 1e-9, -0.5]:
        assert float(msa.ellint_kcomp(kk)) == pytest.approx(gsl.gsl_sf_ellint_Kcomp(kk, 0), rel=1e-14)
    for x, y, z in [(0.0, 1e-200, 1.0), (0.0, 1e-8, 1.0), (0.3, 0.9, 1.0), (2.0, 5.0, 1e-3), (1.0, 1.0, 1.0)]:
        assert float(msa.ellint_rf(x, y, z)) == pytest.approx(gsl.gsl_sf_ellint_RF(x, y, z, 0), rel=1e-14)
    assert np.isnan(float(msa.elljac_sn(0.4, 1.5)))


def _scalar(func, frequencies, weights):
    """A smooth scalar of the Euler angles of ``func`` as a function of
    ``(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, inclination,
    f_ref)``."""

    def scalar(x):
        alpha, gamma, cos_beta, _ = func(frequencies, x[0], x[1], [x[2], x[3], x[4]], [x[5], x[6], x[7]], x[8],
                                         x[9], 2)
        return jnp.sum(weights[0] * jnp.sin(alpha) + weights[1] * jnp.cos(gamma) + weights[2] * cos_beta)

    return scalar


GRADIENT_FREQUENCIES = np.geomspace(20.0, 300.0, 40)
GRADIENT_WEIGHTS = np.array([0.3, 0.7, 2.0])


@pytest.mark.parametrize("which", ["msa", "pn"])
def test_gradients(which):
    """``jax.grad`` of a smooth scalar of the angles: finite everywhere (also
    for aligned, nearly aligned and single spins, equal masses, the failing
    MSA system and a face-on line of sight) and equal to central finite
    differences for generic binaries (at aligned spins the angles are not
    differentiable with respect to the in-plane spins, and for nearly aligned
    spins finite differences of LAL's expressions are dominated by rounding)."""
    func = precession.msa_angles if which == "msa" else precession.pn_angles
    scalar = _scalar(func, GRADIENT_FREQUENCIES, GRADIENT_WEIGHTS)
    value_and_grad = jax.jit(jax.value_and_grad(scalar))
    value = jax.jit(scalar)

    def point(case):
        return jnp.array([case.m1, case.m2, *case.s1, *case.s2, case.iota, case.fref])

    for name in ["generic", "q8", "aligned", "inplane_5e-8", "inplane_2e-6", "nearly_aligned_two_spins",
                 "single_spin", "msa_error", "pnr_msa_error", "face_on", "anti_aligned"]:
        _, gradient = value_and_grad(point(CASES[name]))
        assert np.all(np.isfinite(np.asarray(gradient))), name
    # equal masses: the derivatives with respect to the masses are NaN because
    # of sqrt(1 - 4 eta) in the shared fits (chi_eff, the final spin) of the
    # waveform struct, not in these modules
    for name in ["q1", "equal_masses_and_spins"]:
        _, gradient = value_and_grad(point(CASES[name]))
        assert np.all(np.isfinite(np.asarray(gradient)[2:])), name

    for name in ["generic", "q8", "lighter_first"]:
        x = point(CASES[name])
        _, gradient = value_and_grad(x)
        for k in range(len(x)):
            h = 1e-5 * max(abs(float(x[k])), 1e-2)
            e = jnp.zeros_like(x).at[k].set(h)
            fd = (value(x + e) - value(x - e)) / (2 * h)
            assert float(gradient[k]) == pytest.approx(float(fd), rel=1e-6, abs=1e-6 * float(jnp.max(jnp.abs(gradient)))), (
                name, k)


def test_msa_system_gradients_at_removable_singularities():
    """The MSA system and angles have finite derivatives with respect to
    their inputs at equal masses, for a single spin, zero spins and aligned
    spins (where LAL switches to the degenerate roots)."""
    v = jnp.linspace(0.15, 0.45, 16)

    def scalar(x):
        system = msa.msa_system(x[0], x[1], x[2], x[1] * x[1], x[2] * x[2], [x[3], x[4], x[5]], [x[6], x[7], x[8]],
                                x[9], 20.0)
        angles = msa.phi_zeta_costhetal(system, v)
        phiz, zeta, cos_theta_l = angles.phiz, angles.zeta, angles.cos_theta_l
        l = precession.orbital_angular_momentum_3pn(x[0], x[1] - x[2], x[5], x[8])
        cos_with = msa.cos_theta_l_with(system, v, l)
        return (jnp.sum(jnp.sin(phiz) + jnp.cos(zeta) + cos_theta_l + cos_with)
                + sum(msa.psi_terms(system)) + msa.final_spin(system, 0.7, 1.0, x[1] ** 2, x[2] ** 2, x[5], x[8]))

    grad = jax.jit(jax.grad(scalar))
    for chi1, chi2, masses in [
        ([0.4, 0.2, 0.3], [-0.2, 0.3, -0.1], (0.5, 0.5)),  # equal masses
        ([0.4, 0.2, 0.3], [0.0, 0.0, 0.0], (0.6, 0.4)),  # a single spin
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.6, 0.4)),  # no spins
        ([0.0, 0.0, 0.5], [0.0, 0.0, -0.3], (0.6, 0.4)),  # aligned spins
        ([1e-6, 0.0, 0.5], [0.0, 1e-6, -0.3], (0.6, 0.4)),  # nearly aligned spins
        ([0.0, 0.0, 0.5], [0.0, 0.0, -0.3], (0.5, 0.5)),  # aligned spins, equal masses
    ]:
        m1, m2 = masses
        x = jnp.array([m1 * m2, m1, m2, *chi1, *chi2, np.pi * 50.0 * 4.925490947641267e-6])
        assert np.all(np.isfinite(np.asarray(grad(x)))), (chi1, chi2, masses)

    # the system that sets MSA_ERROR: finite angles and derivatives (so that
    # selecting the NNLO fallback with jnp.where has finite gradients)
    case = CASES["msa_error"]
    total = case.m1 + case.m2
    m1, m2 = case.m1 / total, case.m2 / total
    x = jnp.array([m1 * m2, m1, m2, *case.s1, *case.s2, np.pi * total * 4.925490947641267e-6])
    system = msa.msa_system(x[0], x[1], x[2], x[1] * x[1], x[2] * x[2], list(x[3:6]), list(x[6:9]), x[9],
                            case.fref)
    assert bool(msa.msa_error(system))

    def fallback(x):
        system = msa.msa_system(x[0], x[1], x[2], x[1] * x[1], x[2] * x[2], [x[3], x[4], x[5]],
                                [x[6], x[7], x[8]], x[9], case.fref)
        a = msa.phi_zeta_costhetal(system, v)
        angles = jnp.sum(jnp.sin(a.phiz) + jnp.cos(a.zeta) + a.cos_theta_l)
        return jnp.where(msa.msa_error(system), jnp.sum(v), angles)

    assert np.all(np.isfinite(np.asarray(jax.jit(jax.grad(fallback))(x))))
    assert np.all(np.isfinite(np.asarray(jax.jit(jax.grad(scalar))(x))))


def test_vmap_over_binaries(lal_globals):
    """``jax.vmap`` over a batch of binaries (including one whose MSA system
    fails) gives the angles of the individual binaries."""
    names = ["generic", "msa_error", "single_spin", "aligned"]
    frequencies = np.geomspace(20.0, 200.0, 64)
    m1 = jnp.array([CASES[n].m1 for n in names])
    m2 = jnp.array([CASES[n].m2 for n in names])
    s1 = jnp.array([CASES[n].s1 for n in names])
    s2 = jnp.array([CASES[n].s2 for n in names])
    iota = jnp.array([CASES[n].iota for n in names])
    fref = jnp.array([CASES[n].fref for n in names])
    for func in (precession.msa_angles, precession.pn_angles):
        batched = jax.jit(jax.vmap(lambda a, b, c, d, e, f: func(frequencies, a, b, list(c), list(d), e, f, 3)[:3]))
        out = batched(m1, m2, s1, s2, iota, fref)
        for i, n in enumerate(names):
            case = CASES[n]
            single = func(frequencies, case.m1, case.m2, case.s1, case.s2, case.iota, case.fref, 3)[:3]
            # the batched and single evaluations are compiled differently: they
            # agree to the conditioning of the angles
            tolerance = {"msa_error": 1e-10, "aligned": 1e-6}.get(n, 1e-12)
            for a, b in zip(out, single):
                assert compare(np.asarray(a[i]), np.asarray(b))[0] < tolerance, n


def test_nnlo_beta_is_smooth_for_aligned_spins():
    """``nnlo_beta`` is LAL's ``acos(cos(beta))`` with a finite derivative
    for aligned spins (``Sperp = 0``)."""
    grad = jax.grad(lambda sperp: nnlo.nnlo_beta(1.3, sperp, 0.2))
    assert float(grad(0.0)) == pytest.approx(1.0 / 1.5)
    assert float(nnlo.nnlo_beta(1.3, 0.0, 0.2)) == 0.0
    assert float(nnlo.nnlo_beta(1.3, 0.3, 0.2)) == pytest.approx(float(jnp.arccos(nnlo.nnlo_cos_beta(1.3, 0.3, 0.2))))
