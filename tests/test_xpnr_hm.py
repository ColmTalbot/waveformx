# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Milestone 2 of IMRPhenomXPNR: the higher modes of IMRPhenomXHM with
``PhenomXHMReleaseVersion = 122019`` against
``SimIMRPhenomXHMFrequencySequenceOneMode`` of LALSimulation 6.2.0.

The binaries are waveformrs's ``XHM_122019_BINARIES``, chosen so that their
modes go through every branch of the release's amplitude (vetoes of
collocation points, the order of the intermediate polynomial, two
intermediate regions for q > 70, the TaylorT4 inspiral of the (2,1), the
inspiral ansatz in the ringdown of the (3,3) and (4,4), vanishing odd
modes), plus the lighter body first.

Accuracy. Everything except the intermediate amplitude agrees with LAL to
~1e-12. The intermediate amplitude is the inverse of a polynomial fixed by
the derivative of the inspiral (and, for the (3,2), of the rotated
ringdown) amplitude, which LAL takes by fourth-order finite differences
with a step of 1e-9: rounding differences of the ansatz are amplified to
~1e-7 in the derivative and ~1e-8 in the amplitude (LAL's own Linux and
macOS builds differ by up to 2e-8 here, waveformrs's cross-build
tolerances). The intermediate phase of the (3,2) is fixed by the second
derivative of the rotated ringdown phase by finite differences with a step
of 1e-7, which amplifies rounding (of a 22 phase of up to ~1e4 rad for
extreme mass ratios) by ~1e14: the (3,2) agrees to ~1e-4 for comparable
masses and to ~1e-2 (mismatch < 1e-4) at mass ratio 100, as different
builds of LAL do.
"""

import ctypes
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformjax.models.xpnr import hm, phase22
from waveformjax.models.xpnr._math import MPC_SI, MSUN_SI
from waveformjax.models.xpnr.hm import roots
from waveformjax.models.xpnr.internals import waveform_struct

# mass_1, mass_2, chi_1, chi_2, f_ref (waveformrs's XHM_122019_BINARIES)
XHM_122019_BINARIES = [
    (36.0, 29.0, 0.3, -0.4, 30.0),
    (30.0, 30.0, 0.5, -0.5, 20.0),
    (150.2452, 2.957, 0.945, -1.0, 30.0),
    (79.0458, 0.2689, -0.89, 0.959, 0.0),
    (10.7049, 10.1951, 0.63, 0.61, 0.0),
    (100.9494, 12.6503, 0.988, -0.02, 30.0),
    (148.5018, 3.4991, -0.914, -0.93, 20.0),
    (109.3352, 91.2648, -0.966, 0.991, 0.0),
    (66.0796, 65.6203, 0.77, 0.99, 0.0),
    (109.45, 109.45, -0.947, -0.947, 20.0),
    (91.0032, 63.1967, 0.989, 0.31, 30.0),
    (118.8765, 113.3236, -0.991, 0.63, 30.0),
    (28.1353, 9.1646, 0.907, -0.97, 30.0),
    (200.0, 2.0, 0.95, 0.3, 15.0),
]
# the lighter body first, more mass ratios and equal masses with equal spins
EXTRA_BINARIES = [
    (20.0, 45.0, 0.3, -0.4, 0.0),
    (12.6503, 100.9494, -0.02, 0.988, 25.0),
    (60.0, 15.0, 0.8, 0.5, 20.0),
    (8.0, 1.4, 0.9, 0.0, 50.0),
    (30.0, 30.0, 0.5, 0.5, 20.0),
]
BINARIES = XHM_122019_BINARIES + EXTRA_BINARIES
NEGATIVE = [(2, -1), (3, -3), (3, -2), (4, -4)]
# relative error and mismatch: the (3,2) and the others (see above)
TOLERANCE_32 = (3e-2, 1e-4)
TOLERANCE = (5e-8, 1e-12)


def lal_mode(lal, freqs, m1, m2, c1, c2, ell, emm, distance, phi_ref, f_ref):
    lal_module, lalsim = lal
    vector = lal_module.CreateREAL8Vector(len(freqs))
    vector.data = freqs
    params = lal_module.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(params, 122019)
    return lalsim.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, c1, c2, ell, emm,
        distance * 1e6 * lal_module.PC_SI, phi_ref, f_ref, params,
    ).data.data


_one_mode = jax.jit(hm.frequency_sequence_one_mode, static_argnums=(5, 6))


@pytest.mark.parametrize("index", range(len(BINARIES)))
def test_modes_match_lalsimulation(lal, index):
    lal_module, _ = lal
    m1, m2, c1, c2, f_ref = BINARIES[index]
    distance, phi_ref = 500.0, 0.1 + 0.4 * index
    f_cut = 0.3 / ((m1 + m2) * lal_module.MTSUN_SI)
    # from f_min to beyond the cutoff of the modes
    freqs = np.geomspace(10.0, 1.2 * f_cut, 1500)
    modes = NEGATIVE + ([(2, 1), (3, 3), (3, 2), (4, 4), (2, -2), (2, 2)] if index in (0, 14) else [])
    failures = []
    for ell, emm in modes:
        reference = lal_mode(lal, freqs, m1, m2, c1, c2, ell, emm, distance, phi_ref, f_ref)
        test = np.asarray(_one_mode(freqs, m1, m2, c1, c2, ell, emm, distance, phi_ref, f_ref))
        assert np.all(np.isfinite(test))
        # zero above the cutoff, as LAL
        assert np.all(test[freqs > f_cut * 1.0001] == 0)
        error, mismatch = compare(test, reference)
        print(f"m1={m1} m2={m2} chi=({c1}, {c2}) ({ell},{emm:+d}): rel. error {error:.2e}, mismatch {mismatch:.2e}")
        rel, mm = TOLERANCE_32 if (ell, abs(emm)) == (3, 2) else (1e-10, 1e-14) if ell == 2 and abs(emm) == 2 else TOLERANCE
        if not (error < rel and mismatch < mm):
            failures.append(((ell, emm), error, mismatch))
    assert not failures, failures


def test_equal_masses_and_spins_odd_modes_vanish(lal):
    freqs = np.geomspace(10.0, 300.0, 200)
    for ell, emm in [(2, -1), (3, -3), (2, 1), (3, 3)]:
        test = np.asarray(_one_mode(freqs, 30.0, 30.0, 0.5, 0.5, ell, emm, 400.0, 0.2, 20.0))
        assert np.all(test == 0)
    assert np.all(np.abs(np.asarray(_one_mode(freqs, 30.0, 30.0, 0.5, 0.5, 4, -4, 400.0, 0.2, 20.0))) > 0)


def _coefficients(m1, m2, c1, c2, f_ref=20.0, ell=2, emm=1):
    wf = waveform_struct(m1 * MSUN_SI, m2 * MSUN_SI, c1, c2, 0.0, f_ref, 0.3, 10.0, 1000.0, 500 * MPC_SI, 0.0)
    wf.hm_release = 122019
    return hm.mode_coefficients(wf, phase22.phase_coefficients(wf), ell, emm)


def test_branches_are_exercised():
    """The binaries go through the discrete choices of the 122019 amplitude."""
    seen = set()
    for m1, m2, c1, c2, f_ref in BINARIES:
        for ell, emm in hm.HIGHER_MODES:
            c = _coefficients(m1, m2, c1, c2, f_ref, ell, emm)
            a = c.amp
            if bool(c.amp_zero):
                seen.add("amp_zero")
                continue
            assert not bool(c.error_poly_roots)
            seen.add(f"inspiral {int(a.inspiral_amp_version)}")
            seen.add(f"{int(a.intermediate_amp_version)} -> {int(a.inter_amp_pol_order)}")
            for flag in ("amp_emr", "use_famp_pn", "use_insp_ansatz_ringdown"):
                if bool(getattr(a, flag)):
                    seen.add(flag)
            if ell == 2 and not bool(a.use_famp_pn):
                seen.add("21 PN series")
    expected = {
        "amp_zero", "amp_emr", "use_famp_pn", "21 PN series", "use_insp_ansatz_ringdown",
        "inspiral 0", "inspiral 1", "inspiral 2", "inspiral 3",
        "105 -> 105", "105 -> 104", "105 -> 102", "1042 -> 104", "1032 -> 103", "101 -> 101",
    }
    assert expected <= seen, expected - seen


def _libgsl():
    for name in ("libgsl.dylib", "libgsl.so"):
        path = Path(sys.prefix) / "lib" / name
        if path.exists():
            return ctypes.CDLL(str(path))
    pytest.skip("libgsl not found")


def test_gsl_poly_complex_solve():
    """The port of gsl_poly_complex_solve returns GSL's roots in GSL's
    order (which CrossZeroP5 depends on)."""
    gsl = _libgsl()
    gsl.gsl_poly_complex_workspace_alloc.restype = ctypes.c_void_p
    gsl.gsl_poly_complex_workspace_alloc.argtypes = [ctypes.c_size_t]
    gsl.gsl_poly_complex_workspace_free.argtypes = [ctypes.c_void_p]
    double_p = ctypes.POINTER(ctypes.c_double)
    gsl.gsl_poly_complex_solve.argtypes = [double_p, ctypes.c_size_t, ctypes.c_void_p, double_p]
    workspace = gsl.gsl_poly_complex_workspace_alloc(6)
    rng = np.random.default_rng(1)
    polynomials = [rng.normal(size=6) for _ in range(100)]
    # scaled like the intermediate polynomials, delta_k ~ v / f^k
    polynomials += [rng.normal(size=6) * 0.05 ** -np.arange(6.0) for _ in range(100)]
    polynomials += [np.poly(rng.uniform(0.005, 0.2, size=5) * rng.choice([-1, 1], size=5))[::-1] * 1e5
                    for _ in range(100)]
    polynomials = np.array(polynomials)
    test, ok = jax.jit(jax.vmap(roots.complex_solve_quintic))(jnp.asarray(polynomials))
    for a, z, success in zip(polynomials, np.asarray(test), np.asarray(ok)):
        a = np.ascontiguousarray(a)
        expected = np.zeros(10)
        status = gsl.gsl_poly_complex_solve(a.ctypes.data_as(double_p), 6, workspace, expected.ctypes.data_as(double_p))
        assert status == 0 and success
        np.testing.assert_allclose(z, expected, rtol=1e-12, atol=1e-12 * np.max(np.abs(expected)))
        # real roots stay real
        assert np.all((z[1::2] == 0) == (expected[1::2] == 0))
    gsl.gsl_poly_complex_workspace_free(workspace)


def test_cross_zero_tests():
    # (f - 0.05)(f - 0.2)(f + 1)
    cubic = jnp.array([0.01, -0.24, 0.75, 1.0])
    assert roots.cross_zero_p3(cubic, 0.04, 0.06)
    assert not roots.cross_zero_p3(cubic, 0.06, 0.15)
    # times (f^2 + 1): GSL returns the roots -1, 0.2, 0.05, i, -i and, like
    # LAL, CrossZeroP5 misses the third root
    quintic = jnp.array([0.01, -0.24, 0.76, 0.76, 0.75, 1.0])
    assert roots.cross_zero_p5(quintic, 0.15, 0.25)[0]
    assert not roots.cross_zero_p5(quintic, 0.04, 0.06)[0]
    assert not roots.cross_zero_p5(quintic, 0.06, 0.15)[0]
    # the cubic times (f - 0.3)
    quartic = jnp.array([-0.003, 0.082, -0.465, 0.45, 1.0])
    assert roots.cross_zero_p4(quartic, 0.28, 0.32)
    assert not roots.cross_zero_p4(quartic, 0.21, 0.28)
    quadratic = jnp.array([-0.01, 0.0, 1.0])
    assert roots.cross_zero_p2(quadratic, 0.05, 0.15)
    assert not roots.cross_zero_p2(quadratic, 0.15, 0.25)


def test_qnm_fits_match_lalsimulation(lal):
    library = Path(sys.prefix) / "lib" / ("liblalsimulation.dylib" if sys.platform == "darwin" else "liblalsimulation.so")
    if not library.exists():
        pytest.skip("liblalsimulation not found")
    lalsim = ctypes.CDLL(str(library))
    names = [f"{kind}{mode}" for kind in ("fring", "fdamp") for mode in (21, 33, 32, 44)]
    names += [f"{part}_{coeff}" for part in ("re", "im") for coeff in ("l2m2lp2", "l2m2lp3", "l3m2lp2", "l3m2lp3")]
    for name in names:
        function = getattr(lalsim, f"evaluate_QNMfit_{name}")
        function.restype = ctypes.c_double
        function.argtypes = [ctypes.c_double]
        for af in [-0.9, -0.3, 0.0, 0.5, 0.9, 0.99]:
            # the numerators and denominators cancel strongly near extremal
            # spins (rounding amplified to ~1e-11 at af = 0.99)
            assert float(getattr(hm, name)(af)) == pytest.approx(function(af), rel=1e-10, abs=1e-15), (name, af)


def test_jit_vmap_grad():
    """mode_h(mode_coefficients(...)) as a function of the masses and spins
    compiles, maps over a batch of binaries and has finite gradients that
    agree with finite differences."""
    def h_of(params, ell, emm, mf=jnp.geomspace(0.004, 0.2, 120)):
        m1, m2, c1, c2 = params
        wf = waveform_struct(m1 * MSUN_SI, m2 * MSUN_SI, c1, c2, 0.0, 20.0, 0.3, 10.0, 1000.0, 500 * MPC_SI, 0.0)
        wf.hm_release = 122019
        coefficients = hm.mode_coefficients(wf, phase22.phase_coefficients(wf), ell, emm)
        return hm.mode_h(coefficients, mf)

    batch = jnp.array([(36.0, 29.0, 0.3, -0.4), (79.0458, 0.2689, -0.89, 0.959), (100.9494, 12.6503, 0.988, -0.02)])
    for ell, emm in hm.HIGHER_MODES:
        mapped = jax.jit(jax.vmap(lambda p: h_of(p, ell, emm)))(batch)
        single = jnp.stack([jax.jit(lambda p: h_of(p, ell, emm))(p) for p in batch])
        # the vectorised code rounds differently, which the finite
        # differences of the intermediate region amplify (see above)
        for test, reference in zip(np.asarray(mapped), np.asarray(single)):
            error, mismatch = compare(test, reference)
            print(f"({ell},{emm}) vmap against single binaries: {error:.1e}, mismatch {mismatch:.1e}")
            assert error < (TOLERANCE_32 if (ell, emm) == (3, 2) else TOLERANCE)[0]

        def loss(params):
            # above Mf = 0.02, where the phase varies slowly with the masses
            h = h_of(params, ell, emm, jnp.geomspace(0.02, 0.2, 60)) * 1e21
            return jnp.sum(jnp.real(h) + jnp.abs(h) ** 2)

        grad = jax.jit(jax.grad(loss))
        loss = jax.jit(loss)
        for params in batch[:2]:
            g = np.asarray(grad(params))
            assert np.all(np.isfinite(g)), ((ell, emm), g)
            for k in range(4):
                # the phase of extreme mass ratios varies quickly with the
                # masses (small steps), while the finite differences inside
                # the model make it rough at the 1e-8 level (large steps):
                # the gradient must agree with one of a few steps (the (3,2)
                # is rough at the 1e-4 level, see above)
                mode32 = (ell, emm) == (3, 2)
                fds = []
                for relative_step in (1e-3, 3e-4, 1e-4) if mode32 else (1e-4, 1e-5, 1e-6):
                    step = relative_step * max(0.1, abs(float(params[k])))
                    e = jnp.zeros(4).at[k].set(step)
                    fds.append(float((loss(params + e) - loss(params - e)) / (2 * step)))
                best = min(fds, key=lambda fd: abs(fd - g[k]))
                print(f"({ell},{emm}) d/dx{k}: grad {g[k]:.6e}, finite difference {best:.6e}")
                tolerance = 1e-2 if mode32 else 1e-3
                assert abs(best - g[k]) <= tolerance * (abs(g[k]) + np.max(np.abs(g))), ((ell, emm), k, fds)
