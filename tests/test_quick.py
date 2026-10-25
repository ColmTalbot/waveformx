# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The quick tests: what ``pytest`` runs by default (``testpaths`` in
``pyproject.toml``, with test_api.py), within a minute from an empty
compilation cache, before every commit. The whole suite is ``pytest tests``.

Every model is compiled once, as one function that gives the waveforms and
their derivatives for a batch of binaries: ``jax.vmap`` of
``jax.value_and_grad`` of a fixed real function of the polarizations (with
the polarizations as auxiliary output), or of ``jax.jvp`` for IMRPhenomXHM,
whose reverse-mode derivative takes minutes to compile
(test_transforms_xhm.py). The batch holds a binary ``x`` and ``x +- h v``
along a direction ``v``, so one call checks ``jax.jit``, ``jax.vmap``, the
derivative (against the central difference along ``v``), that everything is
finite, and the waveform at ``x`` against LAL (or waveformrs, which
reproduces LAL to rounding). The functions are compiled concurrently, in
threads, when the first test asks for one.

* TaylorF2, IMRPhenomD, IMRPhenomXAS: an aligned-spin binary on 64
  frequencies, against LAL; float32 against float64;
* IMRPhenomXHM: the same binary, against LAL;
* IMRPhenomXPNR: the (2,+-2) and (3,+-3) modes of a generic precessing
  binary of 65 solar masses from 20 Hz, against waveformrs with LAL's parity
  fix; finite at an aligned and a non-spinning binary;
* NRSur7dq4 with ``simulate_fd`` (the time-domain model and its conversion
  to the frequency domain): the (2,+-2) and (3,+-3) modes, against LAL's
  ``SimInspiralFD`` (if its data file is found);
* the conversion of a frequency-domain model to the time domain
  (``SimInspiralTD``), given LAL's IMRPhenomXPNR;
* ``waveformx.match``.

What these leave out is in the per-model files: every option, mode and
binary of the suites against LAL, float32 throughout, and the derivatives
with respect to every parameter.
"""

import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf
from conftest import NRSUR_DATA, compare, requires_nrsur
from waveformx import Binary, WaveformOptions
from waveformx.models import get

MTSUN_SI = 4.925490947641267e-06
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]
#: precessing, from waveformrs's suite (65 solar masses: no SpinTaylor
#: evolution of light binaries from low frequencies in the quick tests)
GENERIC = [36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0]
ALIGNED = [36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.4, 500.0, 1.0, 1.2, 30.0]
NON_SPINNING = [36.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 1.0, 1.2, 30.0]
NRSUR = [40.0, 20.0, 0.4, -0.3, 0.2, 0.1, 0.5, -0.3, 100.0, 0.7, 0.1, 20.0]
#: the dominant modes and one higher mode
MODES = ((2, 2), (2, -2), (3, 3), (3, -3))
#: the relative step of the central difference along the direction (larger
#: for IMRPhenomXPNR, whose SpinTaylor angles add noise of ~1e-10 to its
#: values: the difference converges to the gradient, to 2e-8, at 1e-4)
STEP = 1e-6
XPNR_STEP = 1e-4


def binary_of(x):
    return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                  inclination=x[9], phi_ref=x[10], f_ref=x[11])


def keywords_of(x):
    x = [float(v) for v in x]
    return dict(mass_1=x[0], mass_2=x[1], spin_1=tuple(x[2:5]), spin_2=tuple(x[5:8]), luminosity_distance=x[8],
                inclination=x[9], phi_ref=x[10], f_ref=x[11])


def direction(x, aligned=False):
    """A fixed direction, scaled by the parameters (and ``1`` for those
    near zero); along the aligned spins only for aligned-spin models."""
    x = np.asarray(x)
    v = np.random.default_rng(7).normal(size=12) * np.maximum(np.abs(x), 0.1)
    if aligned:
        v[[2, 3, 5, 6]] = 0.0
    return v


def batch(x, v, *others, step=STEP):
    """``x``, ``x + h v``, ``x - h v`` and others."""
    x, v = np.asarray(x), np.asarray(v)
    return jnp.asarray(np.stack([x, x + step * v, x - step * v, *others]))


def weighted_sum(polarizations, n):
    """A fixed real function of the polarizations (``n`` frequencies or
    samples)."""
    w = np.random.default_rng(3).normal(size=(4, n))
    hp, hc = polarizations
    scale = 1e21 / (n * np.sqrt(n))
    return scale * jnp.sum(w[0] * hp.real + w[1] * hp.imag + w[2] * hc.real + w[3] * hc.imag)


def value_and_grad(polarizations, n):
    """``vmap`` of ``(loss, polarizations), gradient``."""
    def function(x):
        h = polarizations(x)
        return weighted_sum(h, n), h

    return jax.jit(jax.vmap(jax.value_and_grad(function, has_aux=True)))


def value_and_jvp(polarizations, n, v):
    """``vmap`` of ``(loss, polarizations), derivative along v``."""
    def function(x):
        h = polarizations(x)
        return weighted_sum(h, n), h

    def along(x):
        (loss, h), (tangent, _) = jax.jvp(function, (x,), (jnp.asarray(v),))
        return (loss, h), tangent

    return jax.jit(jax.vmap(along))


def fetch(tree):
    return jax.tree_util.tree_map(np.asarray, jax.block_until_ready(tree))


# --- the jobs: each compiles and runs one function ---------------------------

#: the strain itself (not times 1e21), as LAL
STRAIN = WaveformOptions(scaled_strain=False)
ALIGNED_MODELS = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS"]
FREQUENCIES = np.geomspace(20.0, 0.18 / (65.0 * MTSUN_SI), 64)
XPNR_FREQUENCIES = np.geomspace(20.0, 0.3 / (65.0 * MTSUN_SI), 64)


def aligned_job(name):
    """The batch in float64, and the polarizations at ``ALIGNED`` in float32
    and float64 (rounded to float32 first)."""
    model = get(name)
    f = jnp.asarray(FREQUENCIES)
    function = value_and_grad(lambda x: model.polarizations(binary_of(x), f, STRAIN), len(f))
    v = direction(ALIGNED, aligned=True)
    derivatives = v, fetch(function(batch(ALIGNED, v, NON_SPINNING)))
    single = jax.jit(lambda x, f: model.polarizations(binary_of(x), f, WaveformOptions()))
    x, f = np.float32(ALIGNED), np.float32(FREQUENCIES)
    float32 = fetch(single(jnp.asarray(x), jnp.asarray(f))), fetch(single(jnp.asarray(x, jnp.float64),
                                                                         jnp.asarray(f, jnp.float64)))
    return derivatives, float32


def xhm_job():
    model = get("IMRPhenomXHM")
    f = jnp.asarray(FREQUENCIES)
    v = direction(ALIGNED, aligned=True)
    function = value_and_jvp(lambda x: model.polarizations(binary_of(x), f, STRAIN), len(f), v)
    return v, fetch(function(batch(ALIGNED, v, NON_SPINNING)))


def xpnr_job():
    model = get("IMRPhenomXPNR")
    f = jnp.asarray(XPNR_FREQUENCIES)
    options = WaveformOptions(mode_array=MODES, scaled_strain=False)
    function = value_and_grad(lambda x: model.polarizations(binary_of(x), f, options), len(f))
    v = direction(GENERIC)
    return v, fetch(function(batch(GENERIC, v, ALIGNED, NON_SPINNING, step=XPNR_STEP)))


def nrsur_job():
    simulate = wf.simulate_fd_function("NRSur7dq4", 0.0, 20.0, 1024.0, mode_array=MODES, scaled_strain=False,
                                       **keywords_of(NRSUR))
    function = value_and_grad(lambda x: simulate(binary_of(x)), len(simulate))
    v = direction(NRSUR)
    return (len(simulate), simulate.epoch), v, fetch(function(batch(NRSUR, v)))


JOBS = {
    **{name: functools.partial(aligned_job, name) for name in ALIGNED_MODELS},
    "IMRPhenomXHM": xhm_job,
    "IMRPhenomXPNR": xpnr_job,
}
if NRSUR_DATA is not None:
    JOBS["NRSur7dq4"] = nrsur_job


@pytest.fixture(scope="session")
def results():
    """The jobs' results (futures), all started at once in processes of
    their own: tracing is serial under the GIL, and the models keep the
    precision being traced in a global (``models.xpnr._math.precision``),
    so threads would mix float32 and float64."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(JOBS), mp_context=context) as pool:
        yield {name: pool.submit(JOBS[name]) for name in JOBS}


def check_derivative(loss, derivative, rtol=1e-5, step=STEP):
    """The derivative at ``x`` along ``v`` against the central difference
    of the loss at ``x +- h v``."""
    assert np.all(np.isfinite(loss)) and np.all(np.isfinite(derivative))
    numeric = (loss[1] - loss[2]) / (2 * step)
    assert derivative == pytest.approx(numeric, rel=rtol, abs=1e-3 * rtol * max(abs(numeric), 1.0))


def check_close(test, reference, error_tolerance, mismatch_tolerance):
    for a, b in zip(test, reference):
        assert np.all(np.isfinite(a)) and a.shape == np.shape(b)
        error, mismatch = compare(a, b)
        assert error < error_tolerance and mismatch < mismatch_tolerance, (error, mismatch)


def lal_fd(lal, approximant, x, frequencies, mode_array=None):
    """``SimInspiralChooseFDWaveformSequence``."""
    lal_module, lalsim = lal
    x = [float(v) for v in x]
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = np.asarray(frequencies)
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        x[10], x[0] * lal_module.MSUN_SI, x[1] * lal_module.MSUN_SI, *x[2:8], x[11], x[8] * 1e6 * lal_module.PC_SI,
        x[9], None, getattr(lalsim, approximant), vector,
    )
    return hp.data.data, hc.data.data


# --- the tests ---------------------------------------------------------------

#: (relative error, mismatch) against LAL: rounding, but IMRPhenomD's fit of
#: the QNM frequencies in place of LAL's spline (test_phenomd.py)
LAL_TOLERANCE = {"TaylorF2": (1e-10, 1e-14), "IMRPhenomD": (2e-5, 1e-10), "IMRPhenomXAS": (1e-10, 1e-14)}


@pytest.mark.parametrize("name", ALIGNED_MODELS)
def test_aligned_spin_model(lal, results, name):
    """jit, vmap, grad, and LAL's polarizations (to rounding)."""
    v, ((loss, polarizations), gradient) = results[name].result()[0]
    check_derivative(loss, gradient[0] @ v)
    assert np.all(gradient[:, [2, 3, 5, 6]] == 0.0)
    assert np.all(np.isfinite(gradient))
    ours = [h[0] for h in polarizations]
    check_close(ours, lal_fd(lal, name, ALIGNED, FREQUENCIES), *LAL_TOLERANCE[name])
    assert all(np.all(np.isfinite(h[3])) for h in polarizations)


@pytest.mark.parametrize("name", ALIGNED_MODELS)
def test_float32(results, name):
    single, double = results[name].result()[1]
    assert single[0].dtype == np.complex64
    for a, b in zip(single, double):
        assert compare(np.asarray(a, np.complex128), b)[1] < 1e-6


def test_xhm(lal, results):
    """jit, vmap, jvp, and LAL's polarizations (to the (3,2) mode's finite
    differences in LAL; test_xhm.py)."""
    _, ((loss, polarizations), tangent) = results["IMRPhenomXHM"].result()
    check_derivative(loss, tangent[0])
    ours = [h[0] for h in polarizations]
    check_close(ours, lal_fd(lal, "IMRPhenomXHM", ALIGNED, FREQUENCIES), 1e-3, 1e-7)
    assert all(np.all(np.isfinite(h[3])) for h in polarizations)


def test_xpnr(results):
    """jit, vmap, grad (the transposed custom JVP), finite at aligned and
    vanishing spins, and waveformrs (LAL with the parity fix) for the same
    modes: a mismatch of ~1e-9 (test_xpnr.py)."""
    rs = pytest.importorskip("waveformrs")
    v, ((loss, polarizations), gradient) = results["IMRPhenomXPNR"].result()
    check_derivative(loss, gradient[0] @ v, step=XPNR_STEP)
    assert np.all(np.isfinite(gradient))
    for h in polarizations:
        assert np.all(np.isfinite(h))
    reference = rs.polarizations("IMRPhenomXPNR", XPNR_FREQUENCIES, mode_array=MODES, pnr_parity_fix=True,
                                 **keywords_of(GENERIC))
    check_close([h[0] for h in polarizations], [np.asarray(h) for h in reference], 1e-3, 1e-7)


@requires_nrsur
def test_nrsur7dq4_and_simulate_fd(lal, results):
    """jit, vmap and grad of ``simulate_fd_function``, and LAL's
    ``SimInspiralFD`` with the same modes (to rounding; the epoch and
    grid: test_conversions.py)."""
    lal_module, lalsim = lal
    (length, epoch), v, ((loss, polarizations), gradient) = results["NRSur7dq4"].result()
    check_derivative(loss, gradient[0] @ v)
    assert np.all(np.isfinite(gradient))
    x = [float(v) for v in NRSUR]
    params = lal_module.CreateDict()
    modes = lalsim.SimInspiralCreateModeArray()
    for ell, emm in MODES:
        lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
    lp, lc = lalsim.SimInspiralFD(x[0] * lal_module.MSUN_SI, x[1] * lal_module.MSUN_SI, *x[2:8],
                                  x[8] * 1e6 * lal_module.PC_SI, x[9], x[10], 0.0, 0.0, 0.0, 0.0, 20.0, 1024.0, x[11],
                                  params, lalsim.NRSur7dq4)
    assert length == lp.data.length and epoch == float(lp.epoch)
    check_close([h[0] for h in polarizations], [lp.data.data, lc.data.data], 1e-10, 1e-12)


def test_frequency_to_time_domain(lal):
    """``SimInspiralTD`` of a frequency-domain model, given LAL's
    IMRPhenomXPNR: LAL's length, epoch and values."""
    from test_conversions import TD_CASES, test_time_domain_conversion_matches_lal

    test_time_domain_conversion_matches_lal(lal, *TD_CASES[2])


def test_match():
    """jit, vmap and grad of the polarization mismatch (the time and phase
    maximised by FFT, then refined) against a central difference."""
    from test_match import FREQUENCIES as F
    from test_match import PSD, chirp, shifted

    from waveformx.match import polarization_mismatch

    signal = shifted(chirp(), 0.0377, 0.4)
    cross = -1j * chirp(1e-3)
    loss = lambda theta: polarization_mismatch(signal, chirp(theta), cross, F, PSD)  # noqa: E731
    thetas = jnp.asarray([3e-4, 3e-4 + 1e-6, 3e-4 - 1e-6, 0.0])
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(loss)))(thetas)
    assert float(values[3]) < float(values[0]) < 1
    np.testing.assert_allclose(gradients[0], (values[1] - values[2]) / 2e-6, rtol=1e-5)
    # a waveform matches itself, shifted
    assert float(polarization_mismatch(shifted(chirp(), 0.05, 0.0), chirp(), 0 * chirp(), F, PSD)) < 1e-12
