# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The interface shared by the models: options, arguments, errors and the
spherical harmonics."""

import pickle
import sys
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformx as wf

BINARY = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1),
              luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0)


def test_x64_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_approximants():
    assert wf.approximants() == ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomXPNR",
                                 "NRSur7dq4"]
    assert wf.approximants("frequency") == ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM",
                                            "IMRPhenomXPNR"]
    assert wf.approximants("time") == ["NRSur7dq4"]
    assert wf.is_frequency_domain("IMRPhenomXPNR") and not wf.is_frequency_domain("NRSur7dq4")
    assert wf.is_precessing("NRSur7dq4") and not wf.is_precessing("IMRPhenomXAS")
    assert not wf.is_precessing("IMRPhenomXHM")
    with pytest.raises(wf.UnknownApproximantError):
        wf.is_precessing("IMRPhenomXPHM")
    with pytest.raises(ValueError):
        wf.polarizations("SEOBNRv4_ROM", [20.0, 30.0], mass_1=1.4, mass_2=1.4)


def test_options():
    options = wf.WaveformOptions()
    assert options.mode_array is None and not options.extended_phase_grid and options.reference_mf is None
    assert wf.WaveformOptions(precession_version="MSA").precession_version == 223
    assert wf.WaveformOptions(mode_array=[[2, 2], (3, -3)]).mode_array == ((2, 2), (3, -3))
    assert hash(options) == hash(wf.WaveformOptions())
    assert pickle.loads(pickle.dumps(options)) == options
    assert options.scaled_strain and not wf.WaveformOptions.lalsimulation_defaults().scaled_strain
    assert options.replace(scaled_strain=False) == (
        wf.WaveformOptions.lalsimulation_defaults().replace(precession_version=320)
    )
    assert wf.WaveformOptions(**options.as_dict()) == options
    with pytest.raises(TypeError):
        wf.WaveformOptions(scaled_strain="yes")
    with pytest.raises(TypeError):
        options.replace(unknown=1)
    with pytest.raises(TypeError):
        # the options of the LAL-faithful version are gone
        options.replace(version_fallback=True)
    with pytest.raises(wf.InvalidParameterError):
        wf.WaveformOptions(precession_version=101)


def test_arguments_are_checked():
    f = np.linspace(20.0, 100.0, 5)
    with pytest.raises(TypeError, match="mass_2"):
        wf.polarizations("IMRPhenomXPNR", f, mass_1=30.0)
    with pytest.raises(TypeError, match="unexpected"):
        wf.polarizations("IMRPhenomXPNR", f, mass_1=30.0, mass_2=20.0, mass_3=1.0)
    with pytest.raises(wf.InvalidParameterError):
        wf.polarizations("IMRPhenomXPNR", f, mass_1=-30.0, mass_2=20.0)
    with pytest.raises(wf.InvalidParameterError):
        wf.polarizations("IMRPhenomXPNR", f, mass_1=30.0, mass_2=20.0, spin_1=(0.9, 0.9, 0.0))
    with pytest.raises(wf.InvalidParameterError):
        wf.polarizations("IMRPhenomXPNR", [30.0, 20.0], mass_1=30.0, mass_2=20.0)
    with pytest.raises(wf.UnsupportedError):
        wf.td_polarizations("IMRPhenomXPNR", 1 / 4096, 20.0, mass_1=30.0, mass_2=20.0)
    with pytest.raises(wf.InvalidParameterError):
        wf.simulate_td("IMRPhenomXPNR", 1 / 4096, 0.0, mass_1=30.0, mass_2=20.0)


def test_parameter_table():
    n = 4
    parameters = dict(mass_1=np.full(n, 30.0), mass_2=np.full(n, 20.0), spin_1=np.zeros((n, 3)),
                      spin_2z=np.full(n, 0.1), f_ref=np.full(n, 20.0))
    table = wf.parameter_table(parameters)
    assert table.shape == (n, 12)
    assert np.all(table[:, 7] == 0.1) and np.all(table[:, 8] == 1.0)
    structured = np.zeros(n, dtype=[(name, float) for name in wf.COLUMNS])
    structured["mass_1"], structured["mass_2"] = 30.0, 20.0
    assert wf.parameter_table(structured).shape == (n, 12)
    with pytest.raises(TypeError):
        wf.parameter_table(dict(mass_1=np.ones(n)))


def test_binary_is_a_pytree():
    binary = wf.Binary(30.0, 20.0, jnp.zeros(3), jnp.zeros(3))
    leaves = jax.tree_util.tree_leaves(binary)
    assert len(leaves) == 8
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.asarray(x)] * 2), binary)
    assert batched.spin_1.shape == (2, 3)


@pytest.mark.parametrize("ell", [2, 3, 4])
def test_spin_weighted_spherical_harmonics(lal, ell):
    lal_module, _ = lal
    theta, phi = 0.7, 1.9
    for m in range(-ell, ell + 1):
        ours = complex(wf.spin_weighted_spherical_harmonic(theta, phi, ell, m))
        theirs = lal_module.SpinWeightedSphericalHarmonic(theta, phi, -2, ell, m)
        assert abs(ours - theirs) < 1e-14 * max(1.0, abs(theirs)), (ell, m)


def test_reference_converters():
    """The converters dispatch to each model's (IMRPhenomXPNR's by
    default), and refuse unknown approximants."""
    import jax.numpy as jnp

    from waveformx.models.xas import reference as xas_reference
    from waveformx.models.xpnr import reference as xpnr_reference

    binary = wf.Binary(*(jnp.asarray(v, jnp.float64) for v in wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3),
                                                                          (0.0, 0.0, -0.4), 500.0, 1.0, 1.2, 30.0)))
    for name in ("IMRPhenomXAS", "IMRPhenomXHM"):
        ours = wf.to_mass_scaled_reference(binary, 0.01, approximant=name, f_min=20.0)
        assert float(ours.phi_ref) == float(xas_reference.to_mass_scaled_reference(binary, 0.01, 20.0).phi_ref)
        back = wf.to_physical_reference(ours, 30.0, 0.01, approximant=name)
        assert float(back.phi_ref) == pytest.approx(1.2, abs=1e-9)
    assert float(wf.to_mass_scaled_reference(binary).phi_ref) == float(
        xpnr_reference.to_mass_scaled_reference(binary).phi_ref)
    # aligned spins: IMRPhenomXPNR's converters move phi_ref as IMRPhenomXAS's
    xpnr = float(wf.to_mass_scaled_reference(binary, 0.01).phi_ref)
    xas = float(wf.to_mass_scaled_reference(binary, 0.01, approximant="IMRPhenomXAS").phi_ref)
    assert np.cos(xpnr - xas) == pytest.approx(1.0, abs=1e-8)
    with pytest.raises(wf.UnknownApproximantError):
        wf.to_mass_scaled_reference(binary, approximant="SEOBNRv4_ROM")


def test_precision_is_per_thread():
    """The precision being traced is each thread's own: a thread tracing in
    float32 leaves another's float64 (and the reverse)."""
    from waveformx.models.xpnr import _math

    entered, checked = threading.Barrier(2), threading.Barrier(2)
    seen = {}

    def trace(dtype):
        with _math.precision(dtype):
            entered.wait()
            seen[dtype] = _math.real(), _math.complex_()
            checked.wait()

    threads = [threading.Thread(target=trace, args=(dtype,)) for dtype in (np.float32, np.float64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {np.float32: (np.float32, np.complex64), np.float64: (np.float64, np.complex128)}
    assert _math.real() == np.float64


def test_concurrent_tracing():
    """TaylorF2 traced in float32 and float64 at once, in threads, gives the
    types and values of serial calls."""
    frequencies = np.linspace(20.0, 200.0, 16)
    binary = wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.4), 500.0, 1.0, 1.2, 30.0)
    model = wf.models.get("TaylorF2")

    def evaluate(dtype):
        # a new function each time, so that every call traces the model
        function = jax.jit(lambda b, f: model.polarizations(b, f, wf.WaveformOptions()))
        b = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype), binary)
        return [np.asarray(h) for h in function(b, jnp.asarray(frequencies, dtype))]

    dtypes = [np.float32, np.float64] * 2
    results = [None] * len(dtypes)

    def run(i):
        results[i] = evaluate(dtypes[i])

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # interleave the tracing finely
    try:
        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(dtypes))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(interval)
    for dtype, concurrent in zip(dtypes, results):
        serial = evaluate(dtype)
        for a, b in zip(concurrent, serial):
            assert a.dtype == b.dtype == np.result_type(dtype, np.complex64)
            np.testing.assert_array_equal(a, b)
