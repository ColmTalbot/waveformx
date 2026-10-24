# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The interface shared by the models: options, arguments, errors and the
spherical harmonics."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf

BINARY = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1),
              luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0)


def test_x64_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_approximants():
    assert wf.approximants() == ["IMRPhenomXAS", "IMRPhenomXPNR", "NRSur7dq4"]
    assert wf.approximants("frequency") == ["IMRPhenomXAS", "IMRPhenomXPNR"]
    assert wf.approximants("time") == ["NRSur7dq4"]
    assert wf.is_frequency_domain("IMRPhenomXPNR") and not wf.is_frequency_domain("NRSur7dq4")
    assert wf.is_precessing("NRSur7dq4") and not wf.is_precessing("IMRPhenomXAS")
    with pytest.raises(wf.UnknownApproximantError):
        wf.is_precessing("IMRPhenomXPHM")
    with pytest.raises(ValueError):
        wf.polarizations("TaylorF2", [20.0, 30.0], mass_1=1.4, mass_2=1.4)


def test_options():
    options = wf.WaveformOptions()
    assert options.pnr_parity_fix and not options.version_fallback and options.mode_array is None
    assert not wf.WaveformOptions.lalsimulation_defaults().pnr_parity_fix
    assert wf.WaveformOptions(precession_version="MSA").precession_version == 223
    assert wf.WaveformOptions(mode_array=[[2, 2], (3, -3)]).mode_array == ((2, 2), (3, -3))
    assert hash(options) == hash(wf.WaveformOptions())
    assert pickle.loads(pickle.dumps(options)) == options
    assert options.scaled_strain and not wf.WaveformOptions.lalsimulation_defaults().scaled_strain
    assert options.replace(pnr_parity_fix=False, scaled_strain=False) == (
        wf.WaveformOptions.lalsimulation_defaults().replace(precession_version=320)
    )
    assert wf.WaveformOptions(**options.as_dict()) == options
    with pytest.raises(TypeError):
        wf.WaveformOptions(pnr_parity_fix="yes")
    with pytest.raises(TypeError):
        options.replace(unknown=1)
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
