# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The NNLO and MSA prescriptions of IMRPhenomXPNR (``IMRPhenomXPNR_NNLO``,
``IMRPhenomXPNR_MSA``) against ``SimInspiralChooseFDWaveformSequence`` of
LALSimulation 6.2.0 with the parity fix of lalsuite !2593 at
``PhenomXPrecVersion`` 102 and 223, on the binaries of waveformrs's
``imrphenomxpnr`` suite (``tests/data/imrphenomxpnr_prescriptions.json``,
written by ``scripts/xpnr_prescriptions_reference.py``).

The two agree to mismatches of ``1e-16`` to ``1e-10`` for most binaries
(``TOLERANCE``). The exceptions are LAL's discrete choices, which the model
makes continuous (``DELIBERATE``):

* in-plane spins below ``1e-6`` (LAL's 223 takes the NNLO angles there,
  and below ``1e-7`` both turn the PNR angles off; LAL's 223 with aligned
  spins is not the limit of its waveforms, up to a mismatch of ``0.09``
  from its 102 and 330), and the opening angle of LAL's PNR ``beta`` at
  small in-plane spins (``~3e-7``);
* beyond ``q = 12`` (the PNR transition region);
* the connection frequencies of the merger-ringdown ``beta`` where LAL's
  arithmetic gives NaN and it drops the connection (complex extrema of
  ``beta``, binary 24): the model takes their finite limit, which moves the
  frequency map of the higher modes;
* with the MSA angles, the two-spin construction for every binary (LAL's
  for ``|chi2| >= 1e-3``) and the ``beta`` of a constant spin magnitude
  where the roots of its cubic are not resolved (equal masses).

Where LAL's MSA system fails (binary 21) LAL takes the NNLO angles; the
MSA prescription is NaN there, and the public functions raise
``ModelError``.
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
import waveformx as wf
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL_MSA, MODEL_NNLO

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "imrphenomxpnr_parityfix.json").read_text())["cases"]
REFERENCE = json.loads((DATA / "imrphenomxpnr_prescriptions.json").read_text())["versions"]
MODELS = {102: MODEL_NNLO, 223: MODEL_MSA}
TOLERANCE = 1e-6
#: the binaries where the model departs from LAL deliberately (see the
#: module's docstring): their bounds, with ~2x margin over the measured
DELIBERATE = {
    102: {15: 3e-3, 24: 1e-3},
    223: {7: 2e-3, 8: 0.2, 9: 5e-3, 15: 1.5e-3, 24: 1.2e-5},
}
#: the MSA system fails: LAL takes the NNLO angles
MSA_FAILS = 21


def binary_of(case):
    return Binary(
        mass_1=jnp.float64(case["mass_1"]), mass_2=jnp.float64(case["mass_2"]),
        spin_1=jnp.asarray(case["spin_1"], float), spin_2=jnp.asarray(case["spin_2"], float),
        luminosity_distance=jnp.float64(case["luminosity_distance"]), inclination=jnp.float64(case["inclination"]),
        phi_ref=jnp.float64(case["phi_ref"]), f_ref=jnp.float64(case["f_ref"]),
    )


def mode_array_of(case):
    modes = case.get("mode_array")
    return None if modes is None else tuple(tuple(m) for m in modes)


@functools.lru_cache(maxsize=None)
def compiled(version, mode_array):
    options = WaveformOptions(mode_array=mode_array, scaled_strain=False)
    return jax.jit(lambda binary, frequencies: MODELS[version].polarizations(binary, frequencies, options))


def polarizations(version, case):
    hp, hc = compiled(version, mode_array_of(case))(binary_of(case), jnp.asarray(case["frequencies"]))
    return np.asarray(hp), np.asarray(hc)


def stored(version, index):
    entry = REFERENCE[str(version)][index]
    return [np.asarray(entry[f"{name}_real"]) + 1j * np.asarray(entry[f"{name}_imag"]) for name in ("hplus", "hcross")]


@pytest.mark.parametrize("index", range(len(CASES)))
@pytest.mark.parametrize("version", [102, 223])
def test_polarizations_close_to_lalsimulation(version, index):
    test = polarizations(version, CASES[index])
    if version == 223 and index == MSA_FAILS:
        assert all(np.all(np.isnan(h)) for h in test)
        return
    tolerance = DELIBERATE[version].get(index, TOLERANCE)
    for a, b in zip(test, stored(version, index)):
        assert np.all(np.isfinite(a))
        error, mismatch = compare(a, b)
        assert mismatch < tolerance, (error, mismatch)


def keywords_of(case):
    return {key: case[key] for key in ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination",
                                       "phi_ref", "f_ref")}


def test_msa_failure_raises():
    case = CASES[MSA_FAILS]
    with pytest.raises(wf.ModelError, match="MSA"):
        wf.polarizations("IMRPhenomXPNR_MSA", np.asarray(case["frequencies"]), **keywords_of(case))


@pytest.mark.parametrize("approximant, version", [("IMRPhenomXPNR_NNLO", 102), ("IMRPhenomXPNR_MSA", 223)])
def test_public_functions(approximant, version):
    """The public functions with the approximant names: the model's
    polarizations, and ``polarizations_function`` under ``jax.jit``."""
    case = CASES[0]
    f = np.asarray(case["frequencies"])
    public = wf.polarizations(approximant, f, scaled_strain=False, **keywords_of(case))
    for a, b in zip(public, polarizations(version, case)):
        np.testing.assert_allclose(np.asarray(a), b, rtol=0, atol=1e-12 * np.max(np.abs(b)))
    function = jax.jit(wf.polarizations_function(approximant, scaled_strain=False))
    for a, b in zip(function(binary_of(case), jnp.asarray(f)), polarizations(version, case)):
        np.testing.assert_allclose(np.asarray(a), b, rtol=0, atol=1e-12 * np.max(np.abs(b)))


def test_simulate_td():
    """The conversions with a prescription: ``simulate_td`` and the pure
    ``simulate_td_function`` agree, and differ from the SpinTaylor angles."""
    keywords = keywords_of(CASES[0])
    dominant = ((2, 2), (2, -2))
    result = wf.simulate_td("IMRPhenomXPNR_NNLO", 1 / 1024, 25.0, mode_array=dominant, scaled_strain=False,
                            **keywords)
    hp = np.asarray(result.hplus)
    assert np.all(np.isfinite(hp))
    function = wf.simulate_td_function("IMRPhenomXPNR_NNLO", 1 / 1024, 25.0, mode_array=dominant,
                                       scaled_strain=False, **keywords)
    pure = np.asarray(jax.jit(function)(binary_of(CASES[0]))[0])
    np.testing.assert_allclose(pure, hp, rtol=0, atol=1e-10 * np.max(np.abs(hp)))
    spintaylor = np.asarray(wf.simulate_td("IMRPhenomXPNR", 1 / 1024, 25.0, mode_array=dominant,
                                           scaled_strain=False, **keywords).hplus)
    assert spintaylor.shape == hp.shape and np.max(np.abs(spintaylor - hp)) > 1e-3 * np.max(np.abs(hp))


def test_reference_converters():
    """The converters between reference conventions are IMRPhenomXPNR's
    (the spins evolved with the SpinTaylor equations) for every
    prescription."""
    binary = binary_of(CASES[0])
    expected = wf.to_mass_scaled_reference(binary, approximant="IMRPhenomXPNR")
    for approximant in ("IMRPhenomXPNR_NNLO", "IMRPhenomXPNR_MSA"):
        converted = wf.to_mass_scaled_reference(binary, approximant=approximant)
        for a, b in zip(converted, expected):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
