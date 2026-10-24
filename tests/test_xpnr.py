# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Milestone 4 of IMRPhenomXPNR: the polarizations against
``SimInspiralChooseFDWaveformSequence``.

The binaries are waveformrs's ``imrphenomxpnr`` suite
(``scripts/generate_reference_data.py``): generic precessing binaries and the
edge cases of the PNR machinery (in-plane spins below ``1e-7`` and
``1e-6``, SpinTaylor integrations that start from ``f_ref`` or end early,
the transition and outer regions of the calibration, MSA fallbacks, equal
masses with equal spins, ...), and two restricted mode arrays.

* ``pnr_parity_fix=False`` is compared with the installed LALSimulation
  (6.2.0);
* the default, with the fix of lalsuite !2593, with the polarizations of
  6.2.0 patched with the fix, stored by waveformrs in
  ``tests/data/imrphenomxpnr_parityfix.json``.

All co-precessing modes agree to ``~1e-5`` (mismatch ``~1e-10``): the
(3,2) mode of IMRPhenomXHM fixes its intermediate phase with a finite
difference of a phase of ``~1e4`` rad, which amplifies rounding to that
level (LAL's own builds differ by up to ``2e-2``, see test_xpnr_hm.py).
The (2,2) mode alone, with the PNR angles, the tuned co-precessing mode and
the antisymmetric part, agrees to ``~1e-9`` (LAL's own integration error of
the SpinTaylor equations below ``f_ref``).
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformjax import Binary, WaveformOptions
from waveformjax.models.xpnr import MODEL

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "imrphenomxpnr_parityfix.json").read_text())["cases"]
DOMINANT = ((2, 2), (2, -2))
#: (max relative error, mismatch) with all co-precessing modes and with the
#: (2,2) mode alone
TOLERANCE = (1e-4, 1e-9)
TOLERANCE_22 = (1e-7, 1e-14)


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
def compiled(options):
    return jax.jit(lambda binary, frequencies: MODEL.polarizations(binary, frequencies, options))


def jax_polarizations(case, mode_array, parity_fix):
    options = WaveformOptions(mode_array=mode_array, pnr_parity_fix=parity_fix, version_fallback=True)
    hp, hc = compiled(options)(binary_of(case), jnp.asarray(case["frequencies"]))
    return np.asarray(hp), np.asarray(hc)


def lal_polarizations(lal, case, mode_array):
    lal_module, lalsim = lal
    params = lal_module.CreateDict()
    if mode_array is not None:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, emm in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
    frequencies = np.asarray(case["frequencies"])
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = frequencies
    hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
        case["phi_ref"], case["mass_1"] * lal_module.MSUN_SI, case["mass_2"] * lal_module.MSUN_SI,
        *case["spin_1"], *case["spin_2"], case["f_ref"], case["luminosity_distance"] * 1e6 * lal_module.PC_SI,
        case["inclination"], params, lalsim.IMRPhenomXPNR, vector,
    )
    return hp.data.data, hc.data.data


def check(test, reference, tolerance):
    for a, b in zip(test, reference):
        error, mismatch = compare(a, b)
        assert error < tolerance[0], (error, mismatch)
        assert mismatch < tolerance[1], (error, mismatch)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_polarizations_match_lalsimulation(lal, index):
    case = CASES[index]
    mode_array = mode_array_of(case)
    check(jax_polarizations(case, mode_array, False), lal_polarizations(lal, case, mode_array), TOLERANCE)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_parity_fix_matches_patched_lalsimulation(index):
    case = CASES[index]
    reference = [np.asarray(case[f"{name}_real"]) + 1j * np.asarray(case[f"{name}_imag"]) for name in ("hplus", "hcross")]
    check(jax_polarizations(case, mode_array_of(case), True), reference, TOLERANCE)


# the binaries without a mode array of their own, but for (325, 156), where
# LAL fails with the (2,2) mode alone (tested below)
DOMINANT_CASES = [i for i, case in enumerate(CASES) if case.get("mode_array") is None and i != 21]


@pytest.mark.parametrize("index", DOMINANT_CASES)
def test_dominant_mode_matches_lalsimulation(lal, index):
    case = CASES[index]
    check(jax_polarizations(case, DOMINANT, False), lal_polarizations(lal, case, DOMINANT), TOLERANCE_22)


def test_failures_give_nan(lal):
    """LAL cannot evaluate the PNR beta of (325, 156) at f_ref when the grid
    of the interpolants holds the (2,2) mode alone."""
    case = CASES[21]
    with pytest.raises(RuntimeError):
        lal_polarizations(lal, case, DOMINANT)
    hp, hc = jax_polarizations(case, DOMINANT, False)
    assert np.all(np.isnan(hp)) and np.all(np.isnan(hc))


def mirror(case):
    """The mirror image in the orbital plane at f_ref (in-plane spins
    reversed), seen from the mirrored direction."""
    flip = lambda s: [-s[0], -s[1], s[2]]  # noqa: E731
    return dict(case, spin_1=flip(case["spin_1"]), spin_2=flip(case["spin_2"]), inclination=np.pi - case["inclination"])


@pytest.mark.parametrize("index", [0, 1, 4])
def test_parity_fix_restores_the_mirror_symmetry(index):
    case = CASES[index]
    for fix, bound in ((True, lambda v: v < 1e-11), (False, lambda v: v > 1e-4)):
        hp, hc = jax_polarizations(case, None, fix)
        hp_m, hc_m = jax_polarizations(mirror(case), None, fix)
        scale = max(np.abs(hp).max(), np.abs(hc).max())
        violation = max(np.abs(hp_m - hp).max(), np.abs(hc_m + hc).max()) / scale
        assert bound(violation), (fix, violation)


def test_cutoff_and_reference_frequency(lal):
    """Zero above f_cut = 0.3 / M; NaN when f_ref lies outside the
    frequencies (LAL's tuned angles require f_min <= f_ref <= f_max)."""
    case = dict(CASES[0], frequencies=list(np.geomspace(20.0, 4000.0, 256)))
    hp, hc = jax_polarizations(case, None, True)
    f = np.asarray(case["frequencies"])
    m_sec = (case["mass_1"] + case["mass_2"]) * 4.925490947641267e-06
    assert np.all(hp[f > 0.3 / m_sec] == 0.0) and np.all(np.isfinite(hp))
    hp, hc = jax_polarizations(dict(case, f_ref=10.0), None, True)
    assert np.all(np.isnan(hp))
