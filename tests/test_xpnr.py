# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR: the polarizations against ``SimInspiralChooseFDWaveformSequence``
of LALSimulation 6.2.0 with the parity fix of lalsuite !2593.

The binaries are waveformrs's ``imrphenomxpnr`` suite: generic precessing
binaries and the edge cases of the PNR machinery (in-plane spins below
``1e-7`` and ``1e-6``, SpinTaylor integrations that start from ``f_ref`` or
end early, the transition and outer regions of the calibration, MSA
fallbacks, equal masses with equal spins, ...), and two restricted mode
arrays. LAL's polarizations are those stored by waveformrs in
``tests/data/imrphenomxpnr_parityfix.json``; the (2,2) mode alone is
compared with waveformrs, which reproduces LAL to rounding.

The model keeps LAL's physics but not its numerical procedure (see
:mod:`waveformx.models.xpnr.model`; the version that reproduces LAL's
numerics is the tag ``lal-faithful``): where LAL uses the SpinTaylor angles
the two agree to mismatches of ``1e-11`` to ``5e-5`` (closest for two
generic spins away from LAL's switches; ``2e-4`` beyond ``q = 12``, where
the angles go smoothly to LAL's outside the transition region). Where LAL
leaves them for the MSA or NNLO angles (``FALLBACKS``), the model keeps the
SpinTaylor angles and differs from LAL by up to a mismatch of ``0.3``.
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "imrphenomxpnr_parityfix.json").read_text())["cases"]
DOMINANT = ((2, 2), (2, -2))
#: the binaries for which LAL leaves the SpinTaylor angles: its integration
#: fails (MSA angles, or NNLO angles where the MSA system fails too)
FALLBACKS = {19, 21, 23, 24, 28}
#: mismatch where LAL uses the SpinTaylor angles: up to q = 12 and beyond
TOLERANCE = 1e-4
TOLERANCE_BEYOND_Q12 = 1e-3
#: the (2,2) mode alone of the two equal-mass binaries seen nearly edge-on,
#: whose cross polarization nearly vanishes: mismatches of up to 1.5e-3
#: (h_+) and 2.5e-2 (h_x), as for the extended angles at ``lal-faithful``
DOMINANT_TOLERANCE = {3: 3e-2, 18: 3e-2}


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


def tolerance_of(case):
    q = max(case["mass_1"], case["mass_2"]) / min(case["mass_1"], case["mass_2"])
    return TOLERANCE_BEYOND_Q12 if q > 12.0 else TOLERANCE


@functools.lru_cache(maxsize=None)
def compiled(options):
    return jax.jit(lambda binary, frequencies: MODEL.polarizations(binary, frequencies, options))


def jax_polarizations(case, mode_array):
    options = WaveformOptions(mode_array=mode_array, scaled_strain=False)
    hp, hc = compiled(options)(binary_of(case), jnp.asarray(case["frequencies"]))
    return np.asarray(hp), np.asarray(hc)


def stored_polarizations(case):
    return [np.asarray(case[f"{name}_real"]) + 1j * np.asarray(case[f"{name}_imag"]) for name in ("hplus", "hcross")]


def rs_polarizations(case, mode_array):
    rs = pytest.importorskip("waveformrs")
    hp, hc = rs.polarizations(
        "IMRPhenomXPNR", np.asarray(case["frequencies"]), mass_1=case["mass_1"], mass_2=case["mass_2"],
        spin_1=tuple(case["spin_1"]), spin_2=tuple(case["spin_2"]), luminosity_distance=case["luminosity_distance"],
        inclination=case["inclination"], phi_ref=case["phi_ref"], f_ref=case["f_ref"], mode_array=mode_array,
        pnr_parity_fix=True,
    )
    return np.asarray(hp), np.asarray(hc)


def check(test, reference, tolerance):
    for a, b in zip(test, reference):
        assert np.all(np.isfinite(a))
        error, mismatch = compare(a, b)
        assert mismatch < tolerance, (error, mismatch)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_polarizations_close_to_lalsimulation(index):
    case = CASES[index]
    test = jax_polarizations(case, mode_array_of(case))
    if index in FALLBACKS:
        assert all(np.all(np.isfinite(h)) for h in test)
        return
    check(test, stored_polarizations(case), tolerance_of(case))


# the binaries without a mode array of their own
DOMINANT_CASES = [i for i, case in enumerate(CASES) if case.get("mode_array") is None and i not in FALLBACKS]


@pytest.mark.parametrize("index", DOMINANT_CASES)
def test_dominant_mode_close_to_lalsimulation(index):
    case = CASES[index]
    tolerance = DOMINANT_TOLERANCE.get(index, tolerance_of(case))
    check(jax_polarizations(case, DOMINANT), rs_polarizations(case, DOMINANT), tolerance)


def mirror(case):
    """The mirror image in the orbital plane at f_ref (in-plane spins
    reversed), seen from the mirrored direction."""
    flip = lambda s: [-s[0], -s[1], s[2]]  # noqa: E731
    return dict(case, spin_1=flip(case["spin_1"]), spin_2=flip(case["spin_2"]), inclination=np.pi - case["inclination"])


@pytest.mark.parametrize("index", [0, 1, 4])
def test_mirror_symmetry(index):
    """The parity fix of lalsuite !2593 restores the mirror symmetry of the
    waveform (LAL 6.2.0 breaks it by more than 1e-4)."""
    case = CASES[index]
    hp, hc = jax_polarizations(case, None)
    hp_m, hc_m = jax_polarizations(mirror(case), None)
    scale = max(np.abs(hp).max(), np.abs(hc).max())
    violation = max(np.abs(hp_m - hp).max(), np.abs(hc_m + hc).max()) / scale
    assert violation < 1e-11


def test_cutoff_and_reference_frequency():
    """Zero above f_cut = 0.3 / M; NaN when f_ref lies outside the
    frequencies (LAL's tuned angles require f_min <= f_ref <= f_max)."""
    case = dict(CASES[0], frequencies=list(np.geomspace(20.0, 4000.0, 256)))
    hp, hc = jax_polarizations(case, None)
    f = np.asarray(case["frequencies"])
    m_sec = (case["mass_1"] + case["mass_2"]) * 4.925490947641267e-06
    assert np.all(hp[f > 0.3 / m_sec] == 0.0) and np.all(np.isfinite(hp))
    hp, hc = jax_polarizations(dict(case, f_ref=10.0), None)
    assert np.all(np.isnan(hp))
