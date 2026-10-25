# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR: the J-frame modes of ``SimIMRPhenomXPHMFrequencySequenceOneMode``
(with the flags of IMRPhenomXPNR) and of ``SimInspiralChooseFDModes``, and
:meth:`MODEL.validate`.

Without a mode array LAL sets the binary up again for every J-frame mode,
with a mode range depending on the mode, as the model does. The references
are LAL with the parity fix of lalsuite !2593: the modes stored by
waveformrs (``tests/data/imrphenomxpnr_parityfix_sequence_modes.json``) and
waveformrs itself, which reproduces LAL to rounding. Where LAL uses the
SpinTaylor angles, the modes agree to the tolerances of the polarizations
(see test_xpnr.py).
"""

import functools
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformx import Binary, WaveformOptions
from waveformx.exceptions import InvalidParameterError
from waveformx.models.xpnr import MODEL

DATA = Path(__file__).parent / "data"
SEQUENCE = json.loads((DATA / "imrphenomxpnr_parityfix_sequence_modes.json").read_text())["cases"]
#: mismatch of each J-frame mode
TOLERANCE = 1e-4
#: the sequence cases for which LAL leaves the SpinTaylor angles (in-plane
#: spins of 5e-8: plain IMRPhenomXPHM with the NNLO angles)
FALLBACKS = {1}

# waveformrs's imrphenomxpnr_modes suite (SimInspiralChooseFDModes): the
# generic binary and a co-precessing mode array
FD_CASES = [
    dict(mass_1=36.0, mass_2=29.0, spin_1=[0.4, 0.2, 0.3], spin_2=[-0.2, 0.3, -0.1], luminosity_distance=800.0,
         inclination=0.7, phi_ref=0.9, f_min=20.0, f_ref=25.0, delta_f=1.0, f_max=512.0),
    dict(mass_1=40.0, mass_2=12.0, spin_1=[0.5, -0.2, 0.3], spin_2=[0.1, 0.6, -0.2], luminosity_distance=800.0,
         inclination=0.7, phi_ref=0.9, f_min=15.0, f_ref=25.0, delta_f=1.0, f_max=0.0,
         mode_array=[[2, 2], [2, -2], [2, 1], [2, -1], [3, 3], [3, -3]]),
]


def binary_of(case):
    return Binary(
        mass_1=jnp.float64(case["mass_1"]), mass_2=jnp.float64(case["mass_2"]),
        spin_1=jnp.asarray(case["spin_1"], float), spin_2=jnp.asarray(case["spin_2"], float),
        luminosity_distance=jnp.float64(case["luminosity_distance"]), inclination=jnp.float64(case["inclination"]),
        phi_ref=jnp.float64(case["phi_ref"]), f_ref=jnp.float64(case["f_ref"]),
    )


def options_of(case):
    modes = case.get("mode_array")
    return WaveformOptions(mode_array=None if modes is None else [tuple(m) for m in modes], scaled_strain=False)


@functools.lru_cache(maxsize=None)
def jax_sequence_modes(index):
    case = SEQUENCE[index]
    return MODEL.modes(binary_of(case), np.asarray(case["frequencies"]), options_of(case))


def check_mode(case, result, ell, emm, positive, negative):
    n = len(case["frequencies"])
    h = np.asarray(result.modes[(ell, emm)])
    test = np.concatenate([h[n:], h[:n][::-1]])
    error, mismatch = compare(test, np.concatenate([positive, negative]))
    assert mismatch < TOLERANCE, ((ell, emm), error, mismatch)


@pytest.mark.parametrize("index", range(len(SEQUENCE)))
def test_sequence_modes_close_to_lalsimulation(index):
    case = SEQUENCE[index]
    result = jax_sequence_modes(index)
    assert sorted(result.modes) == sorted((m["ell"], m["emm"]) for m in case["modes"])
    np.testing.assert_array_equal(result.frequencies, np.concatenate([-np.asarray(case["frequencies"])[::-1],
                                                                      case["frequencies"]]))
    if index in FALLBACKS:
        assert all(np.all(np.isfinite(np.asarray(h))) for h in result.modes.values())
        return
    for mode in case["modes"]:
        positive = np.asarray(mode["positive_real"]) + 1j * np.asarray(mode["positive_imag"])
        negative = np.asarray(mode["negative_real"]) + 1j * np.asarray(mode["negative_imag"])
        check_mode(case, result, mode["ell"], mode["emm"], positive, negative)


@pytest.mark.parametrize("index", range(len(FD_CASES)))
def test_choose_fd_modes_close_to_lalsimulation(index):
    rs = pytest.importorskip("waveformrs")
    case = FD_CASES[index]
    result = MODEL.choose_fd_modes(binary_of(case), case["delta_f"], case["f_min"], case["f_max"], options_of(case))
    reference = rs.choose_fd_modes(
        "IMRPhenomXPNR", case["delta_f"], case["f_min"], case["f_max"], mass_1=case["mass_1"],
        mass_2=case["mass_2"], spin_1=tuple(case["spin_1"]), spin_2=tuple(case["spin_2"]),
        luminosity_distance=case["luminosity_distance"], inclination=case["inclination"], phi_ref=case["phi_ref"],
        f_ref=case["f_ref"], mode_array=case.get("mode_array") and [tuple(m) for m in case["mode_array"]],
        pnr_parity_fix=True,
    )
    assert sorted(result.modes) == sorted(reference.modes)
    np.testing.assert_allclose(result.frequencies, np.asarray(reference.frequencies))
    for mode, h in result.modes.items():
        h, expected = np.asarray(h), np.asarray(reference.modes[mode])
        assert h.shape == expected.shape
        error, mismatch = compare(h, expected)
        assert mismatch < TOLERANCE, (mode, error, mismatch)
    half = (len(result.frequencies) - 1) // 2
    np.testing.assert_allclose(result.frequencies[half:], np.arange(half + 1) * case["delta_f"])


# validate: LAL's checks

GENERIC = dict(mass_1=36.0, mass_2=29.0, spin_1=[0.4, 0.2, 0.3], spin_2=[-0.2, 0.3, -0.1], luminosity_distance=500.0,
               inclination=0.7, phi_ref=1.2, f_ref=20.0)
FREQUENCIES = np.geomspace(20.0, 1024.0, 64)


def validate(case, **options):
    MODEL.validate(binary_of(case), FREQUENCIES, WaveformOptions(**options))


def test_validate_accepts_generic_and_nearly_aligned_binaries():
    """Also where LAL leaves the SpinTaylor angles for the NNLO angles
    (in-plane spins below 1e-6 or 1e-7)."""
    validate(GENERIC)
    for spin in ([5e-8, 0.0, 0.5], [5e-7, 0.0, 0.5]):
        validate(dict(GENERIC, spin_1=spin, spin_2=[0.0, 0.0, -0.3]))


def test_validate_rejects_lal_errors():
    with pytest.raises(InvalidParameterError, match="f_ref"):
        validate(dict(GENERIC, f_ref=10.0))
    with pytest.raises(InvalidParameterError, match="mass ratio"):
        validate(dict(GENERIC, mass_1=1500.0, mass_2=1.0))
    with pytest.raises(InvalidParameterError, match="unavailable mode"):
        validate(GENERIC, mode_array=[(5, 5)])
