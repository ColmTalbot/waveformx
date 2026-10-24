# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Milestone 5 of IMRPhenomXPNR: the J-frame modes of
``SimIMRPhenomXPHMFrequencySequenceOneMode`` (with the flags of
IMRPhenomXPNR) and of ``SimInspiralChooseFDModes``, the version fallbacks
and :meth:`MODEL.validate`.

Without a mode array LAL sets the binary up again for every J-frame mode,
with a mode range depending on the mode (and integrates the SpinTaylor
equations again where the grid of the PNR interpolants starts below the
first integration). The ``l = 3`` modes inherit the rounding noise of the
(3,2) co-precessing mode (see test_xpnr.py); the others agree to ``~1e-9``.
"""

import functools
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformjax import Binary, WaveformOptions
from waveformjax.exceptions import InvalidParameterError, PrecessionVersionUnavailableError
from waveformjax.models.xpnr import MODEL

DATA = Path(__file__).parent / "data"
SEQUENCE = json.loads((DATA / "imrphenomxpnr_parityfix_sequence_modes.json").read_text())["cases"]
#: (max relative error, mismatch) of each J-frame mode; the l = 3 modes when
#: the (3,2) mode is among the co-precessing modes
TOLERANCE = (1e-7, 1e-12)
TOLERANCE_32 = (1e-3, 1e-6)

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


def options_of(case, parity_fix):
    modes = case.get("mode_array")
    return WaveformOptions(mode_array=None if modes is None else [tuple(m) for m in modes], pnr_parity_fix=parity_fix,
                           version_fallback=True, scaled_strain=False)


def tolerance_of(case, ell):
    modes = case.get("mode_array")
    with_32 = modes is None or [3, 2] in modes or [3, -2] in modes
    return TOLERANCE_32 if ell == 3 and with_32 else TOLERANCE


def lal_params(lal, case, jframe=False):
    lal_module, lalsim = lal
    params = lal_module.CreateDict()
    for key, value in case.get("lal_flags", {}).items():
        getattr(lalsim, f"SimInspiralWaveformParamsInsert{key}")(params, value)
    if case.get("mode_array") is not None:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, emm in case["mode_array"]:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
        if jframe:
            # SimInspiralChooseFDModes: the J-frame modes of the l of the
            # co-precessing modes
            modes = lalsim.SimInspiralCreateModeArray()
            for ell in sorted({ell for ell, _ in case["mode_array"]}):
                for emm in range(-ell, ell + 1):
                    lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
            lalsim.SimInspiralWaveformParamsInsertModeArrayJframe(params, modes)
    return params


@functools.lru_cache(maxsize=None)
def jax_sequence_modes(index, parity_fix):
    case = SEQUENCE[index]
    return MODEL.modes(binary_of(case), np.asarray(case["frequencies"]), options_of(case, parity_fix))


def check_mode(case, result, ell, emm, positive, negative):
    n = len(case["frequencies"])
    h = np.asarray(result.modes[(ell, emm)])
    test = np.concatenate([h[n:], h[:n][::-1]])
    error, mismatch = compare(test, np.concatenate([positive, negative]))
    tolerance = tolerance_of(case, ell)
    assert error < tolerance[0], ((ell, emm), error, mismatch)
    assert mismatch < tolerance[1], ((ell, emm), error, mismatch)


@pytest.mark.parametrize("index", range(len(SEQUENCE)))
def test_sequence_modes_parity_fix(index):
    case = SEQUENCE[index]
    result = jax_sequence_modes(index, True)
    assert sorted(result.modes) == sorted((m["ell"], m["emm"]) for m in case["modes"])
    np.testing.assert_array_equal(result.frequencies, np.concatenate([-np.asarray(case["frequencies"])[::-1],
                                                                      case["frequencies"]]))
    for mode in case["modes"]:
        positive = np.asarray(mode["positive_real"]) + 1j * np.asarray(mode["positive_imag"])
        negative = np.asarray(mode["negative_real"]) + 1j * np.asarray(mode["negative_imag"])
        check_mode(case, result, mode["ell"], mode["emm"], positive, negative)


@pytest.mark.parametrize("index", range(len(SEQUENCE)))
def test_sequence_modes_match_lalsimulation(lal, index):
    lal_module, lalsim = lal
    case = SEQUENCE[index]
    result = jax_sequence_modes(index, False)
    frequencies = np.asarray(case["frequencies"])
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = frequencies
    for ell, emm in result.modes:
        positive, negative = lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode(
            vector, ell, emm, case["mass_1"] * lal_module.MSUN_SI, case["mass_2"] * lal_module.MSUN_SI,
            *case["spin_1"], *case["spin_2"], case["luminosity_distance"] * 1e6 * lal_module.PC_SI,
            case["inclination"], case["phi_ref"], case["f_ref"], lal_params(lal, case),
        )
        check_mode(case, result, ell, emm, positive.data.data, negative.data.data)


@pytest.mark.parametrize("index", range(len(FD_CASES)))
def test_choose_fd_modes_match_lalsimulation(lal, index):
    lal_module, lalsim = lal
    case = dict(FD_CASES[index], lal_flags=dict(PhenomXHMThresholdMband=0.0, PhenomXPHMThresholdMband=0.0))
    result = MODEL.choose_fd_modes(binary_of(case), case["delta_f"], case["f_min"], case["f_max"],
                                   options_of(case, False))
    node = lalsim.SimInspiralChooseFDModes(
        case["mass_1"] * lal_module.MSUN_SI, case["mass_2"] * lal_module.MSUN_SI, *case["spin_1"], *case["spin_2"],
        case["delta_f"], case["f_min"], case["f_max"], case["f_ref"], case["phi_ref"],
        case["luminosity_distance"] * 1e6 * lal_module.PC_SI, case["inclination"], lal_params(lal, case, jframe=True),
        lalsim.IMRPhenomXPNR,
    )
    count = 0
    while node is not None:
        reference = node.mode.data.data
        h = np.asarray(result.modes[(node.l, node.m)])
        assert h.shape == reference.shape
        error, mismatch = compare(h, reference)
        tolerance = tolerance_of(case, node.l)
        assert error < tolerance[0], ((node.l, node.m), error, mismatch)
        assert mismatch < tolerance[1], ((node.l, node.m), error, mismatch)
        count += 1
        node = node.next
    assert count == len(result.modes)
    half = (len(result.frequencies) - 1) // 2
    np.testing.assert_allclose(result.frequencies[half:], np.arange(half + 1) * case["delta_f"])


# validate: LAL's checks and the fallbacks of the precession prescription

GENERIC = dict(mass_1=36.0, mass_2=29.0, spin_1=[0.4, 0.2, 0.3], spin_2=[-0.2, 0.3, -0.1], luminosity_distance=500.0,
               inclination=0.7, phi_ref=1.2, f_ref=20.0)
FREQUENCIES = np.geomspace(20.0, 1024.0, 64)


def validate(case, **options):
    MODEL.validate(binary_of(case), FREQUENCIES, WaveformOptions(**options))


def test_validate_accepts_generic_binaries():
    validate(GENERIC)
    validate(GENERIC, version_fallback=True)


@pytest.mark.parametrize("spin, reason", [
    ([5e-8, 0.0, 0.5], "1e-7"),
    ([5e-7, 0.0, 0.5], "1e-6"),
])
def test_validate_without_fallbacks(spin, reason):
    """Nearly aligned binaries: LAL leaves the SpinTaylor angles (version
    330) for the NNLO angles."""
    case = dict(GENERIC, spin_1=spin, spin_2=[0.0, 0.0, -0.3])
    validate(case, version_fallback=True)
    with pytest.raises(PrecessionVersionUnavailableError, match=reason):
        validate(case)


def test_validate_rejects_lal_errors():
    with pytest.raises(InvalidParameterError, match="f_ref"):
        validate(dict(GENERIC, f_ref=10.0))
    with pytest.raises(InvalidParameterError, match="mass ratio"):
        validate(dict(GENERIC, mass_1=1500.0, mass_2=1.0))
    with pytest.raises(InvalidParameterError, match="unavailable mode"):
        validate(GENERIC, mode_array=[(5, 5)])
