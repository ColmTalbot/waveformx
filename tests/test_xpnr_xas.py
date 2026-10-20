# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Milestone 1 of IMRPhenomXPNR: the aligned-spin (2,2) mode IMRPhenomXAS
against ``SimIMRPhenomXASFrequencySequence`` of LALSimulation 6.2.0."""

import jax
import numpy as np
import pytest

from conftest import compare
from waveformjax.models.xpnr import xas
from waveformjax.models.xpnr._generated import qnm, utilities

# mass_1, mass_2, chi_1, chi_2, distance, phi_ref, f_min, f_ref (waveformrs's
# aligned_cases, plus the lighter body first and f_ref = 0)
CASES = [
    (30.0, 30.0, 0.0, 0.0, 400.0, 0.0, 20.0, 20.0),
    (36.0, 29.0, 0.3, -0.4, 500.0, 1.2, 20.0, 30.0),
    (60.0, 15.0, 0.8, 0.5, 1000.0, 4.0, 15.0, 20.0),
    (80.0, 10.0, -0.7, 0.2, 2000.0, 0.3, 10.0, 25.0),
    (1.6, 1.3, 0.05, 0.02, 40.0, 2.0, 30.0, 40.0),
    (8.0, 1.4, 0.9, 0.0, 200.0, 5.5, 20.0, 50.0),
    (150.0, 100.0, 0.95, 0.95, 3000.0, 3.0, 10.0, 15.0),
    (25.0, 20.0, -0.9, -0.9, 300.0, 0.7, 20.0, 20.0),
    (20.0, 45.0, 0.3, -0.4, 300.0, 0.7, 20.0, 0.0),
    (50.0, 2.0, 0.995, 0.3, 300.0, 1.7, 20.0, 25.0),
]


@pytest.mark.parametrize("case", CASES)
def test_xas_matches_lalsimulation(lal, case):
    lal_module, lalsim = lal
    m1, m2, c1, c2, distance, phi, f_min, f_ref = case
    f_max = 0.33 / ((m1 + m2) * lal_module.MTSUN_SI) * 1.05
    frequencies = np.geomspace(f_min, f_max, 1500)
    vector = lal_module.CreateREAL8Vector(len(frequencies))
    vector.data = frequencies
    reference = lalsim.SimIMRPhenomXASFrequencySequence(
        vector, m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI, c1, c2,
        distance * 1e6 * lal_module.PC_SI, phi, f_ref, None,
    ).data.data
    h22 = jax.jit(xas.frequency_sequence)(frequencies, m1, m2, c1, c2, distance, phi, f_ref)
    error, mismatch = compare(h22, reference)
    assert error < 1e-10, error
    assert mismatch < 1e-14, mismatch


def test_fits_match_lalsimulation(lal):
    _, lalsim = lal
    for eta, chi1, chi2 in [(0.25, 0.0, 0.0), (0.2, 0.6, -0.3), (0.09, -0.9, 0.95), (0.03, 0.99, 0.5)]:
        assert utilities.final_spin_2017(eta, chi1, chi2) == pytest.approx(
            lalsim.SimIMRPhenomXFinalSpin2017(eta, chi1, chi2), rel=1e-14)
        assert utilities.final_mass_2017(eta, chi1, chi2) == pytest.approx(
            lalsim.SimIMRPhenomXFinalMass2017(eta, chi1, chi2), rel=1e-14)
        assert utilities.f_meco(eta, chi1, chi2) == pytest.approx(lalsim.SimIMRPhenomXfMECO(eta, chi1, chi2), rel=1e-14)
    for af in [-0.99, -0.3, 0.0, 0.5, 0.9, 0.999]:
        # the denominators cancel strongly near af = -1 (rounding amplified ~1e4)
        assert qnm.fring22(af) == pytest.approx(lalsim.SimIMRPhenomXfring22(af), rel=1e-10)
        assert qnm.fdamp22(af) == pytest.approx(lalsim.SimIMRPhenomXfdamp22(af), rel=1e-10)
        assert utilities.f_isco(af) == pytest.approx(lalsim.SimIMRPhenomXfISCO(af), rel=1e-14)
