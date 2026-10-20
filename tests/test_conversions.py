# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``simulate_td`` and ``simulate_fd`` against LALSimulation 6.2.0's
``SimInspiralTD`` and ``SimInspiralFD``.

The conversions are checked on their own by giving them LAL's own model
(``SimInspiralChooseFDWaveform`` on the uniform grid they ask for, or
``SimInspiralChooseTDWaveform``) in place of the JAX one: the lengths, the
epochs and the sampling must be LAL's exactly, and the values agree to
``~1e-12`` of the peak in the time domain (LAL's high-pass filter amplifies
the rounding differences of the Fourier transforms, which are not FFTW's,
by ``~1e4``) and to rounding in the frequency domain. The pieces are checked
against LAL's own functions: the bounds on the duration bit for bit, the
GPS arithmetic, and the Butterworth filter to rounding.

End to end, the conversions of the JAX models agree with LAL as well as the
models do (see test_xpnr.py and test_nrsur7dq4.py), with LAL's lengths and
epochs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import compare, requires_nrsur
from waveformjax import Binary, WaveformOptions
from waveformjax.conversions import _conditioning as c
from waveformjax.conversions import _from_fd, _from_td, _models

XPNR = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), luminosity_distance=500.0,
            inclination=0.7, phi_ref=1.2, f_ref=20.0)
NRSUR = dict(mass_1=40.0, mass_2=20.0, spin_1=(0.4, -0.3, 0.2), spin_2=(0.1, 0.5, -0.3), luminosity_distance=100.0,
             inclination=0.7, phi_ref=0.1, f_ref=20.0)

#: the time-domain conversions, given LAL's model (waveformrs: ~2e-12); the
#: amplification by the filter grows as f_min / f_s decreases (waveformrs
#: has 3.7e-11 at 10 Hz and 8192 Hz)
TD_TOLERANCE = 5e-12
#: the frequency-domain conversions, given LAL's model
FD_TOLERANCE = 1e-13


def binary_of(keywords):
    return Binary(
        np.float64(keywords["mass_1"]), np.float64(keywords["mass_2"]), np.asarray(keywords["spin_1"], float),
        np.asarray(keywords["spin_2"], float), np.float64(keywords["luminosity_distance"]),
        np.float64(keywords["inclination"]), np.float64(keywords["phi_ref"]), np.float64(keywords["f_ref"]),
    )


def lal_arguments(lal_module, keywords):
    return (
        keywords["mass_1"] * lal_module.MSUN_SI, keywords["mass_2"] * lal_module.MSUN_SI, *keywords["spin_1"],
        *keywords["spin_2"], keywords["luminosity_distance"] * 1e6 * lal_module.PC_SI, keywords["inclination"],
        keywords["phi_ref"], 0.0, 0.0, 0.0,
    )


def lal_dict(lal, mode_array=None):
    """A LALDict with multibanding off (waveformjax does not implement it)."""
    lal_module, lalsim = lal
    params = lal_module.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(params, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(params, 0)
    if mode_array is not None:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, emm in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, emm)
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
    return params


def check_values(test, reference, tolerance):
    for a, b in zip(test, reference):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        error = np.max(np.abs(a - b)) / np.max(np.abs(b))
        assert error <= tolerance, error


class LALModel:
    """LAL's IMRPhenomXPNR on the uniform grid in place of the JAX model:
    the conversion asks for the grid of its plan, which is generated in
    advance."""

    name = "IMRPhenomXPNR"
    domain = "frequency"

    def __init__(self, lal, keywords, plan, mode_array=None):
        lal_module, lalsim = lal
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            *lal_arguments(lal_module, keywords), plan.delta_f, plan.f_start, plan.f_max, keywords["f_ref"],
            lal_dict(lal, mode_array), lalsim.IMRPhenomXPNR,
        )
        assert hp.data.length == plan.n and hp.deltaF == plan.delta_f
        assert float(hp.epoch) == c.GpsTime.from_seconds(-1.0 / plan.delta_f).seconds()
        self.hplus, self.hcross = hp.data.data, hc.data.data

    def uniform(self, binary, frequencies, delta_f, f_min, f_max, options):
        """LAL's values at the frequencies of the grid nearest to
        ``frequencies``."""
        index = jnp.clip(jnp.round(frequencies / delta_f).astype(int), 0, len(self.hplus) - 1)
        return jnp.asarray(self.hplus)[index], jnp.asarray(self.hcross)[index]


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------


def test_bounds_are_lals_bit_for_bit(lal):
    lal_module, lalsim = lal
    rng = np.random.default_rng(1)
    for _ in range(300):
        m1, m2 = rng.uniform(1.0, 200.0, 2)
        s1, s2 = rng.uniform(-1.0, 1.0, 2)
        f = rng.uniform(5.0, 100.0)
        kg1, kg2 = m1 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI
        s = lalsim.SimInspiralFinalBlackHoleSpinBound(s1, s2)
        assert c.chirp_time_bound(f, m1, m2, s1, s2) == lalsim.SimInspiralChirpTimeBound(f, kg1, kg2, s1, s2)
        assert c.merge_time_bound(m1, m2) == lalsim.SimInspiralMergeTimeBound(kg1, kg2)
        assert c.final_black_hole_spin_bound(s1, s2) == s
        assert c.ringdown_time_bound_kg(kg1 + kg2, s) == lalsim.SimInspiralRingdownTimeBound(kg1 + kg2, s)
        assert c.chirp_start_frequency_bound(f / 10, m1, m2) == lalsim.SimInspiralChirpStartFrequencyBound(
            f / 10, kg1, kg2)


def test_traced_bounds_agree_with_the_host():
    args = (36.0, 29.0, 0.3, -0.1)
    for function, arguments in ((c.chirp_time_bound, (20.0, *args)), (c.chirp_start_frequency_bound, (2.0, 36.0, 29.0)),
                                (c.merge_and_ringdown_time, args), (c.kerr_isco_frequency, (36.0, 29.0))):
        host = function(*arguments)
        traced = float(jax.jit(lambda *x: function(*x, m=c.TRACED))(*arguments))
        assert abs(traced - host) <= 4e-16 * abs(host)


def test_rounding_helpers():
    assert [c.c_round(x) for x in (0.5, 1.5, 2.5, -0.5, 0.49999999999999994, 3.2)] == [1, 2, 3, -1, 0, 3]
    assert c.next_power_of_two_above(4096.0) == 8192
    assert c.next_power_of_two_above(4097.0) == 8192
    assert c.next_power_of_two_above(4095.0) == 4096
    assert c.nyquist_frequency(2048.0, 0.25) == 2048.0
    assert c.nyquist_frequency(1000.0, 0.25) == 1024.0
    assert c.nyquist_frequency(1000.0, 0.0) == 1000.0
    assert c.c_int_difference(3, 10) == -7


def test_gps_arithmetic_is_lals(lal):
    lal_module, _ = lal
    for t in (-3.0773925781249996, 0.1234567894, -4.0, 1.25e-9, -7.5e-10, -1.0 / 0.1, 123.4567891234):
        mine = c.GpsTime.from_seconds(-1.5).add_seconds(t).seconds()
        reference = lal_module.LIGOTimeGPS(-1.5) + t
        assert mine == float(reference), t


@pytest.mark.parametrize("delta_t, frequency",
                         [(1 / 4096, 20.0), (1 / 1024, 13.0), (1 / 256, 90.0), (1 / 4096, 1500.0)])
def test_high_pass_filter_matches_lal(lal, delta_t, frequency):
    lal_module, _ = lal
    data = np.random.default_rng(2).normal(size=8192)
    series = lal_module.CreateREAL8TimeSeries("x", lal_module.LIGOTimeGPS(0), 0.0, delta_t, lal_module.StrainUnit,
                                              len(data))
    series.data.data = data.copy()
    lal_module.HighPassREAL8TimeSeries(series, frequency, 0.99, 8)
    filtered = jax.jit(lambda x, dt, f: c.high_pass(x[None], dt, f)[0])(jnp.asarray(data), delta_t, frequency)
    check_values([filtered], [series.data.data], 1e-13)


def test_high_pass_filter_of_a_part_of_a_buffer():
    """The filter of the samples ``first .. first + length - 1`` of a
    longer, zero-padded buffer is the filter of those samples."""
    data = np.random.default_rng(3).normal(size=(2, 1000))
    whole = c.high_pass(jnp.asarray(data), 1 / 1024, 20.0)
    buffer = jnp.zeros((2, 1500)).at[:, 300:1300].set(data)
    part = c.high_pass(buffer, 1 / 1024, 20.0, 300, 1000)
    np.testing.assert_allclose(np.asarray(part[:, 300:1300]), np.asarray(whole), rtol=0, atol=1e-15)
    assert np.all(np.asarray(part[:, :300]) == 0) and np.all(np.asarray(part[:, 1300:]) == 0)


# ---------------------------------------------------------------------------
# the conversions with LAL's models
# ---------------------------------------------------------------------------

TD_CASES = [
    ("4096 Hz from 20 Hz", XPNR, 1 / 4096, 20.0, None, TD_TOLERANCE),
    ("2048 Hz from 15 Hz", XPNR, 1 / 2048, 15.0, None, TD_TOLERANCE),
    ("1024 Hz from 25 Hz, f_ref = 0", dict(XPNR, f_ref=0.0), 1 / 1024, 25.0, None, TD_TOLERANCE),
    ("light, 4096 Hz", dict(XPNR, mass_1=12.0, mass_2=8.0, f_ref=25.0), 1 / 4096, 20.0, None, TD_TOLERANCE),
    ("heavy, 8192 Hz from 10 Hz, (2,2) mode", dict(XPNR, mass_1=80.0, mass_2=50.0, f_ref=10.0), 1 / 8192, 10.0,
     ((2, 2), (2, -2)), 1e-10),
]


@pytest.mark.parametrize("name, keywords, delta_t, f_min, mode_array, tolerance", TD_CASES,
                         ids=[case[0] for case in TD_CASES])
def test_time_domain_conversion_matches_lal(lal, name, keywords, delta_t, f_min, mode_array, tolerance):
    lal_module, lalsim = lal
    binary = binary_of(keywords)
    plan = _from_fd.td_plan(_from_fd.params_of(binary), delta_t, f_min, _models.XPNR.cutoff(binary))
    model = LALModel(lal, keywords, plan.fd, mode_array)
    hp, hc, nzeros = _from_fd.time_domain(binary, plan, model, WaveformOptions(mode_array=mode_array))
    stop = plan.first + plan.length - int(nzeros)
    lp, lc = lalsim.SimInspiralTD(*lal_arguments(lal_module, keywords), delta_t, f_min, keywords["f_ref"],
                                  lal_dict(lal, mode_array), lalsim.IMRPhenomXPNR)
    assert stop - plan.first == lp.data.length
    assert plan.epoch.seconds() == float(lp.epoch)
    assert plan.delta_t == lp.deltaT
    check_values([hp[plan.first:stop], hc[plan.first:stop]], [lp.data.data, lc.data.data], tolerance)


FD_CASES = [
    ("automatic delta_f, 2048 Hz", XPNR, 0.0, 20.0, 2048.0),
    ("delta_f = 1/8, 1024 Hz", XPNR, 0.125, 20.0, 1024.0),
    ("delta_f = 0.1, 1000 Hz, f_ref = 0", dict(XPNR, f_ref=0.0), 0.1, 15.0, 1000.0),
    ("delta_f = 8, a short band", XPNR, 8.0, 20.0, 1024.0),
]


@pytest.mark.parametrize("name, keywords, delta_f, f_min, f_max", FD_CASES, ids=[case[0] for case in FD_CASES])
def test_conditioned_frequency_domain_matches_lal(lal, name, keywords, delta_f, f_min, f_max):
    lal_module, lalsim = lal
    binary = binary_of(keywords)
    plan = _from_fd.fd_plan(_from_fd.params_of(binary), delta_f, f_min, f_max, _models.XPNR.cutoff(binary))
    hp, hc = _from_fd.conditioned_fd(binary, plan, f_min, LALModel(lal, keywords, plan), WaveformOptions())
    lp, lc = lalsim.SimInspiralFD(*lal_arguments(lal_module, keywords), delta_f, f_min, f_max, keywords["f_ref"],
                                  lal_dict(lal), lalsim.IMRPhenomXPNR)
    assert len(hp) == lp.data.length and plan.delta_f == lp.deltaF
    assert plan.epoch.seconds() == float(lp.epoch)
    check_values([hp, hc], [lp.data.data, lc.data.data], FD_TOLERANCE)


NRSUR_CASES = [
    ("automatic delta_f, 2048 Hz", NRSUR, 0.0, 20.0, 2048.0),
    ("delta_f = 1/8, 2048 Hz", NRSUR, 0.125, 20.0, 2048.0),
    ("delta_f = 1/4, 1000 Hz, f_ref = 0", dict(NRSUR, f_ref=0.0), 0.25, 19.0, 1000.0),
    ("f_min = 0, 1024 Hz", NRSUR, 0.0, 0.0, 1024.0),
    ("delta_f too large", NRSUR, 2.0, 20.0, 2048.0),
]


@requires_nrsur
@pytest.mark.parametrize("name, keywords, delta_f, f_min, f_max", NRSUR_CASES, ids=[case[0] for case in NRSUR_CASES])
def test_nrsur7dq4_conversions_match_lal(lal, name, keywords, delta_f, f_min, f_max):
    lal_module, lalsim = lal
    arguments = lal_arguments(lal_module, keywords)
    delta_t = _from_td.sampling(delta_f, f_max)
    tp, tc = lalsim.SimInspiralChooseTDWaveform(*arguments, delta_t, f_min, keywords["f_ref"], None, lalsim.NRSur7dq4)
    taper = _from_td.taper_plan(tp.data.data, tc.data.data)
    # SimInspiralTD
    hp, hc = _from_td.tapered(jnp.asarray(tp.data.data), jnp.asarray(tc.data.data), taper)
    lp, lc = lalsim.SimInspiralTD(*arguments, delta_t, f_min, keywords["f_ref"], None, lalsim.NRSur7dq4)
    assert float(lp.epoch) == float(tp.epoch) and lp.data.length == tp.data.length
    check_values([hp, hc], [lp.data.data, lc.data.data], FD_TOLERANCE)
    # SimInspiralFD
    plan = _from_td.fd_plan(tp.data.length, tp.deltaT, float(tp.epoch), delta_f, f_max)
    hp, hc = _from_td.frequency_domain(jnp.asarray(tp.data.data), jnp.asarray(tc.data.data), taper, plan)
    lp, lc = lalsim.SimInspiralFD(*arguments, delta_f, f_min, f_max, keywords["f_ref"], None, lalsim.NRSur7dq4)
    assert len(hp) == lp.data.length and plan.delta_f == lp.deltaF
    assert plan.epoch.seconds() == float(lp.epoch)
    check_values([hp, hc], [lp.data.data, lc.data.data], FD_TOLERANCE)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

#: the (2,2) mode of IMRPhenomXPNR agrees with LAL to ~1e-9, all its modes
#: to ~1e-5 (test_xpnr.py)
XPNR_TOLERANCE = {((2, 2), (2, -2)): (1e-7, 1e-13), None: (1e-4, 1e-9)}


@pytest.mark.parametrize("mode_array", list(XPNR_TOLERANCE), ids=["(2,2) mode", "all modes"])
@pytest.mark.parametrize("delta_t, f_min", [(1 / 4096, 20.0), (1 / 2048, 15.0)])
def test_simulate_td_matches_lal(lal, mode_array, delta_t, f_min):
    lal_module, lalsim = lal
    result = wf.simulate_td("IMRPhenomXPNR", delta_t, f_min, mode_array=mode_array, pnr_parity_fix=False, **XPNR)
    lp, lc = lalsim.SimInspiralTD(*lal_arguments(lal_module, XPNR), delta_t, f_min, XPNR["f_ref"],
                                  lal_dict(lal, mode_array), lalsim.IMRPhenomXPNR)
    assert len(result.hplus) == lp.data.length
    assert result.epoch == float(lp.epoch) and result.delta_t == lp.deltaT
    np.testing.assert_array_equal(np.asarray(result.times), result.epoch + result.delta_t * np.arange(lp.data.length))
    tolerance, mismatch_tolerance = XPNR_TOLERANCE[mode_array]
    for test, reference in ((result.hplus, lp.data.data), (result.hcross, lc.data.data)):
        error, mismatch = compare(test, reference)
        assert error < tolerance and mismatch < mismatch_tolerance, (error, mismatch)


@requires_nrsur
@pytest.mark.parametrize("delta_f, f_min, f_max", [(0.0, 20.0, 2048.0), (0.25, 19.0, 1000.0)])
def test_simulate_fd_matches_lal(lal, delta_f, f_min, f_max):
    lal_module, lalsim = lal
    result = wf.simulate_fd("NRSur7dq4", delta_f, f_min, f_max, **NRSUR)
    lp, lc = lalsim.SimInspiralFD(*lal_arguments(lal_module, NRSUR), delta_f, f_min, f_max, NRSUR["f_ref"], None,
                                  lalsim.NRSur7dq4)
    assert len(result.hplus) == lp.data.length
    assert result.epoch == float(lp.epoch) and result.delta_f == lp.deltaF
    np.testing.assert_array_equal(np.asarray(result.frequencies), np.arange(lp.data.length) * lp.deltaF)
    for test, reference in ((result.hplus, lp.data.data), (result.hcross, lc.data.data)):
        error, mismatch = compare(test, reference)
        assert error < 1e-10 and mismatch < 1e-12, (error, mismatch)


@requires_nrsur
def test_simulate_td_of_nrsur7dq4_matches_lal(lal):
    lal_module, lalsim = lal
    result = wf.simulate_td("NRSur7dq4", 1 / 4096, 20.0, **NRSUR)
    lp, lc = lalsim.SimInspiralTD(*lal_arguments(lal_module, NRSUR), 1 / 4096, 20.0, NRSUR["f_ref"], None,
                                  lalsim.NRSur7dq4)
    assert len(result.hplus) == lp.data.length and result.epoch == float(lp.epoch)
    for test, reference in ((result.hplus, lp.data.data), (result.hcross, lc.data.data)):
        error, mismatch = compare(test, reference)
        assert error < 1e-10 and mismatch < 1e-12, (error, mismatch)


def test_arguments_are_checked():
    with pytest.raises(wf.InvalidParameterError):
        wf.simulate_td("IMRPhenomXPNR", 1 / 4096, 0.0, **XPNR)
    with pytest.raises(wf.InvalidParameterError):
        wf.simulate_td("IMRPhenomXPNR", -1.0, 20.0, **XPNR)
    with pytest.raises(wf.InvalidParameterError):
        wf.simulate_fd("NRSur7dq4", -0.125, 20.0, 2048.0, **NRSUR)
    with pytest.raises(wf.InvalidParameterError):
        wf.simulate_fd("NRSur7dq4", 0.125, 20.0, 0.0, **NRSUR)
    with pytest.raises(wf.UnknownApproximantError):
        wf.simulate_fd("IMRPhenomXPHM", 0.125, 20.0, 1024.0, **XPNR)
