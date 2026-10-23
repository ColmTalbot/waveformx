#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``simulate_td`` and ``simulate_fd`` against LALSimulation's
``SimInspiralTD`` and ``SimInspiralFD`` (and waveformrs's), end to end.

LALSimulation is called with multibanding off (``SimInspiralTD`` and
``SimInspiralFD`` call ``SimInspiralChooseFDWaveform``, where it is on by
default for IMRPhenomXPNR; waveformjax and waveformrs do not implement it):
how far LAL's default is from that is recorded too.
IMRPhenomXPNR is compared without the parity fix (as LALSimulation 6.2.0),
with all its modes and with the (2,2) mode alone (which isolates the
conversions: the JAX (2,2) mode agrees with LAL to ~1e-9, the higher modes
to ~1e-5). Recorded for each case: whether the lengths, the epochs and the
sampling are LAL's exactly, the largest relative error and the mismatch of
both polarizations (uniform weights), and the wall-clock times (waveformjax
after its first call, which compiles).

Usage::

    python report/validate_conversions.py
"""

import argparse
import time

import lal
import lalsimulation as lalsim
import numpy as np

from common import BINARY, NRSUR, environment, lal_td_arguments, max_relative_error, median_time, rs, wf, write

DOMINANT = ((2, 2), (2, -2))
HEAVY = dict(BINARY, mass_1=60.0, mass_2=40.0, spin_1=(0.5, -0.2, 0.4), spin_2=(0.1, 0.4, -0.3), f_ref=20.0)
LIGHT = dict(BINARY, mass_1=20.0, mass_2=12.0, spin_1=(0.2, 0.3, -0.1), spin_2=(-0.3, 0.1, 0.2), f_ref=20.0)
NRSUR_HEAVY = dict(NRSUR, mass_1=70.0, mass_2=30.0, spin_1=(-0.3, 0.4, 0.5), spin_2=(0.2, -0.2, 0.1))

# (label, approximant, function, binary, arguments, mode array); NRSur7dq4
# starts at ~18.4 Hz for 40 + 20 solar masses and ~11 Hz for 70 + 30
CASES = [
    ("36+29", "IMRPhenomXPNR", "td", BINARY, (1 / 4096, 20.0), None),
    ("36+29", "IMRPhenomXPNR", "td", BINARY, (1 / 4096, 20.0), DOMINANT),
    ("36+29", "IMRPhenomXPNR", "td", BINARY, (1 / 2048, 15.0), None),
    ("36+29", "IMRPhenomXPNR", "td", BINARY, (1 / 2048, 15.0), DOMINANT),
    ("60+40", "IMRPhenomXPNR", "td", HEAVY, (1 / 2048, 20.0), None),
    ("20+12", "IMRPhenomXPNR", "td", LIGHT, (1 / 4096, 20.0), None),
    ("36+29", "IMRPhenomXPNR", "fd", BINARY, (0.0, 20.0, 2048.0), None),
    ("36+29", "IMRPhenomXPNR", "fd", BINARY, (0.0, 20.0, 2048.0), DOMINANT),
    ("36+29", "IMRPhenomXPNR", "fd", BINARY, (0.25, 19.0, 1000.0), None),
    ("60+40", "IMRPhenomXPNR", "fd", HEAVY, (0.125, 15.0, 1024.0), None),
    ("40+20", "NRSur7dq4", "td", NRSUR, (1 / 4096, 20.0), None),
    ("40+20", "NRSur7dq4", "td", dict(NRSUR, f_ref=25.0), (1 / 2048, 25.0), None),
    ("70+30", "NRSur7dq4", "td", NRSUR_HEAVY, (1 / 2048, 14.0), None),
    ("40+20", "NRSur7dq4", "fd", NRSUR, (0.0, 20.0, 2048.0), None),
    ("40+20", "NRSur7dq4", "fd", NRSUR, (0.25, 19.0, 1000.0), None),
    ("70+30", "NRSur7dq4", "fd", NRSUR_HEAVY, (0.125, 14.0, 1024.0), None),
]


def lal_dict(mode_array, multibanding=False):
    """A LALDict with multibanding off (waveformjax and waveformrs do not
    implement it; ``SimInspiralTD`` and ``SimInspiralFD`` call
    ``SimInspiralChooseFDWaveform``, where it is on by default)."""
    params = lal.CreateDict()
    if not multibanding:
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(params, 0)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(params, 0)
    if mode_array is not None:
        modes = lalsim.SimInspiralCreateModeArray()
        for ell, m in mode_array:
            lalsim.SimInspiralModeArrayActivateMode(modes, ell, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modes)
    return params


def call_lal(approximant, function, keywords, arguments, mode_array, multibanding=False):
    approximant = lalsim.GetApproximantFromString(approximant)
    params = lal_dict(mode_array, multibanding)
    if function == "td":
        delta_t, f_min = arguments
        hp, hc = lalsim.SimInspiralTD(*lal_td_arguments(keywords, delta_t, f_min), params, approximant)
        step = hp.deltaT
    else:
        delta_f, f_min, f_max = arguments
        common = lal_td_arguments(keywords, 0.0, 0.0)[:14]
        hp, hc = lalsim.SimInspiralFD(*common, delta_f, f_min, f_max, keywords["f_ref"], params, approximant)
        step = hp.deltaF
    return float(hp.epoch), step, hp.data.data.copy(), hc.data.data.copy()


def call_package(package, approximant, function, keywords, arguments, mode_array):
    options = dict(mode_array=mode_array)
    if approximant == "IMRPhenomXPNR":
        options["pnr_parity_fix"] = False
    elif package is wf:
        # NRSur7dq4 gives the strain times 1e21 by default
        options["scaled_strain"] = False
    if function == "td":
        out = package.simulate_td(approximant, *arguments, **options, **keywords)
        step = out.delta_t
    else:
        out = package.simulate_fd(approximant, *arguments, **options, **keywords)
        step = out.delta_f
    return float(out.epoch), float(step), np.asarray(out.hplus), np.asarray(out.hcross)


def compare(test, reference):
    epoch_t, step_t, hp_t, hc_t = test
    epoch_r, step_r, hp_r, hc_r = reference
    out = dict(same_length=len(hp_t) == len(hp_r), same_epoch=epoch_t == epoch_r, same_step=step_t == step_r)
    if out["same_length"]:
        out["relative_error"] = max(max_relative_error(hp_t, hp_r), max_relative_error(hc_t, hc_r))
        mismatches = []
        for a, b in ((hp_t, hp_r), (hc_t, hc_r)):
            aa, bb = np.vdot(a, a).real, np.vdot(b, b).real
            mismatches.append(0.0 if aa == bb == 0 else 1 - np.vdot(b, a).real / np.sqrt(aa * bb))
        out["mismatch"] = float(max(mismatches))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="validation_conversions.json")
    args = parser.parse_args()
    records = []
    for label, approximant, function, keywords, arguments, mode_array in CASES:
        if approximant == "NRSur7dq4" and wf.nrsur7dq4_data_file() is None:
            continue
        reference = call_lal(approximant, function, keywords, arguments, mode_array)
        start = time.perf_counter()
        test = call_package(wf, approximant, function, keywords, arguments, mode_array)
        first = time.perf_counter() - start
        record = dict(
            label=label, approximant=approximant, function=f"simulate_{function}", arguments=list(arguments),
            modes="(2,±2)" if mode_array else "all", samples=len(reference[2]),
            jax_vs_lal=compare(test, reference), first_seconds_jax=first,
            seconds_jax=median_time(lambda: call_package(wf, approximant, function, keywords, arguments, mode_array),
                                    repeats=3),
            seconds_lal=median_time(lambda: call_lal(approximant, function, keywords, arguments, mode_array),
                                    repeats=3),
        )
        if approximant == "IMRPhenomXPNR":
            # LAL's default, with multibanding: how far it is from the waveform without
            banded = call_lal(approximant, function, keywords, arguments, mode_array, multibanding=True)
            record["lal_multibanding_vs_lal"] = compare(banded, reference)
            record["seconds_lal_multibanding"] = median_time(
                lambda: call_lal(approximant, function, keywords, arguments, mode_array, multibanding=True),
                repeats=3)
        if rs is not None:
            theirs = call_package(rs, approximant, function, keywords, arguments, mode_array)
            record["rs_vs_lal"] = compare(theirs, reference)
            record["jax_vs_rs"] = compare(test, theirs)
            record["seconds_rs"] = median_time(
                lambda: call_package(rs, approximant, function, keywords, arguments, mode_array), repeats=3)
        records.append(record)
        cmp = record["jax_vs_lal"]
        print(f"{approximant:>13} {record['function']} {label} {arguments} {record['modes']}: n={record['samples']} "
              f"same length/epoch/step {cmp['same_length']}/{cmp['same_epoch']}/{cmp['same_step']} "
              f"error {cmp.get('relative_error', float('nan')):.2e} first call {first:.1f} s", flush=True)
    write(args.output, dict(environment=environment(), records=records))


if __name__ == "__main__":
    main()
