#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in JAX against LALSimulation's ``SimInspiralChooseTDWaveform``
and waveformrs over random binaries.

The draws are those of waveformrs's ``scripts/validate_nrsur7dq4.py`` (seed
150914): mass ratios 1-6 (with the lighter body first in about a third of
them), isotropic spins of magnitude up to 0.99, total masses 20-300 solar
masses (log-uniform), sampling rates of 1-8 kHz, ``f_min`` above the
surrogate's starting frequency for the binary (or zero, the whole
surrogate, in 15% of the draws) and ``f_ref`` at or above it.

Recorded for each draw: whether the epochs and lengths are LAL's, the
largest relative error ``max |h - h_LAL| / max |h_LAL|`` and the
noise-weighted mismatch of both polarizations (aLIGO design sensitivity
from 10 Hz, without maximisation), the same against waveformrs, and the
wall-clock times (waveformjax through the public ``td_polarizations``,
whose output grids of similar lengths share a compilation).

Usage::

    python report/validate_nrsur7dq4.py [--n-draws 200]
"""

import argparse
import time

import lal
import lalsimulation as lalsim
import numpy as np

from common import environment, lal_td_arguments, max_relative_error, median_time, rs, td_mismatch, wf, write


def ordered(case):
    """Masses (Msun) and spins with the heavier body first, as LAL does."""
    m1, m2 = case["mass_1"], case["mass_2"]
    s1, s2 = case["spin_1"], case["spin_2"]
    if m1 < m2:
        m1, m2 = m2, m1
        s1, s2 = [-s2[0], -s2[1], s2[2]], [-s1[0], -s1[1], s1[2]]
    return m1, m2, s1, s2


def start_frequency(case):
    """The surrogate's starting frequency for this binary, by bisection on
    ``f_ref``: ``PrecessingNRSurDynamics`` fails below it."""
    mass_a, mass_b, s1, s2 = ordered(case)
    m_tot_sec = (mass_a + mass_b) * lal.MTSUN_SI

    def valid(f_ref):
        try:
            lalsim.PrecessingNRSurDynamics(mass_a / mass_b, *s1, *s2, (m_tot_sec * f_ref) * np.pi,
                                           1.0, 0.0, 0.0, 0.0, 0.0, None, lalsim.NRSur7dq4)
            return True
        except RuntimeError:
            return False

    lo, hi = 0.1, 0.19 / (np.pi * m_tot_sec)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if valid(mid) else (mid, hi)
    return hi


def draw(rng):
    total_mass = float(np.exp(rng.uniform(np.log(20.0), np.log(300.0))))
    q = rng.uniform(1.0, 6.0)
    heavy, light = total_mass * q / (1 + q), total_mass / (1 + q)
    mass_1, mass_2 = (light, heavy) if rng.uniform() < 1 / 3 else (heavy, light)
    spins = []
    for _ in range(2):
        magnitude = rng.uniform(0, 0.99)
        cos_theta = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        spins.append((magnitude * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])).tolist())
    rate = 4096.0 if total_mass < 60 else 2048.0 if total_mass < 150 else 1024.0
    if rng.uniform() < 0.25:
        rate *= 2
    case = dict(
        mass_1=float(mass_1), mass_2=float(mass_2), spin_1=spins[0], spin_2=spins[1],
        luminosity_distance=float(np.exp(rng.uniform(np.log(100), np.log(3000)))),
        inclination=float(np.arccos(rng.uniform(-1, 1))), phi_ref=float(rng.uniform(0, 2 * np.pi)),
        delta_t=1.0 / rate,
    )
    f_start = start_frequency(case)
    if rng.uniform() < 0.15:
        f_min = 0.0
        f_ref = 0.0 if rng.uniform() < 0.5 else f_start * rng.uniform(1.001, 1.5)
    else:
        f_min = f_start * rng.uniform(1.001, 1.6)
        f_ref = f_min if rng.uniform() < 0.5 else f_min * rng.uniform(1.0, 1.5)
    case.update(f_min=float(f_min), f_ref=float(f_ref), f_start=float(f_start))
    return case


def keywords_of(case):
    return {k: case[k] for k in ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance", "inclination",
                                 "phi_ref", "f_ref")}


def lal_waveform(case):
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        *lal_td_arguments(keywords_of(case), case["delta_t"], case["f_min"]), None, lalsim.NRSur7dq4)
    return float(hp.epoch), hp.data.data.copy(), hc.data.data.copy()


def jax_waveform(case):
    out = wf.td_polarizations("NRSur7dq4", case["delta_t"], case["f_min"], **keywords_of(case))
    return float(out.epoch), np.asarray(out.hplus), np.asarray(out.hcross)


def rs_waveform(case):
    out = rs.td_polarizations("NRSur7dq4", case["delta_t"], case["f_min"], **keywords_of(case))
    return float(out.epoch), np.asarray(out.hplus), np.asarray(out.hcross)


def run(function, case, repeats=3):
    try:
        first = time.perf_counter()
        result = function(case)
        first = time.perf_counter() - first
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}", None
    return result, median_time(lambda: function(case), repeats=repeats), first


def compare(test, reference, delta_t):
    epoch_t, hp_t, hc_t = test
    epoch_r, hp_r, hc_r = reference
    same_grid = len(hp_t) == len(hp_r) and epoch_t == epoch_r
    out = dict(same_length=len(hp_t) == len(hp_r), same_epoch=epoch_t == epoch_r,
               epoch_difference=epoch_t - epoch_r)
    if same_grid:
        out["relative_error"] = max(max_relative_error(hp_t, hp_r), max_relative_error(hc_t, hc_r))
        out["mismatch"] = max(td_mismatch(hp_t, hp_r, delta_t), td_mismatch(hc_t, hc_r, delta_t))
    return out


def summary(records, key):
    values = [r[key] for r in records if r.get(key) and "relative_error" in r[key]]
    if not values:
        return None
    errors = np.array([v["relative_error"] for v in values])
    mismatches = np.array([v["mismatch"] for v in values])
    return dict(n=len(values), n_same_grid=sum(1 for r in records if r.get(key) and r[key]["same_length"]
                                               and r[key]["same_epoch"]),
                max_relative_error=float(errors.max()), median_relative_error=float(np.median(errors)),
                p90_relative_error=float(np.percentile(errors, 90)), max_mismatch=float(mismatches.max()),
                median_mismatch=float(np.median(mismatches)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=150914)
    parser.add_argument("--output", default="validation_nrsur7dq4.json")
    args = parser.parse_args()
    if wf.nrsur7dq4_data_file() is None:
        raise SystemExit("NRSur7dq4_v1.0.h5 not found: set NRSUR7DQ4_DATA")

    rng = np.random.default_rng(args.seed)
    cases = [draw(rng) for _ in range(args.n_draws)]
    records = []
    for index, case in enumerate(cases):
        record = dict(case, index=index)
        results = {"lal": run(lal_waveform, case), "jax": run(jax_waveform, case)}
        if rs is not None:
            results["rs"] = run(rs_waveform, case)
        for name, (result, info, first) in results.items():
            if result is None:
                record[f"error_{name}"] = info
            else:
                record[f"seconds_{name}"] = info
                record[f"first_seconds_{name}"] = first
                record["samples"] = len(result[1])
        if results["lal"][0] is not None and results["jax"][0] is not None:
            record["jax_vs_lal"] = compare(results["jax"][0], results["lal"][0], case["delta_t"])
        if rs is not None and results["rs"][0] is not None and results["jax"][0] is not None:
            record["jax_vs_rs"] = compare(results["jax"][0], results["rs"][0], case["delta_t"])
        record["failure_agrees"] = (results["lal"][0] is None) == (results["jax"][0] is None)
        records.append(record)
        cmp = record.get("jax_vs_lal")
        print(f"{index:4d} M={case['mass_1'] + case['mass_2']:6.1f} rate={1 / case['delta_t']:5.0f} "
              f"f_min={case['f_min']:6.2f} " + (
                  f"n={record['samples']} error {cmp['relative_error']:.2e} mismatch {cmp['mismatch']:.2e}"
                  if cmp and "relative_error" in cmp else
                  f"not compared ({record.get('error_lal') or record.get('error_jax') or cmp})"), flush=True)

    payload = dict(
        environment=environment(), seed=args.seed, n_draws=args.n_draws,
        n_lal_failures=sum(1 for r in records if "error_lal" in r),
        n_failures_agree=sum(1 for r in records if r["failure_agrees"]),
        summary={key: summary(records, key) for key in ("jax_vs_lal", "jax_vs_rs")},
        median_seconds={name: float(np.median([r[f"seconds_{name}"] for r in records if f"seconds_{name}" in r]))
                        for name in ("jax", "lal", "rs") if any(f"seconds_{name}" in r for r in records)},
        records=records,
    )
    write(args.output, payload)
    for key, value in payload["summary"].items():
        if value:
            print(f"{key:>12}: n {value['n']} (same grid {value['n_same_grid']}), max error "
                  f"{value['max_relative_error']:.2e}, median {value['median_relative_error']:.2e}, "
                  f"max mismatch {value['max_mismatch']:.2e}")


if __name__ == "__main__":
    main()
