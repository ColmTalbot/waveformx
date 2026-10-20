#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR in JAX against LALSimulation and waveformrs over random
binaries.

The draws are those of waveformrs's ``scripts/validate_against_lalsimulation.py``
(seed 170817): total masses log-uniform in 5-300 solar masses, mass ratios
uniform in 1/20-1, isotropic spins of magnitude up to 0.99, ``f_min`` among
10, 15 and 20 Hz, ``f_ref = f_min`` or uniform up to ``3 f_min``, distances
log-uniform in 100-3000 Mpc and isotropic lines of sight. The frequencies are
:data:`N_FREQUENCIES` uniformly spaced points from ``f_min`` to ``min(0.3 /
M, 2048 Hz)``: one length, so one compilation serves every draw (waveformrs
used a spacing of 0.25 Hz, which gives each draw its own length).

For each draw this compares, through the public ``waveformjax.polarizations``
(LAL's checks on the host, then the compiled model):

* the model without the parity fix with LALSimulation 6.2.0
  (``SimInspiralChooseFDWaveformSequence``);
* the model with the fix (the default) with waveformrs's (which agrees with
  LALSimulation 6.2.0 + lalsuite !2593 to ~3e-15) and, if a patched
  liblalsimulation is given (``--lalsimulation-library`` or
  ``PATCHED_LALSIMULATION``, built by waveformrs's
  ``scripts/build_patched_lalsimulation.sh``), with that library directly;
* both with waveformrs, which reproduces LAL's arithmetic bit for bit.

Recorded: the largest relative error ``max |h - h_ref| / max |h_ref|`` and the
noise-weighted mismatch (aLIGO design sensitivity, no maximisation) of both
polarizations, and the wall-clock times. Where LAL fails, waveformjax must
raise or give NaN.

Usage::

    python report/validate_xpnr.py [--n-draws 200]
"""

import argparse
import os
import sys

import numpy as np

from common import (MTSUN_SI, WAVEFORMRS, environment, lal_sequence, max_relative_error, median_time, mismatch,
                    psd, rs, wf, write)

N_FREQUENCIES = 4096


def draw(rng):
    """One binary of waveformrs's draws for IMRPhenomXPNR."""
    while True:
        total_mass = np.exp(rng.uniform(np.log(5.0), np.log(300.0)))
        mass_ratio = rng.uniform(1 / 20, 1.0)
        mass_1 = total_mass / (1 + mass_ratio)
        mass_2 = total_mass - mass_1
        f_min = float(rng.choice([10.0, 15.0, 20.0]))
        f_max = min(0.3 / (total_mass * MTSUN_SI), 2048.0)
        if f_max > 2 * f_min:
            break
    spins = []
    for _ in range(2):
        magnitude = rng.uniform(0, 0.99)
        cos_theta = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        spins.append([float(x) for x in magnitude * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi),
                                                                cos_theta])])
    f_ref = f_min if rng.uniform() < 0.5 else rng.uniform(f_min, 3 * f_min)
    return dict(
        mass_1=float(mass_1), mass_2=float(mass_2), spin_1=spins[0], spin_2=spins[1],
        luminosity_distance=float(np.exp(rng.uniform(np.log(100), np.log(3000)))),
        inclination=float(np.arccos(rng.uniform(-1, 1))), phi_ref=float(rng.uniform(0, 2 * np.pi)),
        f_ref=float(f_ref), f_min=f_min, f_max=float(f_max),
    )


def evaluate(function, repeats=3):
    """``(hplus, hcross, seconds)`` or ``(None, None, error message)``."""
    try:
        hp, hc = function()
    except Exception as error:  # noqa: BLE001 (LAL's RuntimeError, waveformjax's WaveformError)
        return None, None, f"{type(error).__name__}: {error}"
    seconds = median_time(function, repeats=repeats)
    return np.asarray(hp), np.asarray(hc), seconds


def compare(test, reference, weights):
    return dict(
        relative_error=max(max_relative_error(test[0], reference[0]), max_relative_error(test[1], reference[1])),
        mismatch=max(mismatch(test[0], reference[0], weights), mismatch(test[1], reference[1], weights)),
    )


def summary(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    if not values:
        return None
    errors = np.array([v["relative_error"] for v in values])
    mismatches = np.array([v["mismatch"] for v in values])
    return dict(n=len(values), max_relative_error=float(errors.max()), median_relative_error=float(np.median(errors)),
                p90_relative_error=float(np.percentile(errors, 90)), max_mismatch=float(mismatches.max()),
                median_mismatch=float(np.median(mismatches)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=170817)
    parser.add_argument("--lalsimulation-library", default=os.environ.get("PATCHED_LALSIMULATION"))
    parser.add_argument("--output", default="validation_xpnr.json")
    args = parser.parse_args()

    patched, patched_build = None, None
    if args.lalsimulation_library:
        sys.path.insert(0, str(WAVEFORMRS / "scripts"))
        import lalsimulation as lalsim
        from lalsimulation_ctypes import Library, Overlay

        library = Library(args.lalsimulation_library)
        patched = Overlay(lalsim, library)
        patched_build = library.vcs_info

    rng = np.random.default_rng(args.seed)
    cases = [draw(rng) for _ in range(args.n_draws)]
    records = []
    for index, case in enumerate(cases):
        keywords = {k: case[k] for k in ("mass_1", "mass_2", "spin_1", "spin_2", "luminosity_distance",
                                         "inclination", "phi_ref", "f_ref")}
        frequencies = np.linspace(case["f_min"], case["f_max"], N_FREQUENCIES)
        weights = 1 / psd(frequencies)
        record = dict(case, index=index)
        results = {
            "jax": evaluate(lambda: wf.polarizations("IMRPhenomXPNR", frequencies, pnr_parity_fix=False, **keywords)),
            "jax_fix": evaluate(lambda: wf.polarizations("IMRPhenomXPNR", frequencies, **keywords)),
            "lal": evaluate(lambda: lal_sequence(keywords, frequencies)),
        }
        if rs is not None:
            results["rs"] = evaluate(lambda: rs.polarizations("IMRPhenomXPNR", frequencies, pnr_parity_fix=False,
                                                              **keywords))
            results["rs_fix"] = evaluate(lambda: rs.polarizations("IMRPhenomXPNR", frequencies, **keywords))
        if patched is not None:
            results["lal_fix"] = evaluate(lambda: lal_sequence(keywords, frequencies, module=patched))
        for name, (hp, hc, info) in results.items():
            if hp is None:
                record[f"error_{name}"] = info
            else:
                record[f"seconds_{name}"] = info
                record[f"finite_{name}"] = bool(np.all(np.isfinite(hp)) and np.all(np.isfinite(hc)))

        def against(test, reference):
            a, b = results.get(test), results.get(reference)
            if a is None or b is None or a[0] is None or b[0] is None:
                return None
            if not (record[f"finite_{test}"] and record[f"finite_{reference}"]):
                return None
            return compare(a[:2], b[:2], weights)

        record["jax_vs_lal"] = against("jax", "lal")
        record["jax_fix_vs_lal_fix"] = against("jax_fix", "lal_fix")
        record["jax_vs_rs"] = against("jax", "rs")
        record["jax_fix_vs_rs_fix"] = against("jax_fix", "rs_fix")
        record["rs_fix_vs_lal_fix"] = against("rs_fix", "lal_fix")
        # LAL fails: waveformjax must reject the binary or give NaN
        lal_failed = results["lal"][0] is None
        jax_failed = results["jax"][0] is None or not record["finite_jax"]
        record["failure_agrees"] = lal_failed == jax_failed
        records.append(record)
        cmp = record["jax_vs_lal"]
        print(f"{index:4d} M={case['mass_1'] + case['mass_2']:7.2f} q={case['mass_2'] / case['mass_1']:.3f} "
              + (f"error {cmp['relative_error']:.2e} mismatch {cmp['mismatch']:.2e}" if cmp else
                 f"not compared ({record.get('error_lal') or record.get('error_jax') or 'non-finite'})"),
              flush=True)

    payload = dict(
        environment=environment(), seed=args.seed, n_draws=args.n_draws, n_frequencies=N_FREQUENCIES,
        patched_build=patched_build,
        n_lal_failures=sum(1 for r in records if "error_lal" in r),
        n_failures_agree=sum(1 for r in records if r["failure_agrees"]),
        summary={key: summary(records, key) for key in ("jax_vs_lal", "jax_fix_vs_lal_fix", "jax_vs_rs",
                                                         "jax_fix_vs_rs_fix", "rs_fix_vs_lal_fix")},
        median_seconds={name: float(np.median([r[f"seconds_{name}"] for r in records if f"seconds_{name}" in r]))
                        for name in ("jax", "jax_fix", "lal", "rs", "rs_fix", "lal_fix")
                        if any(f"seconds_{name}" in r for r in records)},
        records=records,
    )
    write(args.output, payload)
    for key, value in payload["summary"].items():
        if value:
            print(f"{key:>20}: n {value['n']}, max error {value['max_relative_error']:.2e}, "
                  f"median {value['median_relative_error']:.2e}, max mismatch {value['max_mismatch']:.2e}")


if __name__ == "__main__":
    main()
