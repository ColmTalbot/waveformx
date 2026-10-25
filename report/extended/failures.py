#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Where IMRPhenomXPNR's SpinTaylor angles fail, why, and the extended
angles there.

1. For random binaries (:func:`xcommon.draws`), waveformrs (LAL to
   rounding) with ``version_fallback=False``: whether LAL keeps the
   SpinTaylor angles (in parallel processes).
2. For every failure and a sample of the others, the PN energy test of
   ``XLALSimInspiralSpinTaylorStoppingTest`` at ``f_ref`` with the spins at
   ``f_ref`` and the stopping code of LAL's backward run.
3. For every failure, the extended angles: finite or not, and their
   mismatch with LAL's waveform (which uses the MSA angles there).

Usage::

    python report/extended/failures.py [--n 20000]
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import xcommon as x
from xcommon import jax, jnp

N_FREQUENCIES = 512


def lal_status(d):
    import waveformrs as rs
    f = np.linspace(d["fmin"], x.f_max(d["M"]), 256)
    try:
        rs.polarizations("IMRPhenomXPNR", f, version_fallback=False, mass_1=d["m1"], mass_2=d["m2"], spin_1=d["s1"],
                         spin_2=d["s2"], luminosity_distance=500.0, inclination=0.5, phi_ref=0.3, f_ref=d["fref"])
        return "ok"
    except Exception as error:  # noqa: BLE001 (waveformrs's errors)
        return type(error).__name__


def diagnosis():
    from waveformx.models.xpnr._math import MTSUN_SI
    from waveformx.models.xpnr.spintaylor import angles, dynamics, replica

    def f(b, frequencies):
        x.pnr.clear_cache()
        w, m1_si, m2_si, chi1, chi2 = x.model.setup_waveform(b, 0.0, frequencies[0], frequencies[-1])
        w.hm_release = 122019
        base, wf0 = x.pnr.common_setup(w, m1_si, m2_si, chi1, chi2, x.pnr.mode_range(x.MODES), True)
        msec = (m1_si + m2_si) * MTSUN_SI
        y_ref = angles._initial_state(m1_si, m2_si, base.chi1, base.chi2, wf0.f_ref, msec)
        c_back = dynamics.coefficients(m1_si, m2_si, wf0.f_ref, base.st_flow - 0.5)
        start = replica.evolve(c_back, y_ref, -0.5 * 10.0 / base.st_fcut / msec, 8)
        test, v = dynamics.energy_test(c_back, y_ref)
        return dict(energy_test=test, v_ref=v, back_code=start.code, back_samples=start.n_samples)

    return jax.jit(jax.vmap(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    draws = x.draws(args.n, args.seed)
    t = time.time()
    with Pool(args.processes) as pool:
        status = pool.map(lal_status, draws, chunksize=8)
    print(f"LAL statuses: {time.time() - t:.0f} s")
    for d, s in zip(draws, status):
        d["status"] = s
    fail = [d for d in draws if d["status"] != "ok"]
    rng = np.random.default_rng(0)
    ok = [d for d in draws if d["status"] == "ok"]
    sample = [ok[i] for i in rng.choice(len(ok), min(len(ok), 1024), replace=False)]

    diagnose = diagnosis()
    for group in (fail, sample):
        for i in range(0, len(group), 64):
            rows = group[i:i + 64]
            out = diagnose(x.stack([x.binary(d) for d in rows]), jnp.stack([x.frequencies(d, 64) for d in rows]))
            for j, d in enumerate(rows):
                d.update({k: float(v[j]) for k, v in out.items()})
    print(f"diagnosis: {time.time() - t:.0f} s")

    # the extended angles where LAL fails, against LAL's (MSA) waveform
    import waveformrs as rs
    extended = jax.jit(x.polarizations_fn("extended"))
    for d in fail:
        f = x.frequencies(d, N_FREQUENCIES)
        hp, hc = (np.asarray(v) for v in extended(x.binary(d), f))
        d["extended_finite"] = bool(np.all(np.isfinite(hp)) and np.all(np.isfinite(hc)))
        if d["extended_finite"]:
            rp, rc = rs.polarizations("IMRPhenomXPNR", np.asarray(f), mass_1=d["m1"], mass_2=d["m2"], spin_1=d["s1"],
                                      spin_2=d["s2"], luminosity_distance=500.0, inclination=0.5, phi_ref=0.3,
                                      f_ref=d["fref"])
            d["mismatch_to_lal_msa"] = max(x.weighted_mismatch(hp, rp, f), x.weighted_mismatch(hc, rc, f))
    print(f"extended: {time.time() - t:.0f} s")
    x.write("failures.json", dict(n=len(draws), seed=args.seed, draws=draws))


if __name__ == "__main__":
    main()
