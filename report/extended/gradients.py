#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The derivatives of the extended model (``extended_spintaylor=True``,
all modes) against central finite differences, for every parameter of
three binaries: a generic one, one whose reference frequency lies beyond
LAL's PN energy test and a single-spin one; and that they are finite for
aligned and non-spinning binaries. The loss is a fixed random linear
function of the polarizations (as in ``tests/test_transforms_xpnr.py``).
The steps are ``3e-5`` (relative for the masses, the distance and
``f_ref``): between the truncation error of the differences and the
numerical noise of the higher modes (~1e-9) divided by the step, see
``convergence.py``.

Usage::

    python report/extended/gradients.py
"""

import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL

OPTIONS = WaveformOptions(extended_spintaylor=True)
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]
RELATIVE = {"mass_1", "mass_2", "luminosity_distance", "f_ref"}
POINTS = {
    "generic (36+29)": ([36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0], 20.0),
    "f_ref beyond the energy test (180+120)": ([180.0, 120.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 60.0], 15.0),
    "single spin, q = 8 (48+6)": ([48.0, 6.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 500.0, 0.7, 1.2, 25.0], 20.0),
    "aligned spins (36+29)": ([36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.1, 500.0, 0.7, 1.2, 25.0], 20.0),
    "non-spinning (36+29)": ([36.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.7, 1.2, 25.0], 20.0),
}
CHECKED = 3
STEP = 3e-5


def main():
    out = []
    for k, (name, (point, f_min)) in enumerate(POINTS.items()):
        x0 = jnp.asarray(point)
        grid = jnp.asarray(np.geomspace(f_min, min(x.f_max(point[0] + point[1]), 512.0), 128))
        weights = jnp.asarray(np.random.default_rng(3).normal(size=(2, grid.shape[0], 2)))

        def loss(v, grid=grid, weights=weights):
            b = Binary(mass_1=v[0], mass_2=v[1], spin_1=v[2:5], spin_2=v[5:8], luminosity_distance=v[8],
                       inclination=v[9], phi_ref=v[10], f_ref=v[11])
            hp, hc = MODEL.polarizations(b, grid, OPTIONS)
            s = 1e21
            return jnp.sum(weights[0, :, 0] * hp.real * s + weights[0, :, 1] * hp.imag * s
                           + weights[1, :, 0] * hc.real * s + weights[1, :, 1] * hc.imag * s)

        t = time.time()
        gradient = np.asarray(jax.jit(jax.grad(loss))(x0))
        seconds = time.time() - t
        record = dict(binary=name, finite=bool(np.all(np.isfinite(gradient))), gradient=gradient.tolist(),
                      seconds=seconds, parameters=[])
        if k < CHECKED:
            f = jax.jit(loss)
            scale = float(np.max(np.abs(gradient)))
            for i, pname in enumerate(NAMES):
                step = STEP * (abs(float(x0[i])) if pname in RELATIVE else 1.0)
                numeric = (float(f(x0.at[i].add(step))) - float(f(x0.at[i].add(-step)))) / (2 * step)
                error = abs(gradient[i] - numeric) / max(abs(numeric), 1e-3 * scale)
                record["parameters"].append(dict(name=pname, gradient=float(gradient[i]), finite_difference=numeric,
                                                 relative_error=error))
        print(name, record["finite"], max([p["relative_error"] for p in record["parameters"]] or [0.0]), flush=True)
        out.append(record)
    x.write("gradients.json", dict(records=out))


if __name__ == "__main__":
    main()
