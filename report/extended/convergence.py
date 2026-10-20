#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Central finite differences against the forward-mode derivative, for a
range of steps, for LAL's model and the extended one (all modes): where
the finite differences converge to the derivative (between their
truncation error at large steps and the numerical noise of the waveform
divided by the step at small ones). Parameters: an in-plane spin
component and the heavier mass of a generic binary.

Usage::

    python report/extended/convergence.py
"""

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from waveformjax import Binary, WaveformOptions
from waveformjax.models.xpnr import MODEL

POINT = [36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0]
PARAMETERS = {"spin_2x": (5, False), "mass_1": (0, True)}
STEPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


def main():
    x0 = jnp.asarray(POINT)
    grid = jnp.asarray(np.geomspace(20.0, 512.0, 128))
    weights = jnp.asarray(np.random.default_rng(3).normal(size=(2, 128, 2)))
    out = {}
    for variant, options in (("lal", WaveformOptions()), ("extended", WaveformOptions(extended_spintaylor=True))):
        def loss(v, options=options):
            b = Binary(mass_1=v[0], mass_2=v[1], spin_1=v[2:5], spin_2=v[5:8], luminosity_distance=v[8],
                       inclination=v[9], phi_ref=v[10], f_ref=v[11])
            hp, hc = MODEL.polarizations(b, grid, options)
            s = 1e21
            return jnp.sum(weights[0, :, 0] * hp.real * s + weights[0, :, 1] * hp.imag * s
                           + weights[1, :, 0] * hc.real * s + weights[1, :, 1] * hc.imag * s)

        f = jax.jit(loss)
        jvp = jax.jit(lambda v, d, loss=loss: jax.jvp(loss, (v,), (d,))[1])
        out[variant] = {}
        for name, (i, relative) in PARAMETERS.items():
            derivative = float(jvp(x0, jnp.zeros(12).at[i].set(1.0)))
            rows = []
            for step in STEPS:
                h = step * (abs(POINT[i]) if relative else 1.0)
                numeric = (float(f(x0.at[i].add(h))) - float(f(x0.at[i].add(-h)))) / (2 * h)
                rows.append(dict(step=step, finite_difference=numeric, relative_error=abs(numeric - derivative) /
                                 abs(derivative)))
            out[variant][name] = dict(derivative=derivative, relative=relative, steps=rows)
            print(variant, name, [f"{r['relative_error']:.1e}" for r in rows], flush=True)
    x.write("convergence.json", dict(point=POINT, results=out))


if __name__ == "__main__":
    main()
