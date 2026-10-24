#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The numerical noise of the waveform: the relative change of the
polarizations (aLIGO weights) for a change of ``mass_1`` in its last bit,
at a few masses, for waveformrs (LAL to rounding: IMRPhenomXPNR, XHM and
XAS), LAL's angles in JAX and the extended angles (whose higher modes take
exact derivatives where LAL takes finite differences); and which
co-precessing mode carries it.

Usage::

    python report/extended/noise.py
"""

import numpy as np

import xcommon as x
from xcommon import jax, jnp

MASSES = (36.0, 36.001, 36.37, 35.9, 40.2, 31.7)
BASE = dict(m2=29., s1=[0.4, 0.2, 0.3], s2=[-0.2, 0.3, -0.1], fref=20., fmin=20.)


def main():
    import waveformrs as rs
    f = np.linspace(20.0, x.f_max(65.0), 2048)
    out = dict(masses=MASSES, waveformrs={}, jax={})
    for approximant in ("IMRPhenomXPNR", "IMRPhenomXHM", "IMRPhenomXAS"):
        changes = []
        for m1 in MASSES:
            kw = dict(mass_2=29., spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), luminosity_distance=500.,
                      inclination=0.5, phi_ref=0.3, f_ref=20.)
            if approximant != "IMRPhenomXPNR":
                kw.update(spin_1=(0., 0., 0.3), spin_2=(0., 0., -0.1))
            a = rs.polarizations(approximant, f, mass_1=m1, **kw)[0]
            b = rs.polarizations(approximant, f, mass_1=float(np.nextafter(m1, 100.)), **kw)[0]
            changes.append(x.relative_change(b, a, f))
        out["waveformrs"][approximant] = changes
    fr = jnp.asarray(f)
    for variant in ("lal", "extended"):
        flags = x.VARIANTS[variant]

        def modes(b, frequencies, flags=flags):
            x.pnr.clear_cache()
            data = x.model._decide(b, frequencies, 0.0, frequencies[0], frequencies[-1], x.MODES, True, **flags)
            return jnp.stack(x.model._coprecessing_modes(data.wf, x.MODES, frequencies))

        pol = jax.jit(x.polarizations_fn(variant))
        cm = jax.jit(modes)
        changes, per_mode = [], []
        for m1 in MASSES:
            b, bn = x.binary(dict(BASE, m1=m1)), x.binary(dict(BASE, m1=float(np.nextafter(m1, 100.))))
            changes.append(x.relative_change(pol(bn, fr)[0], pol(b, fr)[0], f))
            ma, mb = np.asarray(cm(b, fr)), np.asarray(cm(bn, fr))
            per_mode.append([float(np.max(np.abs(mb[i] - ma[i])) / np.max(np.abs(ma[i]))) for i in range(len(x.MODES))])
        out["jax"][variant] = dict(polarizations=changes, modes=per_mode, mode_names=[str(m) for m in x.MODES])
        print(variant, changes)
    x.write("noise.json", out)


if __name__ == "__main__":
    main()
