#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""For the binaries of ``validation.json`` above ``q = 12`` whose extended
waveform differs from LAL's by a mismatch above ``1e-2``: LAL's own jump at
``q = 12`` (the same total mass, spins and frequencies, across ``1e-12``
relative of 12: adjacent floats of ``q`` can round to the same mass ratio).
A model that is LAL's for ``q <= 12`` and continuous must differ from LAL
by about that jump just above 12.

Usage::

    python report/extended/q12_jumps.py
"""

import numpy as np

import xcommon as x
from xcommon import jax

THRESHOLD = 1e-2


def main():
    data = x.read("validation.json")
    width = min(data["mismatches"], key=float)
    lal = jax.jit(x.polarizations_fn("lal"))
    out = []
    for m, d in zip(data["mismatches"][width], data["draws"]):
        if m is None or m <= THRESHOLD or d["m1"] / d["m2"] <= 12:
            continue
        fr = x.frequencies(d, 1024)
        lo, hi = (x.binary(d, m1=d["M"] * q / (1 + q), m2=d["M"] / (1 + q)) for q in (12.0 - 1.2e-11, 12.0 + 1.2e-11))
        jump = max(x.weighted_mismatch(np.asarray(lal(hi, fr)[k]), np.asarray(lal(lo, fr)[k]), fr) for k in (0, 1))
        out.append(dict(q=d["m1"] / d["m2"], M=d["M"], mismatch=m, lal_jump_at_12=jump))
        print(out[-1], flush=True)
    x.write("q12_jumps.json", dict(threshold=THRESHOLD, width=width, binaries=out))


if __name__ == "__main__":
    main()
