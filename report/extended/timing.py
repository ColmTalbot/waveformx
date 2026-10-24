#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The cost of the three variants through the public pure function
(``MODEL.polarizations`` under ``jax.jit``): the first call (compilation)
and the median of later calls, for the benchmark binary of the main report
on its 8033 frequencies, and the compilation and median time of
``jax.grad`` of a scalar loss (``(2,2)`` mode). Run on an otherwise idle
machine.

Usage::

    python report/extended/timing.py
"""

import functools
import time

import numpy as np

import xcommon as x
from xcommon import jax, jnp
from common import BENCHMARK_FREQUENCIES, BINARY, binary_of, median_time
from waveformjax import WaveformOptions
from waveformjax.models.xpnr import MODEL

OPTIONS = {"lal": WaveformOptions(version_fallback=True), "strict": WaveformOptions(version_fallback=False),
           "extended": WaveformOptions(extended_spintaylor=True)}


def main():
    b = binary_of(BINARY)
    f = jnp.asarray(BENCHMARK_FREQUENCIES)
    out = {}
    for name, options in OPTIONS.items():
        fn = jax.jit(functools.partial(MODEL.polarizations, options=options))
        t = time.perf_counter()
        jax.block_until_ready(fn(b, f))
        first = time.perf_counter() - t
        median = median_time(lambda: fn(b, f), repeats=7)
        o22 = options.replace(mode_array=((2, 2), (2, -2)))
        grid = jnp.asarray(np.geomspace(20.0, 400.0, 96))

        def loss(m1, o22=o22, grid=grid):
            hp, _ = MODEL.polarizations(b._replace(mass_1=m1), grid, o22)
            return jnp.sum(jnp.abs(hp) ** 2) * 1e42

        g = jax.jit(jax.grad(loss))
        t = time.perf_counter()
        jax.block_until_ready(g(b.mass_1))
        grad_first = time.perf_counter() - t
        grad_median = median_time(lambda: g(b.mass_1), repeats=5)
        out[name] = dict(first_call=first, median=median, grad_first_call=grad_first, grad_median=grad_median)
        print(name, out[name], flush=True)
    x.write("timing.json", dict(n_frequencies=int(f.shape[0]), timings=out))


if __name__ == "__main__":
    main()
