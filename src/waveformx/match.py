# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Matches between waveforms, maximised over time and phase or polarisation,
and their optimisation over the parameters that models define differently.

The waveforms are frequency series on a common uniform grid ``frequencies =
f_0 + k delta_f`` (from :func:`waveformx.polarizations` on those
frequencies, or the band of :func:`waveformx.simulate_fd` of a time-domain
model), weighted by a power spectral density on the same grid. The inner
product is

    <a, b> = 4 delta_f Re sum_k a_k conj(b_k) / S_k,

and the overlap of ``a`` with ``b`` shifted by a time ``t`` is ``z(t) = sum_k
a_k conj(b_k) e^(2 pi i f_k t) / S_k``. It is evaluated on a grid of times
by a zero-padded inverse FFT, and the best time is refined by Newton steps
on the exact ``z(t)``, so that the result is a smooth function of the
waveforms: everything here is pure JAX, works under ``jax.jit`` and
``jax.vmap``, and can be differentiated (the gradient at the maximum is
that of the overlap at the best time, which is held fixed).

* :func:`match`, :func:`mismatch`: maximised over time and (``phase=True``)
  a constant phase, ``|z(t)|``;
* :func:`polarization_match`, :func:`polarization_mismatch`: the template
  ``F_+ h_+ + F_x h_x`` maximised over time and the ratio and signs of the
  antenna patterns (the polarisation angle and sky position), as
  Harry et al. 2016 (arXiv:1603.02444); the phase is not
  maximised, as for precessing binaries with higher modes it is a
  parameter of the binary;
* :func:`with_conventions`, :func:`optimize`: a binary with its reference
  phase and the azimuths of its in-plane spins shifted, the parameters
  whose meaning differs between models, and a minimisation from several
  starts (SciPy's L-BFGS-B with the gradient from ``jax.grad``).

>>> import numpy as np, jax.numpy as jnp
>>> import waveformx as wf
>>> frequencies = np.arange(160, 8193) / 8.0
>>> psd = jnp.ones_like(frequencies)
>>> f = wf.polarizations_function("IMRPhenomXPNR")
>>> binary = wf.Binary(36.0, 29.0, jnp.array([0.4, 0.2, 0.3]), jnp.array([-0.2, 0.3, -0.1]),
...                    500.0, 0.7, 1.2, 20.0)
>>> hp, hc = f(binary, frequencies)
>>> wf.match.mismatch(hp, hp * jnp.exp(2j * jnp.pi * frequencies * 0.01 + 0.3j), frequencies, psd)  # ~0
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._binary import Binary

__all__ = [
    "inner_product",
    "match",
    "mismatch",
    "polarization_match",
    "polarization_mismatch",
    "with_conventions",
    "optimize",
    "OptimizeResult",
]

#: Newton steps that refine the best time of the FFT grid.
NEWTON_STEPS = 3


def _grid(frequencies):
    frequencies = jnp.asarray(frequencies)
    return frequencies, frequencies[1] - frequencies[0]


def inner_product(a, b, frequencies, psd):
    """``<a, b> = 4 delta_f Re sum a conj(b) / S`` (no maximisation)."""
    frequencies, delta_f = _grid(frequencies)
    return 4 * delta_f * jnp.sum((a * jnp.conj(b)).real / psd)


def _overlap(a, b, frequencies, psd):
    """The overlap ``z(t)`` as a function of the time shift."""
    frequencies, delta_f = _grid(frequencies)
    c = 4 * delta_f * a * jnp.conj(b) / psd
    return c, lambda t: jnp.sum(c * jnp.exp(2j * jnp.pi * frequencies * t))


def _series(c, frequencies, pad):
    """``z`` on the times ``j / (n delta_f)`` of one period ``1 / delta_f``
    (``n = pad`` times the power of two above the number of frequencies), and
    those times."""
    frequencies, delta_f = _grid(frequencies)
    k = c.shape[-1]
    n = pad * (1 << (k - 1).bit_length())
    times = jnp.arange(n) / (n * delta_f)
    # sum_k c_k e^(2 pi i (f_0 + k delta_f) t_j)
    z = n * jnp.fft.ifft(c, n) * jnp.exp(2j * jnp.pi * frequencies[0] * times)
    return times, z


def _refine(objective, t):
    """Newton steps towards the maximum of ``objective`` from ``t``, each
    kept only if it increases the objective."""
    grad = jax.grad(objective)
    hess = jax.grad(grad)

    def body(_, t):
        g, h = grad(t), hess(t)
        new = t - jnp.where(h < 0, g / jnp.where(h < 0, h, 1.0), 0.0)
        return jnp.where(objective(new) >= objective(t), new, t)

    return jax.lax.fori_loop(0, NEWTON_STEPS, body, t)


def _best_time(values, times, objective, time):
    if not time:
        return jnp.zeros((), times.dtype)
    t = times[jnp.argmax(values)]
    # the period is 1 / delta_f: shifts beyond half of it are negative ones
    period = times[-1] + times[1]
    t = jnp.where(t > period / 2, t - period, t)
    return jax.lax.stop_gradient(_refine(objective, jax.lax.stop_gradient(t)))


def match(a, b, frequencies, psd, *, time=True, phase=True, pad=4):
    """The match of ``a`` and ``b``: ``<a, b> / sqrt(<a, a> <b, b>)``
    maximised over a time shift (``time``) and a constant phase (``phase``,
    then ``|z|``), with ``z`` on a grid of ``pad`` times as many times as
    frequencies (rounded up to a power of two) before the Newton steps.
    ``frequencies`` must be uniformly spaced."""
    norm = jnp.sqrt(inner_product(a, a, frequencies, psd) * inner_product(b, b, frequencies, psd))
    c, z = _overlap(a, b, frequencies, psd)
    value = (lambda t: jnp.abs(z(t)) ** 2) if phase else (lambda t: z(t).real)
    if time:
        times, series = _series(c, frequencies, pad)
        grid = jnp.abs(series) if phase else series.real
    else:
        times, grid = jnp.zeros(2), None
    t = _best_time(grid, times, value, time)
    best = jnp.abs(z(t)) if phase else z(t).real
    return best / norm


def mismatch(a, b, frequencies, psd, **keywords):
    """``1 - match(a, b, ...)`` (see :func:`match`)."""
    return 1 - match(a, b, frequencies, psd, **keywords)


def polarization_match(signal, hplus, hcross, frequencies, psd, *, time=True, pad=4):
    """The match of ``signal`` with the template ``F_+ hplus + F_x hcross``
    maximised over time and over ``F_+`` and ``F_x``: the norm of the
    projection of ``signal`` onto the real span of the two polarizations,
    over the norm of ``signal`` (Harry et al. 2016)::

        sqrt((rho_+^2 + rho_x^2 - 2 rho_+ rho_x I) / (1 - I^2)) / |signal|,

    with ``rho_+, rho_x`` the real overlaps of ``signal`` with the normalised
    polarizations and ``I`` the overlap of those. Where the polarizations are
    (numerically) parallel, as for an edge-on, non-precessing binary, the
    match with ``hplus`` alone is taken."""
    ss = inner_product(signal, signal, frequencies, psd)
    pp = inner_product(hplus, hplus, frequencies, psd)
    xx = inner_product(hcross, hcross, frequencies, psd)
    px = inner_product(hplus, hcross, frequencies, psd)
    i = px / jnp.sqrt(pp * xx)
    degenerate = ~(1 - i**2 > 1e-10)
    i = jnp.where(degenerate, 0.0, i)
    xx = jnp.where(degenerate, 1.0, xx)
    cp, zp = _overlap(signal, hplus, frequencies, psd)
    cx, zx = _overlap(signal, hcross, frequencies, psd)
    cx = jnp.where(degenerate, 0.0, cx)

    def square(rp, rx):
        rp, rx = rp / jnp.sqrt(pp), jnp.where(degenerate, 0.0, rx / jnp.sqrt(xx))
        return (rp**2 + rx**2 - 2 * rp * rx * i) / (1 - i**2)

    value = lambda t: square(zp(t).real, zx(t).real)  # noqa: E731
    if time:
        times, sp = _series(cp, frequencies, pad)
        _, sx = _series(cx, frequencies, pad)
        grid = square(sp.real, sx.real)
    else:
        times, grid = jnp.zeros(2), None
    t = _best_time(grid, times, value, time)
    return jnp.sqrt(value(t) / ss)


def polarization_mismatch(signal, hplus, hcross, frequencies, psd, **keywords):
    """``1 - polarization_match(...)`` (see :func:`polarization_match`)."""
    return 1 - polarization_match(signal, hplus, hcross, frequencies, psd, **keywords)


def _rotate(spin, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2]])


def with_conventions(binary: Binary, shifts) -> Binary:
    """``binary`` with ``shifts = (delta phi_ref, delta alpha_1, delta
    alpha_2)`` added to its reference phase and to the azimuths of its two
    spins about the orbital angular momentum (at the reference frequency):
    the parameters whose definitions differ between models (the orbital
    phase at the reference frequency, the frame in which the spins are
    given)."""
    shifts = jnp.asarray(shifts)
    return binary._replace(
        phi_ref=binary.phi_ref + shifts[0],
        spin_1=_rotate(jnp.asarray(binary.spin_1), shifts[1]),
        spin_2=_rotate(jnp.asarray(binary.spin_2), shifts[2]),
    )


@functools.lru_cache(maxsize=16)
def _value_and_grad(objective):
    return jax.jit(jax.value_and_grad(objective))


class OptimizeResult(NamedTuple):
    """The result of :func:`optimize`: the best point and value, and the
    points and values reached from every start."""

    x: np.ndarray
    fun: float
    xs: np.ndarray
    funs: np.ndarray


def optimize(objective: Callable, starts, args=(), *, maxiter: int = 100, tolerance: float = 1e-12) -> OptimizeResult:
    """Minimise ``objective(x, *args)``, a scalar JAX function of a
    vector ``x`` (and of JAX arrays or pytrees ``args``), from each row of
    ``starts`` with SciPy's L-BFGS-B and the gradient from ``jax.grad``, and
    keep the best. The objective and its gradient are compiled once per
    ``objective`` (``jax.jit``; pass what changes between calls, such as
    the signal, in ``args`` to reuse the compilation) and evaluated on the
    host. Mismatches over the convention parameters are multimodal: give
    several starts.

    >>> def objective(shifts):
    ...     hp, hc = f(with_conventions(binary, shifts), frequencies)
    ...     return polarization_mismatch(signal, hp, hc, frequencies, psd)
    >>> result = optimize(objective, [[0, 0, 0], [np.pi, 0, 0]])
    """
    import scipy.optimize

    value_and_grad = _value_and_grad(objective)

    def fun(x):
        value, grad = value_and_grad(jnp.asarray(x, jnp.float64), *args)
        return float(value), np.asarray(grad, dtype=np.float64)

    starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
    xs, funs = [], []
    for x0 in starts:
        out = scipy.optimize.minimize(fun, x0, jac=True, method="L-BFGS-B",
                                      options=dict(maxiter=maxiter, ftol=tolerance, gtol=tolerance))
        xs.append(out.x)
        funs.append(out.fun if np.isfinite(out.fun) else np.inf)
    xs, funs = np.array(xs), np.array(funs)
    best = int(np.argmin(funs))
    return OptimizeResult(xs[best], float(funs[best]), xs, funs)
