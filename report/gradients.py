#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The derivatives of the models against finite differences, and their
cost.

For each model and a few binaries this computes the Jacobian of both
polarizations with respect to the twelve parameters of the binary
(``jax.jacfwd``) and compares every column with central finite differences
of the compiled model: the error of a column is ``max |J_i - D_i| / max
|J_i|`` over the frequencies (or times) and both polarizations. Two
differences are taken, the second-order one with a step of 1e-6 and the
fourth-order one with a step of 1e-4 (relative steps for the masses, the
distance and ``f_ref``), and the smaller error is kept: the first is
limited by rounding, the second by the discontinuities of the model (see
the report). The spin components of a body with no spin are reported
apart: the waveform is not differentiable there. It also checks that ``jax.grad`` of a scalar is the Jacobian
contracted with the scalar's weights, and times the compilation and the
evaluation of the model, ``jax.jvp``, ``jax.jacfwd``, ``jax.grad`` and
``jax.vmap`` of ``jax.grad``.

Models: IMRPhenomXPNR (``polarizations``, 2048 frequencies from 20 to
1024 Hz), NRSur7dq4 (``polarizations_at``, 2048 samples at 4096 Hz from
0.45 s before the peak to 0.05 s after it) and the conversions (``simulate_td_function``
of IMRPhenomXPNR's (2,2) mode, ``simulate_fd_function`` of NRSur7dq4; one
binary each).

Usage::

    python report/gradients.py [--models xpnr nrsur7dq4 conversions]
"""

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import NAMES, binary_of_vector, environment, median_time, vector_of, wf, write

RELATIVE = {"mass_1", "mass_2", "luminosity_distance", "f_ref"}

XPNR_BINARIES = {
    "36+29, precessing": dict(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1),
                              luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0),
    "40+20, precessing": dict(mass_1=40.0, mass_2=20.0, spin_1=(0.4, -0.3, 0.2), spin_2=(0.1, 0.5, -0.3),
                              luminosity_distance=100.0, inclination=2.1, phi_ref=0.1, f_ref=25.0),
    "q = 1/6, strongly precessing": dict(mass_1=42.0, mass_2=7.0, spin_1=(0.7, 0.3, -0.4), spin_2=(0.0, 0.2, 0.1),
                                         luminosity_distance=800.0, inclination=1.2, phi_ref=2.5, f_ref=20.0),
    "nearly aligned spins": dict(mass_1=30.0, mass_2=25.0, spin_1=(1e-3, -2e-3, 0.5), spin_2=(2e-3, 1e-3, -0.3),
                                 luminosity_distance=400.0, inclination=0.3, phi_ref=0.4, f_ref=20.0),
}
NRSUR_BINARIES = {
    "40+20, precessing": dict(mass_1=40.0, mass_2=20.0, spin_1=(0.4, -0.3, 0.2), spin_2=(0.1, 0.5, -0.3),
                              luminosity_distance=100.0, inclination=0.7, phi_ref=0.1, f_ref=20.0),
    "lighter first, q = 1/3": dict(mass_1=20.0, mass_2=60.0, spin_1=(-0.5, 0.2, 0.3), spin_2=(0.3, 0.3, -0.6),
                                   luminosity_distance=300.0, inclination=2.4, phi_ref=1.0, f_ref=18.0),
    "aligned spins": dict(mass_1=50.0, mass_2=30.0, spin_1=(0.0, 0.0, 0.6), spin_2=(0.0, 0.0, -0.4),
                          luminosity_distance=200.0, inclination=1.0, phi_ref=0.3, f_ref=20.0),
    "non-spinning": dict(mass_1=35.0, mass_2=30.0, spin_1=(0.0, 0.0, 0.0), spin_2=(0.0, 0.0, 0.0),
                         luminosity_distance=200.0, inclination=0.5, phi_ref=0.0, f_ref=20.0),
}


def steps(x, h):
    return np.array([h * (abs(float(x[i])) if name in RELATIVE else 1.0) for i, name in enumerate(NAMES)])


def finite_differences(f, x, h, order):
    """Columns of the Jacobian of ``f`` (returning a stacked array) by
    central differences of ``order`` 2 or 4."""
    columns = []
    for i, step in enumerate(steps(x, h)):
        def at(k):
            return np.asarray(f(x.at[i].add(k * step)))
        if order == 2:
            columns.append((at(1) - at(-1)) / (2 * step))
        else:
            columns.append((-at(2) + 8 * at(1) - 8 * at(-1) + at(-2)) / (12 * step))
    return np.stack(columns, axis=-1)


def column_errors(jacobian, numeric):
    out = []
    for i in range(jacobian.shape[-1]):
        scale = np.max(np.abs(jacobian[..., i]))
        out.append(float(np.max(np.abs(jacobian[..., i] - numeric[..., i])) / scale) if scale > 0 else
                   float(np.max(np.abs(numeric[..., i]))))
    return out


def check(name, f, binaries, timings_for=None):
    """Jacobian checks for ``f(x) -> stacked polarizations`` at each binary,
    and the costs (``timings_for``) at the first, measured first so that the
    compilation times are those of fresh compilations.

    The spin components of a body whose spin vector is zero are reported
    apart: there the waveform is not differentiable (NRSur7dq4 rescales the
    spins to their initial magnitude at every step, a factor that depends on
    the direction of an infinitesimal spin)."""
    call = jax.jit(f)
    jac = jax.jit(jax.jacfwd(f))
    timings = {}
    if timings_for is not None:
        timings = timings_for(f, vector_of(next(iter(binaries.values()))), call, jac)
    results = {}
    for label, keywords in binaries.items():
        x = vector_of(keywords)
        jacobian = np.asarray(jax.block_until_ready(jac(x)))
        second = column_errors(jacobian, finite_differences(call, x, 1e-6, 2))
        fourth = column_errors(jacobian, finite_differences(call, x, 1e-4, 4))
        best = [min(a, b) for a, b in zip(second, fourth)]
        singular = [f"spin_{body}{c}" for body in (1, 2) if not np.any(keywords[f"spin_{body}"]) for c in "xyz"]
        smooth = [e for n, e in zip(NAMES, best) if n not in singular]
        results[label] = dict(errors=dict(zip(NAMES, best)), second_order=dict(zip(NAMES, second)),
                              fourth_order=dict(zip(NAMES, fourth)), not_differentiable=singular,
                              max_error=max(smooth), finite=bool(np.all(np.isfinite(jacobian))))
        worst = max((n for n in NAMES if n not in singular), key=lambda n: results[label]["errors"][n])
        print(f"{name} {label}: max column error {max(smooth):.1e} (worst {worst}), finite {results[label]['finite']}"
              + (f"; not differentiable in {', '.join(singular)}" if singular else ""), flush=True)
    return dict(binaries=results, timings=timings)


def cost(f, x, call, jac, batch=8):
    """Compilation and evaluation times of the model (``call``) and its
    derivatives at ``x``, and the consistency of ``jax.grad`` with the
    Jacobian (``jac``)."""
    start = time.perf_counter()
    reference = jax.block_until_ready(call(x))
    out = {"forward_compile": time.perf_counter() - start}
    rng = np.random.default_rng(1)
    w_re = jnp.asarray(rng.normal(size=jnp.shape(reference)))
    w_im = jnp.asarray(rng.normal(size=jnp.shape(reference)))
    scale = 1.0 / float(jnp.max(jnp.abs(reference)))

    def loss(x):
        """A real scalar: weighted sums of the real and imaginary parts."""
        h = f(x)
        return (jnp.sum(w_re * jnp.real(h)) + jnp.sum(w_im * jnp.imag(h))) * scale

    direction = jnp.asarray(np.random.default_rng(2).normal(size=x.shape)) * 1e-3
    xs = x + jnp.asarray(np.random.default_rng(3).uniform(-1e-2, 1e-2, size=(batch,) + x.shape)) * jnp.abs(x)
    functions = {
        "forward": (call, (x,)),
        "jvp": (jax.jit(lambda x, t: jax.jvp(f, (x,), (t,))), (x, direction)),
        "jacfwd": (jac, (x,)),
        "grad": (jax.jit(jax.grad(loss)), (x,)),
        f"vmap_grad_{batch}": (jax.jit(jax.vmap(jax.grad(loss))), (xs,)),
    }
    for key, (function, args) in functions.items():
        if key != "forward":
            start = time.perf_counter()
            jax.block_until_ready(function(*args))
            out[f"{key}_compile"] = time.perf_counter() - start
        out[key] = median_time(lambda: function(*args), repeats=5)
        print(f"  {key}: first call {out[f'{key}_compile']:.1f} s, then {out[key] * 1e3:.2f} ms", flush=True)
    gradient = np.asarray(functions["grad"][0](x))
    jacobian = np.asarray(jac(x))
    axes = jacobian.ndim - 1
    contracted = (np.tensordot(np.asarray(w_re), jacobian.real, axes=axes)
                  + np.tensordot(np.asarray(w_im), jacobian.imag, axes=axes)) * scale
    out["grad_vs_jacobian"] = float(np.max(np.abs(gradient - contracted)) / np.max(np.abs(contracted)))
    out["batch"] = batch
    return out


def stacked(function):
    """``x -> [h_+, h_x]`` (complex) as one array."""
    def f(x):
        hp, hc = function(binary_of_vector(x))
        return jnp.stack([hp, hc])
    return f


def xpnr():
    frequencies = jnp.asarray(np.linspace(20.0, 1024.0, 2048))
    model = wf.polarizations_function("IMRPhenomXPNR")
    return check("IMRPhenomXPNR", stacked(lambda b: model(b, frequencies)), XPNR_BINARIES, cost)


def nrsur7dq4():
    times = jnp.asarray(np.arange(-2048 + 205, 205) / 4096.0)
    model = wf.td_polarizations_function("NRSur7dq4")
    return check("NRSur7dq4", stacked(lambda b: model(b, times)), NRSUR_BINARIES, cost)


def conversions():
    out = {}
    td = wf.simulate_td_function("IMRPhenomXPNR", 1 / 1024, 25.0, mode_array=((2, 2), (2, -2)),
                                 **dict(XPNR_BINARIES["36+29, precessing"], f_ref=25.0))
    out["simulate_td IMRPhenomXPNR (2,2)"] = check(
        "simulate_td", stacked(td), {"36+29, precessing": dict(XPNR_BINARIES["36+29, precessing"], f_ref=25.0)},
        functools.partial(cost, batch=4))
    if wf.nrsur7dq4_data_file() is not None:
        fd = wf.simulate_fd_function("NRSur7dq4", 0.0, 20.0, 1024.0, **NRSUR_BINARIES["40+20, precessing"])
        out["simulate_fd NRSur7dq4"] = check(
            "simulate_fd", stacked(fd), {"40+20, precessing": NRSUR_BINARIES["40+20, precessing"]},
            functools.partial(cost, batch=4))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=["xpnr", "nrsur7dq4", "conversions"])
    parser.add_argument("--output", default="gradients.json")
    args = parser.parse_args()
    out = dict(environment=environment(), parameters=NAMES)
    for name in args.models:
        out[name] = {"xpnr": xpnr, "nrsur7dq4": nrsur7dq4, "conversions": conversions}[name]()
    write(args.output, out)


if __name__ == "__main__":
    main()
