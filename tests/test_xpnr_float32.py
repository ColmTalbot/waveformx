# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR in float32 and the scale of its output.

A binary of float32 parameters is evaluated in float32: no float32 value is
promoted to float64 (the weakly typed constants of LAL's expressions are
float64 scalars that take the type of the values they meet). The extended
model is close to its float64 evaluation, under ``jax.jit``, ``jax.jvp``
and ``jax.grad``. The default model, which replays LAL's integration step by
step, runs in float32 but is much less accurate there. Every output is the
strain multiplied by ``1e21`` unless ``scaled_strain=False``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from waveformjax import WaveformOptions
from waveformjax.models.xpnr import model

BINARY = wf.Binary(36.0, 29.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0)
FREQUENCIES = np.geomspace(20.0, 1024.0, 128)
DOMINANT = ((2, 2), (2, -2))
EXTENDED = WaveformOptions(extended_spintaylor=True, mode_array=DOMINANT)
NARROW = (np.dtype(np.float32), np.dtype(np.complex64))
WIDE = (np.dtype(np.float64), np.dtype(np.complex128))


def as_dtype(binary, dtype):
    return wf.Binary(*(jnp.asarray(value, dtype) for value in binary))


def promotions(jaxpr):
    """The conversions of float32 values to float64 in a jaxpr
    (recursively): where JAX promotes a float32 operand."""
    for eqn in jaxpr.eqns:
        if (eqn.primitive.name == "convert_element_type" and eqn.invars[0].aval.dtype in NARROW
                and eqn.outvars[0].aval.dtype in WIDE):
            yield eqn
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else [param]:
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    yield from promotions(inner if hasattr(inner, "eqns") else inner.jaxpr)


def mismatch(a, b):
    return 1 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


@functools.lru_cache(maxsize=None)
def evaluate(options):
    return jax.jit(lambda b, f: model._polarizations(b, f, options))


@pytest.mark.parametrize("options", [WaveformOptions(), EXTENDED], ids=["default", "extended"])
def test_float32_throughout(options):
    """No float32 value of the waveform, its forward derivative or its
    gradient is promoted to float64."""
    binary = as_dtype(BINARY, jnp.float32)
    frequencies = jnp.asarray(FREQUENCIES, jnp.float32)
    tangent = jax.tree_util.tree_map(jnp.ones_like, binary)
    traces = [jax.make_jaxpr(lambda b: jax.jvp(lambda b: model._polarizations(b, frequencies, options), (b,),
                                                (tangent,)))(binary)]
    if options == EXTENDED:
        def power(b):
            return sum(jnp.sum(jnp.abs(h) ** 2) for h in model.polarizations(b, frequencies, options))

        traces.append(jax.make_jaxpr(jax.grad(power))(binary))
    for jaxpr in traces:
        promoted = list(promotions(jaxpr.jaxpr))
        assert not promoted, [str(eqn)[:200] for eqn in promoted[:5]]


def test_float32_close_to_float64():
    """The extended model in float32 agrees with its float64 evaluation to a
    mismatch far below 1e-6; the default model runs and is finite."""
    function = evaluate(EXTENDED)
    single = function(as_dtype(BINARY, jnp.float32), jnp.asarray(FREQUENCIES, jnp.float32))
    double = function(as_dtype(BINARY, jnp.float64), jnp.asarray(FREQUENCIES))
    for a, b in zip(single, double):
        assert a.dtype == jnp.complex64
        a, b = np.asarray(a, dtype=np.complex128), np.asarray(b)
        assert np.all(np.isfinite(a))
        assert mismatch(a, b) < 1e-7
    for h in evaluate(WaveformOptions())(as_dtype(BINARY, jnp.float32), jnp.asarray(FREQUENCIES, jnp.float32)):
        assert h.dtype == jnp.complex64 and np.all(np.isfinite(np.asarray(h)))


def test_float32_gradient():
    """The float32 gradient of a noise-free log-likelihood (which depends on
    every parameter, the distance included) agrees with the float64 one."""
    function = wf.polarizations_function("IMRPhenomXPNR", extended_spintaylor=True, mode_array=DOMINANT)
    data = function(BINARY._replace(mass_1=BINARY.mass_1 * 1.01), FREQUENCIES)

    def log_likelihood(binary, frequencies):
        hp, hc = function(binary, frequencies)
        d = [x.astype(hp.dtype) for x in data]
        overlap = jnp.vdot(d[0], hp) + jnp.vdot(d[1], hc)
        return jnp.real(overlap) - 0.5 * jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    gradient = jax.jit(jax.grad(log_likelihood))
    single = gradient(as_dtype(BINARY, jnp.float32), jnp.asarray(FREQUENCIES, jnp.float32))
    double = gradient(as_dtype(BINARY, jnp.float64), jnp.asarray(FREQUENCIES))
    single = np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in single])
    double = np.concatenate([np.ravel(np.asarray(x)) for x in double])
    assert np.all(np.isfinite(single))
    scale = np.max(np.abs(double))
    assert np.all(np.abs(single - double) <= 1e-2 * np.abs(double) + 1e-3 * scale), (single, double)


def test_scaled_strain():
    """The output is the strain times 1e21 unless scaled_strain=False."""
    keywords = dict(zip(wf.Binary._fields, BINARY))
    scaled = wf.polarizations("IMRPhenomXPNR", FREQUENCIES, mode_array=DOMINANT, **keywords)
    strain = wf.polarizations("IMRPhenomXPNR", FREQUENCIES, mode_array=DOMINANT, scaled_strain=False, **keywords)
    for a, b in zip(scaled, strain):
        a, b = np.asarray(a), np.asarray(b)
        assert np.max(np.abs(a - 1e21 * b)) < 1e-13 * np.max(np.abs(a))
