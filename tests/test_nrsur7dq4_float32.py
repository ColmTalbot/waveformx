# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in float32 and the scale of its output.

A binary of float32 parameters is evaluated in float32 throughout (no
operation, nor its derivative, in float64), close to the float64
evaluation, under ``jax.jit``, ``jax.vmap`` and ``jax.grad``. Every output
is the strain multiplied by ``1e21`` unless ``scaled_strain=False``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import requires_nrsur

pytestmark = requires_nrsur

BINARIES = {
    "precessing": wf.Binary(40.0, 20.0, (0.4, -0.3, 0.2), (0.1, 0.5, -0.3), 400.0, 0.7, 0.1, 20.0),
    "m1<m2, f_ref=0": wf.Binary(12.0, 48.0, (0.7, 0.0, 0.7), (0.0, -0.9, 0.0), 1500.0, 2.1, -0.4, 0.0),
    "heavy, q=6": wf.Binary(240.0, 40.0, (0.0, 0.5, -0.5), (0.3, 0.3, 0.3), 3000.0, 1.3, 2.0, 15.0),
    "light, aligned": wf.Binary(14.0, 7.0, (0.0, 0.0, 0.6), (0.0, 0.0, -0.4), 100.0, 0.4, 0.5, 0.0),
}
TIMES = np.arange(-0.4, 0.03, 1 / 4096)
OPTIONS = wf.WaveformOptions()
MODEL = wf.models.get("NRSur7dq4")


def as_dtype(binary, dtype):
    return wf.Binary(*(jnp.asarray(value, dtype) for value in binary))


def produced(jaxpr):
    """The dtypes produced by every operation of a jaxpr, recursively."""
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            yield var.aval.dtype
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else [param]:
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    yield from produced(inner if hasattr(inner, "eqns") else inner.jaxpr)


def test_float32_data():
    from waveformjax.models.nrsur7dq4 import _data, default_data, load

    data = load(dtype=jnp.float32)
    assert load(None, np.float32) is data and default_data(jnp.float32) is data
    assert data.dtype == jnp.float32
    reference = default_data()
    for leaf, wide in zip(jax.tree_util.tree_leaves(data), jax.tree_util.tree_leaves(reference)):
        if jnp.issubdtype(wide.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(wide).astype(np.float32))
        else:
            assert leaf.dtype == wide.dtype
    with pytest.raises(TypeError):
        load(dtype=np.float16)
    assert _data.precision(jnp.float32) == jnp.float32 and _data.precision(None) == jnp.float64


@pytest.mark.parametrize("method", ["polarizations_at", "modes_at"])
def test_float32_throughout(method):
    """No operation of the waveform or of its gradient is in float64."""
    function = getattr(MODEL, method)
    binary = as_dtype(BINARIES["precessing"], jnp.float32)
    out = jax.jit(lambda b: function(b, TIMES, OPTIONS))(binary)
    for leaf in jax.tree_util.tree_leaves(out):
        assert leaf.dtype in (jnp.float32, jnp.complex64)

    def power(b):
        return sum(jnp.sum(jnp.abs(h) ** 2) for h in jax.tree_util.tree_leaves(function(b, TIMES, OPTIONS)))

    for jaxpr in (jax.make_jaxpr(lambda b: function(b, TIMES, OPTIONS))(binary), jax.make_jaxpr(jax.grad(power))(binary)):
        wide = [d for d in produced(jaxpr.jaxpr) if d in (np.float64, np.complex128)]
        assert not wide


def test_precision_follows_the_binary():
    """Python numbers take the type of the other parameters; float64 data
    are converted to the precision of the binary."""
    from waveformjax.models.nrsur7dq4 import default_data

    binary = BINARIES["precessing"]._replace(mass_1=jnp.float32(40.0), mass_2=jnp.float32(20.0),
                                             spin_1=jnp.asarray((0.4, -0.3, 0.2), jnp.float32),
                                             spin_2=jnp.asarray((0.1, 0.5, -0.3), jnp.float32))
    hp, _ = MODEL.polarizations_at(binary, TIMES, OPTIONS)
    assert hp.dtype == jnp.float32
    hp_data, _ = MODEL.polarizations_at(binary, TIMES, OPTIONS, data=default_data())
    np.testing.assert_array_equal(np.asarray(hp_data), np.asarray(hp))
    hp, _ = MODEL.polarizations_at(BINARIES["precessing"], TIMES, OPTIONS)
    assert hp.dtype == jnp.float64


@pytest.mark.parametrize("name", list(BINARIES))
def test_float32_close_to_float64(name):
    """The float32 waveform agrees with the float64 one to ~1e-4."""
    single = MODEL.polarizations_at(as_dtype(BINARIES[name], jnp.float32), TIMES, OPTIONS)
    double = MODEL.polarizations_at(as_dtype(BINARIES[name], jnp.float64), TIMES, OPTIONS)
    for a, b in zip(single, double):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b)
        assert np.all(np.isfinite(a))
        assert np.max(np.abs(a - b)) < 1e-3 * np.max(np.abs(b))
        assert np.linalg.norm(a - b) < 3e-4 * np.linalg.norm(b)
        assert 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)) < 1e-7


def test_float32_vmap_matches_loop():
    """The batched evaluation sums in another order; the dynamics amplify
    rounding by ~1e3, as in float64 (test_transforms_nrsur7dq4.py)."""
    binaries = [as_dtype(binary, jnp.float32) for binary in BINARIES.values()]
    stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *binaries)
    function = wf.td_polarizations_function("NRSur7dq4")
    batched = jax.jit(jax.vmap(function, in_axes=(0, None)))(stacked, TIMES)
    single = jax.jit(function)
    for i, binary in enumerate(binaries):
        for a, b in zip(batched, single(binary, TIMES)):
            assert a.dtype == jnp.float32
            assert np.max(np.abs(np.asarray(a[i]) - np.asarray(b))) < 1e-3 * np.max(np.abs(np.asarray(b)))


@pytest.mark.parametrize("name", ["precessing", "m1<m2, f_ref=0", "heavy, q=6"])
def test_float32_gradient(name):
    """The jitted float32 gradient of a noise-free log-likelihood (which
    depends on every parameter, the distance included) agrees with the
    float64 one."""
    function = wf.td_polarizations_function("NRSur7dq4")
    data = function(as_dtype(BINARIES[name]._replace(mass_1=BINARIES[name].mass_1 * 1.01), jnp.float64), TIMES)

    def log_likelihood(binary):
        hp, hc = function(binary, TIMES)
        d = [x.astype(hp.dtype) for x in data]
        return jnp.sum(hp * d[0] + hc * d[1]) - 0.5 * jnp.sum(hp * hp + hc * hc)

    gradient = jax.jit(jax.grad(log_likelihood))
    single = jnp.concatenate([jnp.ravel(x) for x in gradient(as_dtype(BINARIES[name], jnp.float32))])
    double = jnp.concatenate([jnp.ravel(x) for x in gradient(as_dtype(BINARIES[name], jnp.float64))])
    single, double = np.asarray(single, dtype=np.float64), np.asarray(double)
    assert np.all(np.isfinite(single))
    scale = np.max(np.abs(double))
    assert np.all(np.abs(single - double) <= 1e-2 * np.abs(double) + 1e-3 * scale), (single, double)


def test_float32_distance_derivative():
    """d<h, h>/dD = -2 <h, h> / D in float32 (with the distance in metres
    the derivative underflows to zero)."""
    binary = as_dtype(BINARIES["precessing"], jnp.float32)
    function = wf.td_polarizations_function("NRSur7dq4")

    def power(binary):
        hp, hc = function(binary, TIMES)
        return jnp.sum(hp * hp + hc * hc)

    value, gradient = jax.jit(jax.value_and_grad(power))(binary)
    expected = -2 * float(value) / float(binary.luminosity_distance)
    assert abs(float(gradient.luminosity_distance) - expected) < 1e-4 * abs(expected)


def test_scaled_strain():
    """Every output is the strain times 1e21 unless scaled_strain=False."""
    physical = OPTIONS.replace(scaled_strain=False)
    binary = BINARIES["precessing"]
    for method in (MODEL.polarizations_at, MODEL.modes_at):
        for a, b in zip(jax.tree_util.tree_leaves(method(binary, TIMES, OPTIONS)),
                        jax.tree_util.tree_leaves(method(binary, TIMES, physical))):
            np.testing.assert_allclose(np.asarray(a), 1e21 * np.asarray(b), rtol=1e-13,
                                       atol=1e-13 * float(np.max(np.abs(a))))
    keywords = dict(zip(wf.Binary._fields, binary))
    for function in (wf.td_polarizations, wf.td_modes, wf.simulate_td):
        scaled = function("NRSur7dq4", 1 / 2048, 20.0, **keywords)
        strain = function("NRSur7dq4", 1 / 2048, 20.0, scaled_strain=False, **keywords)
        assert scaled.epoch == strain.epoch
        series = (scaled.modes.values(), strain.modes.values()) if hasattr(scaled, "modes") else (
            (scaled.hplus, scaled.hcross), (strain.hplus, strain.hcross))
        for a, b in zip(*series):
            a, b = np.asarray(a), np.asarray(b)
            assert np.max(np.abs(a - 1e21 * b)) < 1e-13 * np.max(np.abs(a))
    batch = wf.batch_td_polarizations("NRSur7dq4", 1 / 2048, 20.0, {k: [v] for k, v in keywords.items()})
    single = wf.td_polarizations("NRSur7dq4", 1 / 2048, 20.0, **keywords)
    assert np.max(np.abs(np.asarray(batch[0].hplus) - np.asarray(single.hplus))) < 1e-12 * np.max(
        np.abs(np.asarray(single.hplus)))
    scaled = wf.simulate_fd("NRSur7dq4", 0.0, 20.0, 1024.0, **keywords)
    strain = wf.simulate_fd("NRSur7dq4", 0.0, 20.0, 1024.0, scaled_strain=False, **keywords)
    assert np.max(np.abs(np.asarray(scaled.hplus) - 1e21 * np.asarray(strain.hplus))) < 1e-13 * np.max(
        np.abs(np.asarray(scaled.hplus)))
