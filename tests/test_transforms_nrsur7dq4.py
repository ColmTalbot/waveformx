# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 under the JAX transformations: ``jax.jit`` and ``jax.vmap``
give the plain call, ``jax.grad`` agrees with finite differences for every
parameter of the binary and is finite where LAL's formulas have removable
singularities (aligned or vanishing spins), and repeated evaluations reuse
the compiled function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from conftest import requires_nrsur

pytestmark = requires_nrsur

PRECESSING = wf.Binary(
    mass_1=40.0,
    mass_2=20.0,
    spin_1=jnp.array([0.4, -0.3, 0.2]),
    spin_2=jnp.array([0.1, 0.5, -0.3]),
    luminosity_distance=400.0,
    inclination=0.7,
    phi_ref=0.1,
    f_ref=20.0,
)
NEARLY_ALIGNED = wf.Binary(
    mass_1=28.0,
    mass_2=22.0,
    spin_1=jnp.array([1e-3, -2e-3, 0.4]),
    spin_2=jnp.array([2e-3, 1e-3, -0.3]),
    luminosity_distance=300.0,
    inclination=1.1,
    phi_ref=2.3,
    f_ref=25.0,
)
# the last ~0.25 s and the ringdown, sampled at 4096 Hz
TIMES = jnp.arange(-0.25, 0.03, 1 / 4096)


def as_floats(binary):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), binary)


@pytest.fixture(scope="module")
def function():
    return wf.td_polarizations_function("NRSur7dq4")


def test_jit_matches_plain_call(function):
    plain = function(as_floats(PRECESSING), TIMES)
    compiled = jax.jit(function)(as_floats(PRECESSING), TIMES)
    scale = np.max(np.abs(plain[0]))
    for a, b in zip(plain, compiled):
        assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-13 * scale


def test_data_first_read_under_jit(function):
    """The data read by a first call inside ``jax.jit`` are concrete arrays,
    so later calls (outside that trace) can use them."""
    from waveformjax.models.nrsur7dq4 import _data

    _data._load_as.cache_clear()
    _data._load_cached.cache_clear()
    compiled = jax.jit(function)(as_floats(PRECESSING), TIMES)
    plain = function(as_floats(PRECESSING), TIMES)
    scale = np.max(np.abs(plain[0]))
    for a, b in zip(plain, compiled):
        assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-13 * scale
    assert all(isinstance(leaf, (jax.Array, int, tuple)) and not isinstance(leaf, jax.core.Tracer)
               for leaf in jax.tree_util.tree_leaves(_data.default_data()))


def test_vmap_matches_loop(function):
    binaries = [as_floats(PRECESSING), as_floats(NEARLY_ALIGNED), as_floats(PRECESSING._replace(mass_1=25.0, f_ref=0.0))]
    stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *binaries)
    batched = jax.jit(jax.vmap(function, in_axes=(0, None)))(stacked, TIMES)
    single = jax.jit(function)
    for i, binary in enumerate(binaries):
        hp, hc = single(binary, TIMES)
        scale = np.max(np.abs(hp))
        assert np.max(np.abs(np.asarray(batched[0][i]) - np.asarray(hp))) < 1e-13 * scale
        assert np.max(np.abs(np.asarray(batched[1][i]) - np.asarray(hc))) < 1e-13 * scale


# parameter, component, finite-difference step
PARAMETERS = [
    ("mass_1", None, 1e-3),
    ("mass_2", None, 1e-3),
    ("spin_1", 0, 1e-4),
    ("spin_1", 1, 1e-4),
    ("spin_1", 2, 1e-4),
    ("spin_2", 0, 1e-4),
    ("spin_2", 1, 1e-4),
    ("spin_2", 2, 1e-4),
    ("luminosity_distance", None, 1e-1),
    ("inclination", None, 1e-4),
    ("phi_ref", None, 1e-4),
    ("f_ref", None, 1e-4),
]


def shifted(binary, name, component, delta):
    value = getattr(binary, name)
    value = value + delta if component is None else value.at[component].add(delta)
    return binary._replace(**{name: value})


@pytest.mark.parametrize("binary", [PRECESSING, NEARLY_ALIGNED], ids=["precessing", "nearly aligned"])
def test_grad_matches_finite_differences(function, binary):
    """The gradient of the overlap with a fixed waveform against the 4-point
    central difference, for every parameter."""
    binary = as_floats(binary)
    reference = function(shifted(shifted(binary, "mass_1", None, 0.4), "inclination", None, 0.2), TIMES)
    norm = jnp.sum(reference[0] ** 2 + reference[1] ** 2)

    @jax.jit
    def overlap(binary):
        hp, hc = function(binary, TIMES)
        return jnp.sum(hp * reference[0] + hc * reference[1]) / norm

    gradient = jax.jit(jax.grad(overlap))(binary)
    derivatives = {}
    for name, component, step in PARAMETERS:
        values = [overlap(shifted(binary, name, component, k * step)) for k in (-2, -1, 1, 2)]
        derivatives[name, component] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
    scale = max(abs(float(d)) for d in derivatives.values())
    for (name, component), fd in derivatives.items():
        value = getattr(gradient, name)
        value = float(value if component is None else value[component])
        assert np.isfinite(value)
        assert abs(value - float(fd)) <= 1e-6 * abs(float(fd)) + 1e-9 * scale, (name, component, value, float(fd))


@pytest.mark.parametrize(
    "spin_1, spin_2",
    [
        ((0.0, 0.0, 0.4), (0.0, 0.0, -0.3)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.1)),
    ],
    ids=["aligned", "non-spinning", "one zero spin"],
)
def test_grad_is_finite_at_removable_singularities(function, spin_1, spin_2):
    binary = as_floats(PRECESSING._replace(spin_1=jnp.array(spin_1), spin_2=jnp.array(spin_2)))

    def power(binary):
        hp, hc = function(binary, TIMES)
        return jnp.sum(hp * hp + hc * hc)

    gradient = jax.jit(jax.grad(power))(binary)
    for leaf in jax.tree_util.tree_leaves(gradient):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_grad_of_aligned_spin_in_plane_components(function):
    """At aligned spins the derivative with respect to an in-plane spin
    component agrees with finite differences."""
    binary = as_floats(PRECESSING._replace(spin_1=jnp.array([0.0, 0.0, 0.4]), spin_2=jnp.array([0.0, 0.0, -0.3])))
    reference = function(as_floats(PRECESSING), TIMES)

    def overlap(binary):
        hp, hc = function(binary, TIMES)
        return jnp.sum(hp * reference[0] + hc * reference[1]) / jnp.sum(reference[0] ** 2 + reference[1] ** 2)

    overlap = jax.jit(overlap)
    gradient = jax.jit(jax.grad(overlap))(binary)
    step = 1e-4
    for component in (0, 1):
        values = [overlap(shifted(binary, "spin_1", component, k * step)) for k in (-2, -1, 1, 2)]
        fd = float((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
        assert abs(float(gradient.spin_1[component]) - fd) <= 1e-6 * abs(fd) + 1e-9


def test_modes_under_jit():
    model = wf.models.get("NRSur7dq4")
    options = wf.WaveformOptions()
    modes = jax.jit(lambda b: model.modes_at(b, TIMES, options))(as_floats(PRECESSING))
    hp, hc = jax.jit(lambda b: model.polarizations_at(b, TIMES, options))(as_floats(PRECESSING))
    total = sum(
        h * wf.spin_weighted_spherical_harmonic(0.7, jnp.pi / 2 - 0.1, ell, m) for (ell, m), h in modes.items()
    )
    assert np.max(np.abs(np.asarray(total.real - hp))) < 1e-12 * np.max(np.abs(np.asarray(hp)))
    assert np.max(np.abs(np.asarray(-total.imag - hc))) < 1e-12 * np.max(np.abs(np.asarray(hp)))


def test_no_recompilation():
    """Waveforms of similar lengths share one compilation."""
    from waveformjax.models.nrsur7dq4 import _driver

    binary = dict(
        mass_1=40.0, mass_2=20.0, spin_1=(0.4, -0.3, 0.2), spin_2=(0.1, 0.5, -0.3), luminosity_distance=100.0
    )
    wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **binary)
    size = _driver._polarizations_grid._cache_size()
    lengths = set()
    for mass_1, inclination in ((41.0, 0.3), (39.5, 1.2), (40.5, 2.0)):
        result = wf.td_polarizations("NRSur7dq4", 1 / 4096, 20.0, **dict(binary, mass_1=mass_1, inclination=inclination))
        lengths.add(len(result.hplus))
    assert len(lengths) > 1
    assert _driver._polarizations_grid._cache_size() == size
