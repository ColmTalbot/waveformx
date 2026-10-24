# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in MLX (``waveformjax.models.nrsur7dq4.mlx``) against the JAX
implementation: the float32 MLX waveform, its modes and its gradient agree
with the float64 JAX ones as closely as the float32 JAX evaluation does
(the bounds of test_nrsur7dq4_float32.py). Skipped without MLX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import waveformjax as wf  # noqa: E402
from conftest import requires_nrsur  # noqa: E402

pytestmark = requires_nrsur

BINARIES = {
    "precessing": wf.Binary(40.0, 20.0, (0.4, -0.3, 0.2), (0.1, 0.5, -0.3), 400.0, 0.7, 0.1, 20.0),
    "m1<m2, f_ref=0": wf.Binary(12.0, 48.0, (0.7, 0.0, 0.7), (0.0, -0.9, 0.0), 1500.0, 2.1, -0.4, 0.0),
    "heavy, q=6": wf.Binary(240.0, 40.0, (0.0, 0.5, -0.5), (0.3, 0.3, 0.3), 3000.0, 1.3, 2.0, 15.0),
    "light, aligned": wf.Binary(14.0, 7.0, (0.0, 0.0, 0.6), (0.0, 0.0, -0.4), 100.0, 0.4, 0.5, 0.0),
    "face-on": wf.Binary(30.0, 25.0, (0.2, 0.1, 0.3), (0.0, 0.2, 0.1), 500.0, 0.0, 0.0, 20.0),
}
TIMES = np.arange(-0.4, 0.03, 1 / 4096)
OPTIONS = wf.WaveformOptions()
MODEL = wf.models.get("NRSur7dq4")


@pytest.fixture(scope="module")
def nrsur():
    from waveformjax.models.nrsur7dq4 import mlx as nrsur

    return nrsur


@pytest.fixture(scope="module")
def reference():
    """The jitted float64 JAX polarizations at :data:`TIMES`."""
    function = jax.jit(lambda b: MODEL.polarizations_at(b, TIMES, OPTIONS))
    return lambda binary: [np.asarray(x) for x in function(as_float64(binary))]


@jax.jit
@jax.grad
def jax_log_likelihood_gradient(binary, data):
    """The float64 JAX gradient of a noise-free log-likelihood."""
    hp, hc = MODEL.polarizations_at(binary, TIMES, OPTIONS)
    return jnp.sum(hp * data[0] + hc * data[1]) - 0.5 * jnp.sum(hp * hp + hc * hc)


def as_float64(binary):
    return wf.Binary(*(jnp.asarray(value, jnp.float64) for value in binary))


def as_mlx(binary):
    return wf.Binary(*(mx.array(np.asarray(value, dtype=np.float32)) for value in binary))


def stacked(binaries):
    return wf.Binary(*(np.stack([np.asarray(getattr(b, f), dtype=float) for b in binaries]) for f in wf.Binary._fields))


def assert_close(a, b):
    """The bounds of test_nrsur7dq4_float32.py."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b)
    assert np.all(np.isfinite(a))
    assert np.max(np.abs(a - b)) < 1e-3 * np.max(np.abs(b))
    assert np.linalg.norm(a - b) < 3e-4 * np.linalg.norm(b)
    assert 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)) < 1e-7


def test_data(nrsur):
    """float32 data; the spline operators reproduce JAX's splines (whose
    ``lax.scan`` they replace) in float64."""
    from waveformjax.models.nrsur7dq4 import _spline, default_data
    from waveformjax.models.nrsur7dq4.mlx._data import spline_matrix, spline_system

    data = nrsur.default_data()
    assert nrsur.load() is data
    for name in ("t_ds", "dynamics", "omega_fits", "ds_to_coorb", "ei_basis", "coorb_system_inverse"):
        assert getattr(data, name).dtype == mx.float32
    assert data.monomials.dtype == mx.int32 and data.ei_time.dtype == mx.int32
    jax_data = default_data()
    y = np.random.default_rng(1).standard_normal((jax_data.n_ds, 3))
    knots = jax_data.ds_knots
    expected = _spline.evaluate(knots, y, _spline.second_derivatives(knots, y), jax_data.coorb_index, jax_data.coorb_delx)
    operator = spline_matrix(np.asarray(knots.x), np.asarray(jax_data.coorb_index), np.asarray(jax_data.coorb_delx))
    np.testing.assert_allclose(operator @ y, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))
    np.testing.assert_allclose(np.array(data.ds_to_coorb), operator.astype(np.float32))
    y = np.random.default_rng(2).standard_normal((jax_data.n_coorb, 2))
    c = np.asarray(_spline.second_derivatives(jax_data.coorb_knots, y))
    system, rhs = spline_system(np.asarray(jax_data.t_coorb))
    np.testing.assert_allclose(np.linalg.inv(system) @ (rhs @ y), c[1:-1], rtol=0, atol=1e-10 * np.max(np.abs(c)))


@pytest.mark.parametrize("name", list(BINARIES))
def test_close_to_jax(nrsur, reference, name):
    """The MLX waveform (float32) agrees with the float64 JAX one."""
    hp, hc = nrsur.polarizations_at(BINARIES[name], TIMES, OPTIONS)
    assert hp.dtype == mx.float32 and hp.shape == TIMES.shape
    for a, b in zip((hp, hc), reference(BINARIES[name])):
        assert_close(np.array(a), b)


def test_modes(nrsur):
    """The modes of a mode array, as complex64, and their polarizations."""
    options = OPTIONS.replace(mode_array=((2, 2), (2, -2), (3, 3), (4, -1)))
    binary = BINARIES["m1<m2, f_ref=0"]
    modes = nrsur.modes_at(binary, TIMES, options)
    expected = jax.jit(lambda b: MODEL.modes_at(b, TIMES, options))(as_float64(binary))
    assert set(modes) == set(expected)
    peak = max(np.max(np.abs(np.asarray(h))) for h in expected.values())
    for mode, h in expected.items():
        assert modes[mode].dtype == mx.complex64
        assert np.max(np.abs(np.array(modes[mode]) - np.asarray(h))) < 1e-3 * peak
    hp, _ = nrsur.polarizations_at(binary, TIMES, options)
    expected = jax.jit(lambda b: MODEL.polarizations_at(b, TIMES, options))(as_float64(binary))
    assert_close(np.array(hp), np.asarray(expected[0]))


def test_batch(nrsur):
    """A batch along the leading axis agrees with the binaries one by one
    (summed in another order), with shared or per-binary times; fields
    without the axis are shared."""
    binaries = list(BINARIES.values())
    hp, hc = nrsur.polarizations_at(stacked(binaries), TIMES, OPTIONS)
    assert hp.shape == (len(binaries), TIMES.size)
    for i, binary in enumerate(binaries):
        single, _ = nrsur.polarizations_at(binary, TIMES, OPTIONS)
        single = np.array(single)
        assert np.max(np.abs(np.array(hp[i]) - single)) < 1e-3 * np.max(np.abs(single))
    pair = stacked(binaries[:2])
    times = np.stack([TIMES, TIMES + 0.01])
    shifted, _ = nrsur.polarizations_at(pair, times, OPTIONS)
    single, _ = nrsur.polarizations_at(binaries[1], TIMES + 0.01, OPTIONS)
    assert np.max(np.abs(np.array(shifted[1]) - np.array(single))) < 1e-3 * np.max(np.abs(np.array(single)))
    shared = pair._replace(luminosity_distance=400.0)
    np.testing.assert_allclose(np.array(nrsur.polarizations_at(shared, TIMES, OPTIONS)[0][0]),
                               np.array(nrsur.polarizations_at(pair, TIMES, OPTIONS)[0][0]), rtol=1e-6, atol=0)
    with pytest.raises(ValueError):
        nrsur.polarizations_at(pair._replace(mass_1=np.array([40.0, 41.0, 42.0])), TIMES, OPTIONS)


def test_invalid_reference_frequency(nrsur):
    """NaN where LAL would fail to find the reference time."""
    binary = BINARIES["heavy, q=6"]._replace(f_ref=200.0)
    assert np.all(np.isnan(np.array(nrsur.polarizations_at(binary, TIMES, OPTIONS)[0])))


@pytest.mark.parametrize("name", ["precessing", "m1<m2, f_ref=0", "face-on"])
def test_gradient(nrsur, reference, name):
    """``mx.grad`` of a noise-free log-likelihood agrees with the float64
    JAX gradient (the bound of test_nrsur7dq4_float32.py); face-on tests
    the derivative of the harmonics at zero inclination."""
    binary = BINARIES[name]
    data = reference(binary._replace(mass_1=binary.mass_1 * 1.01))
    data_mx = [mx.array(x.astype(np.float32)) for x in data]
    times = mx.array(TIMES.astype(np.float32))

    def log_likelihood(b):
        hp, hc = nrsur.polarizations_at(b, times, OPTIONS)
        return mx.sum(hp * data_mx[0] + hc * data_mx[1]) - 0.5 * mx.sum(hp * hp + hc * hc)

    gradient = mx.grad(log_likelihood)(as_mlx(binary))
    single = np.concatenate([np.ravel(np.array(x)) for x in gradient]).astype(np.float64)
    double = np.concatenate([np.ravel(x) for x in jax_log_likelihood_gradient(as_float64(binary), data)])
    assert np.all(np.isfinite(single))
    scale = np.max(np.abs(double))
    assert np.all(np.abs(single - double) <= 1e-2 * np.abs(double) + 1e-3 * scale), (single, double)


def test_distance_derivative(nrsur):
    """d<h, h>/dD = -2 <h, h> / D."""
    times = mx.array(TIMES.astype(np.float32))
    binary = as_mlx(BINARIES["precessing"])

    def power(b):
        hp, hc = nrsur.polarizations_at(b, times, OPTIONS)
        return mx.sum(hp * hp + hc * hc)

    value, gradient = mx.value_and_grad(power)(binary)
    expected = -2 * value.item() / binary.luminosity_distance.item()
    assert abs(gradient.luminosity_distance.item() - expected) < 1e-4 * abs(expected)


def test_compile(nrsur):
    """``mx.compile`` gives the eager waveform and gradient (fused in another
    order)."""
    times = mx.array(TIMES.astype(np.float32))
    binary = as_mlx(BINARIES["precessing"])
    compiled = mx.compile(lambda b: nrsur.polarizations_at(b, times, OPTIONS))
    for a, b in zip(compiled(binary), nrsur.polarizations_at(binary, times, OPTIONS)):
        a, b = np.array(a), np.array(b)
        assert np.max(np.abs(a - b)) < 1e-5 * np.max(np.abs(b))

    def power(b):
        hp, hc = nrsur.polarizations_at(b, times, OPTIONS)
        return mx.sum(hp * hp + hc * hc)

    eager = np.concatenate([np.ravel(np.array(x)) for x in mx.grad(power)(binary)])
    fused = np.concatenate([np.ravel(np.array(x)) for x in mx.compile(mx.grad(power))(binary)])
    assert np.max(np.abs(fused - eager)) < 1e-4 * np.max(np.abs(eager))


#: ``(q, chi_a, chi_b, t_ref)``: precessing; aligned from the start of the
#: surrogate (the half nodes); without spins and with the largest spins,
#: starting forwards from nodes 1 to 3; late, starting backwards.
DYNAMICS = np.array([
    [2.0, 0.4, -0.3, 0.2, 0.1, 0.5, -0.3, -2500.0],
    [1.2, 0.0, 0.0, 0.6, 0.0, 0.0, -0.4, -4299.99999999],
    [3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4280.0],
    [1.0, 0.7, 0.0, 0.7, 0.0, -0.9, 0.0, -4250.0],
    [3.9, 0.0, 0.5, -0.5, 0.3, 0.3, 0.3, -600.0],
], dtype=np.float32)


def test_dynamics_kernel(nrsur):
    """The Metal kernel integrates the dynamics as the MLX operations do,
    close to JAX in float64, and its Jacobian with respect to ``(q, chi_a,
    chi_b, t_ref)`` agrees with JAX's float64 forward-mode derivatives;
    ``mx.grad`` contracts it."""
    from waveformjax.models.nrsur7dq4 import _dynamics, default_data
    from waveformjax.models.nrsur7dq4.mlx import _dynamics as mlx_dynamics
    from waveformjax.models.nrsur7dq4.mlx import _kernel

    data = nrsur.default_data()
    jax_data = default_data()

    def jax_states(p):
        q, chi_a, chi_b, t_ref = p[0], p[1:4], p[4:7], p[7]
        norm_a, norm_b = _dynamics.safe_norm(chi_a), _dynamics.safe_norm(chi_b)
        i0, y0 = _dynamics._initialize_at_node(jax_data, q, t_ref, _dynamics.reference_state(chi_a, chi_b), norm_a, norm_b)
        i_start, states, derivs = _dynamics._rk4_start(jax_data, q, i0, y0, norm_a, norm_b)
        return _dynamics._ab4(jax_data, q, i_start, states, derivs, norm_a, norm_b)

    params = mx.array(DYNAMICS)
    states, jacobian = (np.array(x) for x in _kernel.states_and_jacobian(data, params))
    value, _ = _kernel.states_and_jacobian(data, params, 1)
    np.testing.assert_array_equal(np.array(value), states)
    p = params
    ops = np.array(mlx_dynamics.states_from_reference(data, p[:, 0], p[:, 1:4], p[:, 4:7], p[:, 7]))
    wide = jnp.asarray(DYNAMICS.astype(np.float64))
    reference = np.asarray(jax.jit(jax.vmap(jax_states))(wide))
    jac_reference = np.asarray(jax.jit(jax.vmap(jax.jacfwd(jax_states)))(wide))
    for b in range(len(DYNAMICS)):
        scale = np.max(np.abs(reference[b]), axis=0)
        for test in (states[b], ops[b]):
            assert np.all(np.abs(test - reference[b]) <= 2e-4 * scale)
        column = np.max(np.abs(jac_reference[b]), axis=0)
        error = np.max(np.abs(jacobian[b] - jac_reference[b]), axis=0)
        assert np.all(np.isfinite(jacobian[b]))
        assert np.all(error <= 3e-3 * column + 1e-6 * column.max())
    weights = mx.array(np.random.default_rng(0).standard_normal(states.shape).astype(np.float32))
    gradient = mx.grad(lambda x: mx.sum(_kernel.states_function(data)(x) * weights))(params)
    expected = np.einsum("bisk,bis->bk", jacobian, np.array(weights))
    assert np.max(np.abs(np.array(gradient) - expected)) < 1e-5 * np.max(np.abs(expected))


def test_scaled_strain(nrsur):
    """The strain times 1e21 unless scaled_strain=False."""
    binary = BINARIES["precessing"]
    physical = OPTIONS.replace(scaled_strain=False)
    for method in (nrsur.polarizations_at, nrsur.modes_at):
        scaled = method(binary, TIMES, OPTIONS)
        strain = method(binary, TIMES, physical)
        for key in (range(2) if isinstance(scaled, tuple) else scaled):
            a, b = np.array(scaled[key]), np.array(strain[key])
            assert np.max(np.abs(a - 1e21 * b)) < 1e-6 * np.max(np.abs(a))


def test_wigner_kernel(nrsur):
    """The Metal kernels of the rotation to the inertial frame agree with the
    MLX operations, values and VJP, including quaternions without precession
    (``rb = 0``) and with ``ra = 0``; their generators are JAX's
    ``dD/dq`` at the identity."""
    from waveformjax.models.nrsur7dq4 import _frames as jax_frames
    from waveformjax.models.nrsur7dq4.mlx import _frames, _wigner

    rng = np.random.default_rng(0)
    shape = (3, len(_wigner.MODES), 64)
    re, im, c_re, c_im = (mx.array(rng.standard_normal(shape).astype(np.float32)) for _ in range(4))
    q = rng.standard_normal((3, 64, 4))
    q[0, :8] = (1.0, 0.0, 0.0, 0.0)
    q[1, :8] = (0.0, 1.0, 0.0, 0.0)
    q = mx.array((q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32))
    out = _wigner.rotate(re, im, q)
    expected = _frames.coprecessing_to_inertial(re, im, q)
    for a, b in zip(out, expected):
        assert float(mx.max(mx.abs(a - b))) < 1e-6 * float(mx.max(mx.abs(b)))
    _, cotangents = mx.vjp(lambda *x: tuple(_wigner.rotate(*x)), [re, im, q], [c_re, c_im])
    _, expected = mx.vjp(lambda *x: _frames.coprecessing_to_inertial(*x), [re, im, q], [c_re, c_im])
    for a, b in zip(cotangents, expected):
        assert float(mx.max(mx.abs(a - b))) < 1e-6 * float(mx.max(mx.abs(b)))

    n = len(_wigner.MODES)

    def matrix(quaternion):
        rotated = jax_frames.coprecessing_to_inertial(jnp.eye(n), jnp.zeros((n, n)), jnp.tile(quaternion[:, None], (1, n)))
        return jnp.stack(rotated)

    jacobian = np.asarray(jax.jacfwd(matrix)(jnp.array([1.0, 0.0, 0.0, 0.0])))
    np.testing.assert_allclose(_wigner.generators(), np.moveaxis(jacobian[0] + 1j * jacobian[1], -1, 0), atol=1e-12)
