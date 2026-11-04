# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The PyTorch wrapper (``waveformx.torch``): its polarizations are the JAX
pure function's and ``torch.autograd.grad`` gives ``jax.grad``, for one
binary and a batch, in float64 and float32 (IMRPhenomXAS; one
IMRPhenomXPNR and one NRSur7dq4 case in float64), and
``torch.autograd.gradcheck`` passes for TaylorF2. Skipped without torch."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import waveformx as wf  # noqa: E402
import waveformx.torch as wt  # noqa: E402
from conftest import requires_nrsur  # noqa: E402

FREQUENCIES = np.geomspace(20.0, 900.0, 300)
TIMES = np.arange(-0.25, 0.03, 1 / 4096)
ALIGNED = [36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.4, 500.0, 0.7, 1.2, 25.0]
PRECESSING = [36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 20.0]
BATCH = [ALIGNED, [50.0, 8.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3, 300.0, 2.1, 0.3, 30.0],
         [150.0, 100.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.95, 3000.0, 0.4, 2.0, 0.0]]
DTYPES = {"float64": (torch.float64, jnp.float64, 1e-12), "float32": (torch.float32, jnp.float32, 1e-5)}


def binary_of(x, module):
    """The Binary of a row (or rows) of the parameter table, as arrays of
    ``module`` (``jnp`` or ``torch``)."""
    return wf.Binary(x[..., 0], x[..., 1], x[..., 2:5], x[..., 5:8], x[..., 8], x[..., 9], x[..., 10], x[..., 11])


def weights(shape, seed=5):
    return np.random.default_rng(seed).normal(size=(4, *shape))


def jax_loss(function, grid, w):
    def loss(binary):
        hp, hc = function(binary, grid)
        if jnp.iscomplexobj(hp):
            return jnp.sum(w[0] * hp.real + w[1] * hp.imag + w[2] * hc.real + w[3] * hc.imag)
        return jnp.sum(w[0] * hp + w[2] * hc)
    return loss


def torch_loss(hp, hc, w):
    if hp.is_complex():
        return (w[0] * hp.real + w[1] * hp.imag + w[2] * hc.real + w[3] * hc.imag).sum()
    return (w[0] * hp + w[2] * hc).sum()


def compare(approximant, rows, dtype, domain="frequency"):
    """The values and gradients of the torch function against the JAX pure
    function's, for the binaries ``rows`` (one row, or a table)."""
    torch_dtype, jax_dtype, rtol = DTYPES[dtype]
    if domain == "frequency":
        jax_function = wf.polarizations_function(approximant)
        torch_function = wt.polarizations_function(approximant)
        grid = FREQUENCIES
    else:
        jax_function = wf.td_polarizations_function(approximant)
        torch_function = wt.td_polarizations_function(approximant)
        grid = TIMES
    rows = np.asarray(rows)
    if rows.ndim == 2:
        jax_function = jax.vmap(jax_function, in_axes=(0, None))
    x = jnp.asarray(rows, jax_dtype)
    jax_grid = jnp.asarray(grid, jax_dtype)
    hp, hc = jax.jit(jax_function)(binary_of(x, jnp), jax_grid)
    w = weights(hp.shape)
    jax_grad = jax.jit(jax.grad(jax_loss(jax_function, jax_grid, jnp.asarray(w, jax_dtype))))(binary_of(x, jnp))

    fields = [torch.tensor(np.asarray(f), dtype=torch_dtype, requires_grad=True) for f in binary_of(rows, np)]
    torch_hp, torch_hc = torch_function(wf.Binary(*fields), torch.tensor(grid, dtype=torch_dtype))
    assert torch_hp.dtype == (torch.complex64 if dtype == "float32" else torch.complex128) or not torch_hp.is_complex()
    for a, b in ((torch_hp, hp), (torch_hc, hc)):
        assert a.shape == b.shape
        np.testing.assert_allclose(a.detach().numpy(), np.asarray(b), rtol=rtol, atol=rtol * float(jnp.max(jnp.abs(b))))
    grads = torch.autograd.grad(torch_loss(torch_hp, torch_hc, torch.tensor(w, dtype=torch_dtype)), fields)
    for name, g, expected in zip(wf.Binary._fields, grads, jax_grad):
        expected = np.asarray(expected)
        assert g.dtype == torch_dtype
        np.testing.assert_allclose(g.numpy(), expected, rtol=10 * rtol,
                                   atol=10 * rtol * max(np.max(np.abs(expected)), 1e-30), err_msg=name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
def test_xas(dtype, batched):
    compare("IMRPhenomXAS", BATCH if batched else ALIGNED, dtype)


def test_xpnr():
    compare("IMRPhenomXPNR", PRECESSING, "float64")


@requires_nrsur
def test_nrsur7dq4():
    compare("NRSur7dq4", [40.0, 20.0, 0.4, -0.3, 0.2, 0.1, 0.5, -0.3, 400.0, 0.7, 0.1, 20.0], "float64", "time")


def test_gradcheck():
    f = wt.polarizations_function("TaylorF2")
    frequencies = torch.linspace(20.0, 200.0, 20, dtype=torch.float64, requires_grad=True)
    fields = [torch.tensor(np.asarray(v), dtype=torch.float64, requires_grad=True) for v in binary_of(np.asarray(ALIGNED), np)]

    def function(frequencies, *fields):
        return f(wf.Binary(*fields), frequencies)

    assert torch.autograd.gradcheck(function, (frequencies, *fields))


def test_inputs():
    """Mappings, numbers, sequences of tensors and broadcast fields; the
    gradient reaches each tensor it is asked for."""
    f = wt.polarizations_function("IMRPhenomXAS")
    reference = f(binary_of(torch.tensor(ALIGNED, dtype=torch.float64), torch), torch.tensor(FREQUENCIES))
    chi = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    binary = dict(mass_1=36.0, mass_2=29, spin_1=(0.0, 0.0, chi), spin_2=[0.0, 0.0, -0.4],
                  luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=25.0)
    hp, hc = f(binary, FREQUENCIES)
    assert hp.dtype == torch.complex128
    for a, b in zip((hp, hc), reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    (grad,) = torch.autograd.grad(hp.abs().sum(), chi)
    assert torch.isfinite(grad) and grad != 0
    # a batch in mass_1 only: the other fields are broadcast
    masses = torch.tensor([36.0, 40.0], dtype=torch.float64, requires_grad=True)
    hp, hc = f({**binary, "mass_1": masses}, FREQUENCIES)
    assert hp.shape == (2, len(FREQUENCIES))
    torch.testing.assert_close(hp[0].detach(), reference[0], rtol=1e-12, atol=1e-12 * float(reference[0].abs().max()))
    g_masses, g_chi = torch.autograd.grad(hp.abs().sum(), (masses, chi))
    assert g_masses.shape == (2,) and g_chi.shape == ()
    with pytest.raises(TypeError, match="unknown"):
        f({**binary, "mass": 1.0}, FREQUENCIES)
    with pytest.raises(ValueError, match="batch"):
        f({**binary, "mass_1": masses, "mass_2": torch.ones(3)}, FREQUENCIES)
    with pytest.raises(wf.UnsupportedError):
        wt.polarizations_function("NRSur7dq4")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps():
    """Tensors on a device JAX does not have are copied through the CPU."""
    f = wt.polarizations_function("IMRPhenomXAS")
    x = torch.tensor(ALIGNED, dtype=torch.float32)
    mass_1 = x[0].clone().to("mps").requires_grad_()
    binary = binary_of(x, torch)._replace(mass_1=mass_1)
    hp, hc = f(binary, torch.tensor(FREQUENCIES, dtype=torch.float32, device="mps"))
    assert hp.device.type == "mps" and hp.dtype == torch.complex64
    cpu = f(binary._replace(mass_1=x[0]), torch.tensor(FREQUENCIES, dtype=torch.float32))
    torch.testing.assert_close(hp.cpu(), cpu[0])
    (grad,) = torch.autograd.grad((hp.real.sum() + hc.imag.sum()), mass_1)
    assert grad.device.type == "mps" and torch.isfinite(grad)


def test_import():
    """Importing waveformx does not import torch."""
    code = "import sys, waveformx; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
