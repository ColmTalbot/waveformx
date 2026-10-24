# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR (``WaveformOptions(lean=True)``).

* It is the extended model with the direct angles and the parity fix,
  without LAL's numerics: the two agree to the integration error of the
  precession equations (mismatches far below 1e-8), also where LAL leaves
  the SpinTaylor angles.
* It is close to LAL (with the parity fix) where LAL uses the SpinTaylor
  angles.
* It is finite, with finite derivatives, for aligned, single and vanishing
  spins and beyond the PN energy test, in float64 and float32; float32 is
  close to float64.
* ``jax.jit``, ``jax.vmap``, ``jax.jvp`` and ``jax.grad`` work, and the
  derivatives agree with finite differences.
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import waveformjax as wf
from waveformjax import Binary, WaveformOptions
from waveformjax.models.xpnr import MODEL, model

LEAN = WaveformOptions(lean=True)
EXTENDED = WaveformOptions(extended_spintaylor=True)
MTSUN_SI = 4.925490947641267e-06
CASE = json.loads((Path(__file__).parent / "data" / "imrphenomxpnr_parityfix.json").read_text())["cases"][0]

BINARIES = {
    "generic": dict(),
    "light": dict(mass_1=9.0, mass_2=6.0, spin_1=(0.3, 0.3, 0.3), spin_2=(-0.3, 0.2, 0.0)),
    "heavy": dict(mass_1=130.0, mass_2=60.0, f_ref=24.0),
    "unequal": dict(mass_1=90.0, mass_2=5.0, spin_1=(0.6, 0.3, -0.2), spin_2=(0.1, 0.2, 0.3)),
    # the reference frequency beyond the PN energy test
    "beyond": dict(mass_1=180.0, mass_2=120.0, f_ref=60.0),
}
SPECIAL = {
    "aligned": dict(spin_1=(0.0, 0.0, 0.3), spin_2=(0.0, 0.0, -0.1)),
    "small_in_plane": dict(spin_1=(5e-7, 0.0, 0.3), spin_2=(0.0, 0.0, -0.1)),
    "single": dict(spin_1=(0.5, 0.2, 0.3), spin_2=(0.0, 0.0, 0.0)),
    "non-spinning": dict(spin_1=(0.0, 0.0, 0.0), spin_2=(0.0, 0.0, 0.0)),
}
# float32 NaN tangents fixed: the pseudo-PN inspiral amplitude of the (2,1)
# at q = 18 (a denominator ~1e-19 squared), and the ringdown beta below
# -0.3 rad (an overflowing power in the arctan window)
FLOAT32_REGRESSIONS = {
    "q18": BINARIES["unequal"],
    "negative_beta_rd": dict(mass_1=137.657, mass_2=7.195, spin_1=(-0.31, -0.824, 0.303),
                             spin_2=(0.113, 0.336, 0.036)),
}


def binary(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), f_ref=20.0, dtype=jnp.float64):
    return Binary(mass_1=jnp.asarray(mass_1, dtype), mass_2=jnp.asarray(mass_2, dtype),
                  spin_1=jnp.asarray(spin_1, dtype), spin_2=jnp.asarray(spin_2, dtype),
                  luminosity_distance=jnp.asarray(500.0, dtype), inclination=jnp.asarray(0.7, dtype),
                  phi_ref=jnp.asarray(1.2, dtype), f_ref=jnp.asarray(f_ref, dtype))


def frequencies_for(b, n=1024, dtype=jnp.float64):
    mass = float(b.mass_1 + b.mass_2)
    f_min = min(20.0, float(b.f_ref))
    return jnp.linspace(f_min, min(0.3 / (mass * MTSUN_SI), 1024.0), n, dtype=dtype)


@functools.lru_cache(maxsize=None)
def compiled(options):
    return jax.jit(lambda b, f: model._polarizations(b, f, options))


@functools.lru_cache(maxsize=None)
def with_derivative(options):
    """The polarizations and their forward derivative along ``chi_1x`` and
    ``m_1`` together (one compilation of the model's own function)."""
    def along(b, f):
        def h(b):
            return model._polarizations(b, f, options)

        tangent = jax.tree_util.tree_map(jnp.zeros_like, b)
        tangent = tangent._replace(spin_1=tangent.spin_1.at[0].set(1.0), mass_1=jnp.ones_like(b.mass_1))
        return jax.jvp(h, (b,), (tangent,))

    return jax.jit(along)


def stacked(h):
    return np.concatenate([np.asarray(x, complex) for x in h])


def mismatch(a, b):
    a, b = stacked(a), stacked(b)
    return 1.0 - np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def finite(h):
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in h)


@pytest.mark.parametrize("which", list(BINARIES))
def test_lean_is_the_extended_model(which):
    b = binary(**BINARIES[which])
    f = frequencies_for(b)
    lean, extended = compiled(LEAN)(b, f), compiled(EXTENDED)(b, f)
    assert finite(lean) and finite(extended)
    assert mismatch(lean, extended) < 1e-8


def test_lean_close_to_lal():
    """LAL with the parity fix (stored), where it uses the SpinTaylor
    angles: within the extended model's tolerance."""
    b = Binary(mass_1=jnp.float64(CASE["mass_1"]), mass_2=jnp.float64(CASE["mass_2"]),
               spin_1=jnp.asarray(CASE["spin_1"], float), spin_2=jnp.asarray(CASE["spin_2"], float),
               luminosity_distance=jnp.float64(CASE["luminosity_distance"]),
               inclination=jnp.float64(CASE["inclination"]), phi_ref=jnp.float64(CASE["phi_ref"]),
               f_ref=jnp.float64(CASE["f_ref"]))
    hp, hc = compiled(LEAN.replace(scaled_strain=False))(b, jnp.asarray(CASE["frequencies"]))
    reference = [np.asarray(CASE[f"{name}_real"]) + 1j * np.asarray(CASE[f"{name}_imag"])
                 for name in ("hplus", "hcross")]
    assert mismatch((hp, hc), reference) < 1e-5


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("which", list(SPECIAL) + ["beyond"] + list(FLOAT32_REGRESSIONS))
def test_lean_and_its_derivatives_finite(which, dtype):
    spins = SPECIAL.get(which, BINARIES.get(which, FLOAT32_REGRESSIONS.get(which)))
    b = binary(**spins, dtype=dtype)
    f = frequencies_for(b, 256, dtype)
    h, dh = with_derivative(LEAN)(b, f)
    assert finite(h) and finite(dh)
    for x in h + dh:
        assert np.asarray(x).dtype == (np.complex64 if dtype == jnp.float32 else np.complex128)


@pytest.mark.parametrize("which", ["generic", "light", "beyond"])
def test_lean_float32_close_to_float64(which):
    b = binary(**BINARIES[which])
    f = frequencies_for(b)
    wide = compiled(LEAN)(b, f)
    narrow = compiled(LEAN)(binary(**BINARIES[which], dtype=jnp.float32), f.astype(jnp.float32))
    assert mismatch(narrow, wide) < 1e-6


def test_lean_public_api():
    """The public functions take ``lean=True``; the J-frame modes too."""
    b = binary()
    f = np.asarray(frequencies_for(b, 256))
    hp, hc = wf.polarizations("IMRPhenomXPNR", f, **b._asdict(), lean=True)
    ours = model._polarizations(b, jnp.asarray(f), LEAN)
    assert mismatch((hp, hc), ours) < 1e-12
    modes = MODEL.modes(b, f, LEAN)
    assert all(np.all(np.isfinite(np.asarray(h))) for h in modes.modes.values())
    extended = MODEL.modes(b, f, EXTENDED)
    assert mismatch(list(modes.modes.values()), list(extended.modes.values())) < 1e-8


def test_lean_vmap_matches_single_calls():
    bs = [binary(), binary(**BINARIES["heavy"]), binary(**SPECIAL["aligned"])]
    f = jnp.linspace(20.0, 512.0, 256)
    batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *bs)
    hp, hc = jax.jit(jax.vmap(lambda b: model.polarizations(b, f, LEAN)))(batch)
    single = compiled(LEAN)
    for i, b in enumerate(bs):
        ref = single(b, f)
        assert mismatch((hp[i], hc[i]), ref) < 1e-12


def test_lean_gradient_matches_finite_differences():
    f = jnp.linspace(20.0, 256.0, 256)
    x0 = np.array([36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 20.0])

    def of(x):
        return Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                      inclination=x[9], phi_ref=x[10], f_ref=x[11])

    weights = jnp.exp(-f / 100.0)

    def loss(x):
        hp, hc = model.polarizations(of(x), f, LEAN)
        return jnp.sum(weights * (jnp.real(hp) + jnp.imag(hc)))

    value = jax.jit(loss)
    gradient = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(x0)))
    assert np.all(np.isfinite(gradient))
    for k in (0, 2, 4, 6, 9):
        step = 1e-6 * max(abs(x0[k]), 1.0)
        e = np.zeros(12)
        e[k] = step
        fd = (float(value(jnp.asarray(x0 + e))) - float(value(jnp.asarray(x0 - e)))) / (2 * step)
        assert abs(gradient[k] - fd) <= 1e-4 * abs(fd) + 1e-6 * np.max(np.abs(gradient)), (k, gradient[k], fd)
