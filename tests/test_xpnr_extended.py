# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR without LAL's fallbacks (``version_fallback=False``) and
with the extended SpinTaylor angles (``extended_spintaylor=True``).

* Without fallbacks, the model is LAL's where LAL uses the SpinTaylor
  angles and NaN where it would switch to the MSA or NNLO angles.
* The extended angles are finite where LAL leaves the SpinTaylor angles (a
  reference frequency beyond the PN energy test, in-plane spins below
  ``1e-6``, aligned, single and vanishing spins), close to LAL's where LAL
  uses them, and continuous where LAL's waveform jumps: at the end of the
  fine SpinTaylor run, across ``|chi2| = 1e-3``, ``chi_perp = 1e-3`` and
  ``q = 12``, and across the reference frequency at which LAL gives up the
  SpinTaylor angles.
* Their gradients are finite, also for aligned and vanishing spins, and
  agree with finite differences.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from waveformjax import Binary, WaveformOptions
from waveformjax.exceptions import PrecessionVersionUnavailableError
from waveformjax.models.xpnr import MODEL
from waveformjax.models.xpnr.api import _prescription

LAL = WaveformOptions()
STRICT = WaveformOptions(version_fallback=False)
EXTENDED = WaveformOptions(extended_spintaylor=True)
MTSUN_SI = 4.925490947641267e-06


def binary(mass_1=36.0, mass_2=29.0, spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1), f_ref=20.0, inclination=0.7):
    return Binary(mass_1=jnp.float64(mass_1), mass_2=jnp.float64(mass_2), spin_1=jnp.asarray(spin_1, jnp.float64),
                  spin_2=jnp.asarray(spin_2, jnp.float64), luminosity_distance=jnp.float64(500.0),
                  inclination=jnp.float64(inclination), phi_ref=jnp.float64(1.2), f_ref=jnp.float64(f_ref))


def frequencies(mass, f_min=20.0, n=1024):
    return jnp.linspace(f_min, min(0.3 / (mass * MTSUN_SI), 1024.0), n)


@functools.lru_cache(maxsize=None)
def compiled(options):
    return jax.jit(functools.partial(MODEL.polarizations, options=options))


def relative(a, b):
    """``|a - b| / |b|`` of the polarizations (both)."""
    a = np.concatenate([np.asarray(x) for x in a])
    b = np.concatenate([np.asarray(x) for x in b])
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def finite(h):
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in h)


# a binary whose reference frequency lies beyond the PN energy test (LAL
# falls back to the MSA angles)
BEYOND = dict(mass_1=180.0, mass_2=120.0, f_ref=60.0)


def test_strict_is_lal_where_lal_uses_spintaylor():
    b = binary()
    f = frequencies(65.0)
    assert relative(compiled(STRICT)(b, f), compiled(LAL)(b, f)) < 1e-12


@pytest.mark.parametrize("case", ["beyond", "small_in_plane"])
def test_strict_is_nan_where_lal_falls_back(case):
    if case == "beyond":
        b, f = binary(**BEYOND), frequencies(300.0, 15.0)
    else:
        b, f = binary(spin_1=(5e-7, 0.0, 0.3), spin_2=(0.0, 0.0, -0.1)), frequencies(65.0)
    assert finite(compiled(LAL)(b, f))
    assert not finite(compiled(STRICT)(b, f))


def test_validate_strict_and_extended():
    """The host checks: without fallbacks LAL's switch raises; the extended
    angles do not."""
    b, f = binary(**BEYOND), np.asarray(frequencies(300.0, 15.0))
    with pytest.raises(PrecessionVersionUnavailableError):
        MODEL.validate(b, f, STRICT)
    MODEL.validate(b, f, EXTENDED)


@pytest.mark.parametrize("spins", [
    ((0.4, 0.2, 0.3), (-0.2, 0.3, -0.1)),
    ((0.1, -0.3, -0.5), (0.3, 0.1, 0.4)),
    ((0.7, 0.0, 0.1), (0.05, 0.02, 0.0)),
])
def test_extended_close_to_lal(spins):
    """Where LAL uses the SpinTaylor angles (two spins, away from LAL's
    switches), the extended angles differ from LAL's by much less than a
    mismatch of 1e-3."""
    b = binary(spin_1=spins[0], spin_2=spins[1])
    f = frequencies(65.0)
    assert relative(compiled(EXTENDED)(b, f), compiled(LAL)(b, f)) < 1e-2


@pytest.mark.parametrize("spins", [
    ((0.0, 0.0, 0.3), (0.0, 0.0, -0.1)),
    ((5e-7, 0.0, 0.3), (0.0, 0.0, -0.1)),
    ((0.5, 0.2, 0.3), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.4, 0.1, 0.2)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
])
def test_extended_finite_for_special_spins(spins):
    b = binary(spin_1=spins[0], spin_2=spins[1])
    assert finite(compiled(EXTENDED)(b, frequencies(65.0)))


def test_extended_finite_beyond_energy_test():
    assert finite(compiled(EXTENDED)(binary(**BEYOND), frequencies(300.0, 15.0)))


def _jump(options, make, x):
    """The relative change of the polarizations across ``x`` (adjacent
    floats on either side)."""
    lo, hi = np.nextafter(x, -np.inf), np.nextafter(x, np.inf)
    f = make(lo)[1]
    return relative(compiled(options)(make(hi)[0], f), compiled(options)(make(lo)[0], f))


def _bisect(flag, lo, hi, iterations=60):
    """The point between ``lo`` and ``hi`` where the boolean ``flag`` changes."""
    f_lo = flag(lo)
    assert flag(hi) != f_lo
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if flag(mid) == f_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_continuous_across_reference_beyond_energy_test():
    """Across the reference frequency at which LAL's SpinTaylor angles stop
    (LAL switches to the MSA angles there)."""
    f = frequencies(300.0, 15.0)

    def make(f_ref):
        return binary(**dict(BEYOND, f_ref=f_ref)), f

    def failed(f_ref):
        flags = _prescription(make(f_ref)[0], f[0], f[-1], ((2, 1), (2, 2), (3, 2), (3, 3), (4, 4)), True)
        return bool(flags["st_failed"])

    x = _bisect(failed, 15.0, 60.0)
    assert _jump(EXTENDED, make, x) < 1e-6


@pytest.mark.parametrize("which", ["q12", "two_spin", "remap"])
def test_continuous_across_lal_switches(which):
    """LAL's waveform jumps across ``q = 12`` (the edge of the PNR
    transition region), ``|chi2| = 1e-3`` (the two-spin construction) and
    ``chi_perp = 1e-3`` (the remapped J frame); the extended one does not."""
    if which == "q12":
        def make(q):
            return binary(mass_1=120.0 * q / (1 + q), mass_2=120.0 / (1 + q), spin_1=(0.5, 0.0, 0.3),
                          spin_2=(0.0, 0.2, 0.1)), frequencies(120.0)
        x = 12.0
    elif which == "two_spin":
        def make(a):
            return binary(mass_1=50.0, mass_2=6.0, spin_1=(0.5, 0.2, 0.3), spin_2=(a, 0.0, 0.0)), frequencies(56.0)
        x = 1e-3
    else:
        def make(a):
            return binary(spin_1=(0.6 * a, 0.8 * a, 0.3), spin_2=(0.0, 0.0, 0.2)), frequencies(65.0)
        x = 1e-3
    assert _jump(LAL, make, x) > 1e-5
    assert _jump(EXTENDED, make, x) < 1e-6


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

OPTIONS_22 = WaveformOptions(mode_array=((2, 2), (2, -2)), extended_spintaylor=True)
GRID = jnp.asarray(np.geomspace(20.0, 400.0, 96))
_WEIGHTS = np.random.default_rng(3).normal(size=(2, len(GRID), 2))
NAMES = ["mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z", "luminosity_distance",
         "inclination", "phi_ref", "f_ref"]
POINTS = {
    "generic": [36.0, 29.0, 0.4, 0.2, 0.3, -0.2, 0.3, -0.1, 500.0, 0.7, 1.2, 25.0],
    "aligned": [36.0, 29.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.1, 500.0, 0.7, 1.2, 25.0],
    "non_spinning": [36.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.7, 1.2, 25.0],
}
RELATIVE = {"mass_1", "mass_2", "luminosity_distance", "f_ref"}


def _loss(x):
    b = Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8], inclination=x[9],
               phi_ref=x[10], f_ref=x[11])
    hp, hc = MODEL.polarizations(b, GRID, OPTIONS_22)
    w = jnp.asarray(_WEIGHTS)
    scale = 1e21
    return jnp.sum(w[0, :, 0] * hp.real * scale + w[0, :, 1] * hp.imag * scale
                   + w[1, :, 0] * hc.real * scale + w[1, :, 1] * hc.imag * scale)


@functools.lru_cache(maxsize=None)
def _compiled_loss(name):
    return {"loss": jax.jit(_loss), "grad": jax.jit(jax.grad(_loss))}[name]


@pytest.mark.parametrize("which", list(POINTS))
def test_gradient_finite_and_matches_finite_differences(which):
    """The gradients are finite, also at aligned and vanishing spins; at the
    generic binary they agree with central differences (the in-plane spins
    of the aligned binaries, where the waveform is not differentiable in
    them, only finite)."""
    x = jnp.asarray(POINTS[which])
    gradient = np.asarray(_compiled_loss("grad")(x))
    assert np.all(np.isfinite(gradient))
    if which != "generic":
        return
    f = _compiled_loss("loss")
    scale = float(np.max(np.abs(gradient)))
    for i, name in enumerate(NAMES):
        step = 1e-6 * (abs(float(x[i])) if name in RELATIVE else 1.0)
        numeric = (float(f(x.at[i].add(step))) - float(f(x.at[i].add(-step)))) / (2 * step)
        error = abs(gradient[i] - numeric) / max(abs(numeric), 1e-3 * scale)
        print(f"{name}: gradient {gradient[i]:.10e} finite difference {numeric:.10e} relative {error:.1e}")
        assert error <= 2e-5, (name, gradient[i], numeric)


def test_vmap_matches_single_calls():
    xs = [binary(), binary(mass_1=45.0), binary(spin_2=(0.0, 0.0, 0.0))]
    batch = jax.jit(jax.vmap(lambda b: MODEL.polarizations(b, GRID, OPTIONS_22)))(
        jax.tree_util.tree_map(lambda *v: jnp.stack(v), *xs))
    single = jax.jit(lambda b: MODEL.polarizations(b, GRID, OPTIONS_22))
    for i, b in enumerate(xs):
        for a, s in zip(batch, single(b)):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(s), rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(s))))
