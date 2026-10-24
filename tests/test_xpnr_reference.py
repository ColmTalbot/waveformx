# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The mass-scaled reference frequency of the lean IMRPhenomXPNR
(``WaveformOptions(lean=True, reference_mf=...)``) and the converters
between the reference conventions.

* The options are checked; a non-zero ``f_ref`` is rejected.
* The waveform is finite for every mode array, including those whose
  ``m'`` is 1 only (NaN in LAL's convention with ``f_ref = f_min``), over
  mass ratios up to 30 and spins up to 0.99, with finite derivatives in
  float32 and float64.
* The converters: round trips return the binary; aligned spins are
  converted exactly (the waveforms of the two conventions are then equal);
  the in-plane spins stay zero with finite derivatives; jit, vmap and grad
  in float32 and float64.
* Equivalence: as the in-plane spins vanish, the waveforms of the two
  conventions agree. Against LAL (with the parity fix, where available):
  the lean model with the mass-scaled reference agrees with LAL given the
  converted binary at the physical ``f_ref = reference_mf / M``. (XPNR is
  not invariant under a change of its reference: LAL at two references
  differs by as much as 1e-2 for strongly precessing binaries.)
* The waveform and its derivative are continuous where the reference
  crosses the lower end of the band.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import waveformjax as wf
from waveformjax.exceptions import InvalidParameterError
from waveformjax.models.xpnr import lean

MTSUN_SI = 4.925490947641267e-06
REF = 0.01
F = np.arange(20.0, 1024.0, 0.125)
GENERIC = wf.Binary(36.0, 29.0, (0.6, 0.3, -0.2), (0.1, 0.2, 0.3), 500.0, 0.7, 1.2, 0.0)


def as_dtype(binary, dtype):
    return wf.Binary(*(jnp.asarray(value, dtype) for value in binary))


def overlap(h1, h2, maximise=True):
    """``1 - O`` of ``(h_+, h_x)`` with white noise, maximised over time and
    phase (or not)."""
    n = 1 << 18
    z = np.zeros(n, complex)
    k = len(h1[0])
    z[:k] = np.conj(h1[0]) * h2[0] + np.conj(h1[1]) * h2[1]
    norm = np.sqrt(sum(np.real(np.vdot(x, x)) for x in h1) * sum(np.real(np.vdot(x, x)) for x in h2))
    value = np.max(np.abs(np.fft.ifft(z) * n)) if maximise else np.real(np.sum(z))
    return 1.0 - value / norm


def rotated(h, psi):
    """The polarizations in the basis of the other reference (see
    ``to_mass_scaled_reference``)."""
    c, s = np.cos(2.0 * psi), np.sin(2.0 * psi)
    return [h[0] * c - h[1] * s, h[1] * c + h[0] * s]


_SCALED = {}


def scaled(binary, frequencies, modes=None):
    key = modes
    if key not in _SCALED:
        _SCALED[key] = jax.jit(lambda b, f: lean.polarizations(b, f, mode_array=modes, reference_mf=REF))
    return _SCALED[key](binary, frequencies)


_physical = jax.jit(lambda b, f: lean.polarizations(b, f))
_to_scaled = jax.jit(lambda b: wf.to_mass_scaled_reference(b, return_angle=True))
_to_physical = jax.jit(lambda b, f: wf.to_physical_reference(b, f, return_angle=True))


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_options():
    assert wf.WaveformOptions(lean=True, reference_mf=REF).reference_mf == REF
    assert wf.REFERENCE_MF == REF
    with pytest.raises(InvalidParameterError):
        wf.WaveformOptions(reference_mf=REF)
    for bad in (0.0, -0.01, 0.02):
        with pytest.raises(InvalidParameterError):
            wf.WaveformOptions(lean=True, reference_mf=bad)
    with pytest.raises(TypeError):
        wf.WaveformOptions(lean=True, reference_mf=True)


def test_public_api():
    """The public function takes the option, and rejects a non-zero f_ref
    rather than ignoring it; the pure function gives NaN there."""
    keywords = dict(mass_1=36.0, mass_2=29.0, spin_1=(0.6, 0.3, -0.2), spin_2=(0.1, 0.2, 0.3), inclination=0.7,
                    phi_ref=1.2)
    f = np.linspace(20.0, 512.0, 64)
    hp, hc = wf.polarizations("IMRPhenomXPNR", f, lean=True, reference_mf=REF, **keywords)
    assert np.all(np.isfinite(np.asarray(hp)))
    with pytest.raises(InvalidParameterError, match="f_ref must be zero"):
        wf.polarizations("IMRPhenomXPNR", f, lean=True, reference_mf=REF, f_ref=20.0, **keywords)
    hp, _ = scaled(as_dtype(GENERIC._replace(f_ref=20.0), jnp.float64), jnp.asarray(f))
    assert np.all(np.isnan(np.asarray(hp)))


# ---------------------------------------------------------------------------
# finite everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modes", [((2, 1), (2, -1)), ((3, 2), (3, -2)), ((3, 3), (3, -3)), ((4, 4), (4, -4)), None])
def test_finite(modes):
    """Finite for every mode array (m' = 1 alone included) over a random
    set of binaries: mass ratios to 30, total masses 3-400, spins to 0.99,
    the band starting at 10-30 Hz; and for the binary of 90 + 5 solar
    masses with (2,1) alone, NaN in LAL's convention at f_ref = f_min."""
    rng = np.random.default_rng(11)
    n = 48
    q = np.exp(rng.uniform(0, np.log(30), n))
    mass = np.exp(rng.uniform(np.log(3), np.log(400), n))

    def spins():
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None] * rng.uniform(0, 0.99, n)[:, None]

    batch = wf.Binary(jnp.asarray(mass * q / (1 + q)), jnp.asarray(mass / (1 + q)), jnp.asarray(spins()),
                      jnp.asarray(spins()), jnp.full(n, 500.0), jnp.asarray(rng.uniform(0, np.pi, n)),
                      jnp.asarray(rng.uniform(0, 2 * np.pi, n)), jnp.zeros(n))
    batch = batch._replace(mass_1=batch.mass_1.at[0].set(90.0), mass_2=batch.mass_2.at[0].set(5.0),
                           spin_1=batch.spin_1.at[0].set(jnp.asarray([0.6, 0.3, -0.2])),
                           spin_2=batch.spin_2.at[0].set(jnp.asarray([0.1, 0.2, 0.3])))
    f_min = rng.uniform(10, 30, n)
    f_min[0] = 20.0
    frequencies = jnp.asarray(np.stack([np.geomspace(f, 1024.0, 1000) for f in f_min]))
    hp, hc = jax.jit(jax.vmap(lambda b, f: lean.polarizations(b, f, mode_array=modes, reference_mf=REF)))(
        batch, frequencies)
    finite = np.all(np.isfinite(np.asarray(hp)), axis=1) & np.all(np.isfinite(np.asarray(hc)), axis=1)
    assert np.all(finite), np.where(~finite)[0]


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("modes", [((2, 1), (2, -1)), None])
def test_derivatives_finite(dtype, modes):
    """Finite gradients of a log-likelihood, in float32 and float64, for a
    light, a heavy and an aligned-spin binary; and float32 close to
    float64."""
    f = jnp.asarray(np.geomspace(20.0, 1024.0, 256), dtype)
    binaries = [GENERIC, GENERIC._replace(mass_1=180.0, mass_2=120.0),
                GENERIC._replace(spin_1=(0.0, 0.0, 0.3), spin_2=(0.0, 0.0, -0.2))]
    for binary in binaries:
        data = lean.polarizations(as_dtype(binary._replace(mass_1=binary.mass_1 * 1.01), jnp.float64),
                                  jnp.asarray(f, jnp.float64), mode_array=modes, reference_mf=REF)
        data = [jnp.asarray(x, jnp.complex64 if dtype == jnp.float32 else jnp.complex128) for x in data]

        def log_likelihood(b):
            hp, hc = lean.polarizations(b, f, mode_array=modes, reference_mf=REF)
            return jnp.real(jnp.vdot(data[0], hp) + jnp.vdot(data[1], hc)) - 0.5 * jnp.sum(
                jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

        grad = jax.jit(jax.grad(log_likelihood))(as_dtype(binary, dtype))
        values = np.concatenate([np.ravel(np.asarray(x, float)) for x in grad])
        assert np.all(np.isfinite(values))
        # no dependence on f_ref, which the option ignores
        assert float(grad.f_ref) == 0.0
        if dtype == jnp.float32:
            hp32, _ = lean.polarizations(as_dtype(binary, dtype), f, mode_array=modes, reference_mf=REF)
            hp64, _ = lean.polarizations(as_dtype(binary, jnp.float64), jnp.asarray(f, jnp.float64),
                                         mode_array=modes, reference_mf=REF)
            a, b = np.asarray(hp32, np.complex128), np.asarray(hp64)
            assert 1 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)) < 1e-5


# ---------------------------------------------------------------------------
# the converters
# ---------------------------------------------------------------------------

CONVERTED = [
    GENERIC._replace(f_ref=20.0),
    wf.Binary(10.0, 5.0, (0.4, -0.5, 0.3), (-0.3, 0.2, 0.1), 500.0, 2.2, 4.0, 20.0),
    wf.Binary(90.0, 5.0, (0.6, 0.3, -0.2), (0.1, 0.2, 0.3), 500.0, 0.7, 1.2, 20.0),
    wf.Binary(6.0, 1.4, (0.0, 0.0, 0.5), (0.0, 0.0, -0.3), 500.0, 0.4, 0.3, 25.0),
    wf.Binary(20.0, 18.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 500.0, 1.0, 0.3, 20.0),
]


@pytest.mark.parametrize("index", range(len(CONVERTED)))
def test_round_trip(index):
    """Physical -> mass-scaled -> physical returns the binary (to rounding:
    the integration is reversed step by step), and the angle of the
    polarizations changes sign."""
    binary = as_dtype(CONVERTED[index], jnp.float64)
    there, psi = _to_scaled(binary)
    assert float(there.f_ref) == 0.0
    back, psi_back = _to_physical(there, binary.f_ref)
    for name in wf.Binary._fields:
        np.testing.assert_allclose(np.asarray(getattr(back, name)), np.asarray(getattr(binary, name)), atol=1e-10,
                                   err_msg=name)
    np.testing.assert_allclose(float(psi_back), -float(psi), atol=1e-10)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_aligned_exact(dtype):
    """Aligned spins: the spins and the inclination are unchanged, the
    in-plane spins stay exactly zero with finite derivatives, and the
    waveforms of the two conventions are equal (``phi_ref`` takes the
    orbital phase between the references)."""
    binary = as_dtype(CONVERTED[3], dtype)
    there = wf.to_mass_scaled_reference(binary)
    assert np.all(np.asarray(there.spin_1)[:2] == 0.0) and np.all(np.asarray(there.spin_2)[:2] == 0.0)
    np.testing.assert_allclose(np.asarray(there.spin_1), np.asarray(binary.spin_1), atol=1e-6)
    np.testing.assert_allclose(float(there.inclination), float(binary.inclination), atol=1e-6)
    jacobian = jax.jacfwd(lambda b: wf.to_mass_scaled_reference(b))(binary)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(jacobian))
    if dtype == jnp.float64:
        f = jnp.asarray(F)
        h1 = [np.asarray(x) for x in _physical(binary, f)]
        h2 = [np.asarray(x) for x in scaled(there, f)]
        assert overlap(h1, h2, maximise=False) < 1e-9


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_converters_transforms(dtype):
    """jit, vmap and grad, finite; float32 within 5e-4 of float64 (the
    orbital phase between the references is long for light binaries: run
    the converters in float64 there)."""
    batch = wf.Binary(*(jnp.stack([jnp.asarray(getattr(b, name), dtype) for b in CONVERTED[:4]])
                        for name in wf.Binary._fields))
    out = jax.jit(jax.vmap(wf.to_mass_scaled_reference))(batch)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in out)

    def scalar(b):
        o = wf.to_mass_scaled_reference(b)
        return jnp.sum(o.spin_1) + jnp.sum(o.spin_2) + o.inclination + jnp.sin(o.phi_ref)

    for binary in CONVERTED:
        grad = jax.jit(jax.grad(scalar))(as_dtype(binary, dtype))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in grad)
    if dtype == jnp.float32:
        double = jax.vmap(wf.to_mass_scaled_reference)(as_dtype(batch, jnp.float64))
        for name in ("spin_1", "spin_2", "inclination"):
            np.testing.assert_allclose(np.asarray(getattr(out, name), float), np.asarray(getattr(double, name)),
                                       atol=5e-4, err_msg=name)


def test_weak_precession():
    """As the in-plane spins vanish, the two conventions give the same
    waveform (maximised over time and phase)."""
    f = jnp.asarray(F)
    previous = None
    for chi_p in (0.1, 0.01):
        binary = as_dtype(wf.Binary(36.0, 29.0, (chi_p, 0.0, 0.2), (0.0, 0.5 * chi_p, -0.1), 500.0, 0.7, 1.2, 20.0),
                          jnp.float64)
        there, psi = _to_scaled(binary)
        h1 = [np.asarray(x) for x in _physical(binary, f)]
        h2 = rotated([np.asarray(x) for x in scaled(there, f)], float(psi))
        value = overlap(h1, h2)
        if previous is not None:
            assert value < 1e-2 * previous
        previous = value
    assert previous < 1e-7


def _patched_lalsimulation():
    import os
    import sys
    from pathlib import Path

    default = Path.home() / ".cache" / "waveformjax" / "patched_lalsimulation" / "liblalsimulation.37.dylib"
    path = os.environ.get("PATCHED_LALSIMULATION", str(default))
    scripts = Path(__file__).resolve().parents[2] / "waveformrs" / "scripts"
    if not Path(path).is_file() or not scripts.is_dir():
        pytest.skip("no patched LALSimulation (PATCHED_LALSIMULATION)")
    lalsim = pytest.importorskip("lalsimulation")
    sys.path.insert(0, str(scripts))
    from lalsimulation_ctypes import Library, Overlay

    return Overlay(lalsim, Library(path))


def test_against_lal():
    """The lean model with the mass-scaled reference against LAL (with the
    parity fix) given the converted binary at ``f_ref = reference_mf / M``
    (inside the band): as close as the lean model in LAL's convention is to
    LAL (well within its tolerances)."""
    import lal

    patched = _patched_lalsimulation()
    rng = np.random.default_rng(2)
    f = jnp.asarray(F)
    checked = 0
    while checked < 6:
        q, mass = rng.uniform(1, 12), rng.uniform(20, 90)
        spins = [rng.normal(size=3) for _ in range(2)]
        spins = [s / np.linalg.norm(s) * rng.uniform(0, 0.9) for s in spins]
        binary = as_dtype(wf.Binary(mass * q / (1 + q), mass / (1 + q), spins[0], spins[1], 500.0,
                                    rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), 20.0), jnp.float64)
        there, _ = _to_scaled(binary)
        f_ref = REF / (mass * MTSUN_SI)
        physical, _ = _to_physical(there, f_ref)
        sequence = lal.CreateREAL8Vector(len(F))
        sequence.data = F
        p = physical
        hp, hc = patched.SimInspiralChooseFDWaveformSequence(
            float(p.phi_ref), float(p.mass_1) * lal.MSUN_SI, float(p.mass_2) * lal.MSUN_SI, *np.asarray(p.spin_1),
            *np.asarray(p.spin_2), float(p.f_ref), 500e6 * lal.PC_SI, float(p.inclination), lal.CreateDict(),
            lalsim_approximant(), sequence)
        h_lal = [hp.data.data * 1e21, hc.data.data * 1e21]
        # (both at the same reference: the same basis of the polarizations)
        h = [np.asarray(x) for x in scaled(there, f)]
        assert overlap(h_lal, h) < 1e-4, (q, mass)
        checked += 1


def lalsim_approximant():
    import lalsimulation

    return lalsimulation.IMRPhenomXPNR


# ---------------------------------------------------------------------------
# continuity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modes,mass", [(None, 203.0), (((2, 1), (2, -1)), 51.0)])
def test_continuity(modes, mass):
    """Along the total mass across the masses where the reference crosses
    the lower end of the band (~f_min / 2 for all modes, 2 f_min for m' =
    1): no jump in the overlap (fifth differences, less their running
    median, within 8 of their noise),
    and the forward-mode derivative agrees with finite differences."""
    q = 1.6
    f = jnp.asarray(F)

    def h(m):
        b = wf.Binary(m * q / (1 + q), m / (1 + q), jnp.asarray([0.5, -0.3, 0.2]), jnp.asarray([0.2, 0.3, -0.4]),
                      jnp.asarray(500.0), jnp.asarray(0.8), jnp.asarray(0.4), jnp.asarray(0.0))
        hp, hc = lean.polarizations(b, f, mode_array=modes, reference_mf=REF)
        return jnp.concatenate([hp, hc])

    masses = mass * (1 + 0.15 * np.linspace(-1, 1, 161))
    values = np.concatenate([np.asarray(jax.jit(jax.vmap(h))(jnp.asarray(c))) for c in np.array_split(masses, 4)])
    tangents = np.concatenate([np.asarray(jax.jit(jax.vmap(lambda m: jax.jvp(h, (m,), (jnp.ones_like(m),))[1]))(
        jnp.asarray(c))) for c in np.array_split(masses, 4)])
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(tangents))
    h0 = values[80]
    o = np.real(np.conj(values) @ h0) / np.sqrt(np.sum(np.abs(values) ** 2, 1) * np.real(np.vdot(h0, h0)))
    d5 = np.diff(1 - o, 5)
    # (less a running median: the fifth differences of the smooth overlap
    # change slowly, a jump stands out of them)
    padded = np.pad(d5, 4, mode="edge")
    residual = d5 - np.median(np.lib.stride_tricks.sliding_window_view(padded, 9), axis=1)
    noise = np.median(np.abs(residual)) * 1.4826
    assert np.max(np.abs(residual)) < 8 * noise
    fd = (values[2:] - values[:-2]) / (masses[2:] - masses[:-2])[:, None]
    err = np.linalg.norm(fd - tangents[1:-1], axis=1) / np.linalg.norm(tangents[1:-1], axis=1)
    assert np.max(err) < 1e-2
