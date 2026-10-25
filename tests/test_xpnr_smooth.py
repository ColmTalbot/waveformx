# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR where LAL's waveform is discontinuous.

* The model is finite where LAL leaves the SpinTaylor angles (a reference
  frequency beyond the PN energy test, in-plane spins below ``1e-6``,
  aligned, single and vanishing spins).
* It is continuous where LAL's waveform jumps: at the end of LAL's fine
  SpinTaylor run, across ``|chi2| = 1e-3``, ``chi_perp = 1e-3`` and ``q =
  12``, across the refinement of the grid of the PNR interpolants and
  across the reference frequency at which LAL gives up the SpinTaylor
  angles (the points where LAL switches were located with the tag
  ``lal-faithful``).
* The interpolants of the PNR angles do not depend on the branch of
  ``alpha``.
* Its gradients are finite, also for aligned and vanishing spins, and
  agree with finite differences.
* It tends linearly to the waveform of aligned spins as the in-plane spins
  vanish (LAL's construction left an opening angle there), also as all
  spins vanish, and across equal masses; its derivatives along a direction
  of the in-plane spin tend to the one-sided derivative at zero.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from waveformx import Binary, WaveformOptions
from waveformx.models.xpnr import MODEL, model, pnr

OPTIONS = WaveformOptions(scaled_strain=False)
PHASE_GRID = WaveformOptions(extended_phase_grid=True, scaled_strain=False)
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


def test_validate_beyond_energy_test():
    """The host checks accept a binary for which LAL gives up the
    SpinTaylor angles."""
    MODEL.validate(binary(**BEYOND), np.asarray(frequencies(300.0, 15.0)), OPTIONS)


@pytest.mark.parametrize("spins", [
    ((0.0, 0.0, 0.3), (0.0, 0.0, -0.1)),
    ((5e-7, 0.0, 0.3), (0.0, 0.0, -0.1)),
    ((0.5, 0.2, 0.3), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.4, 0.1, 0.2)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
])
def test_finite_for_special_spins(spins):
    b = binary(spin_1=spins[0], spin_2=spins[1])
    assert finite(compiled(OPTIONS)(b, frequencies(65.0)))


def test_finite_beyond_energy_test():
    assert finite(compiled(OPTIONS)(binary(**BEYOND), frequencies(300.0, 15.0)))


def _jump(options, make, x):
    """The relative change of the polarizations across ``x`` (adjacent
    floats on either side)."""
    lo, hi = np.nextafter(x, -np.inf), np.nextafter(x, np.inf)
    f = make(lo)[1]
    return relative(compiled(options)(make(hi)[0], f), compiled(options)(make(lo)[0], f))


#: The reference frequency of BEYOND at which LAL's SpinTaylor integration
#: starts to fail (its energy test), and the second spin of the binary of
#: test_continuous_across_the_refinement_of_the_pnr_grid at which LAL
#: refines the grid of the PNR interpolants (bisected with ``lal-faithful``).
LAL_BEYOND_F_REF = 33.20221992667939
LAL_REFINEMENT = 0.11138705473915744


def test_continuous_across_reference_beyond_energy_test():
    """Across the reference frequency at which LAL's SpinTaylor angles stop
    (LAL switches to the MSA angles there)."""
    f = frequencies(300.0, 15.0)

    def make(f_ref):
        return binary(**dict(BEYOND, f_ref=f_ref)), f

    assert _jump(OPTIONS, make, LAL_BEYOND_F_REF) < 1e-6


@pytest.mark.parametrize("which", ["q12", "two_spin", "remap"])
def test_continuous_across_lal_switches(which):
    """LAL's waveform jumps across ``q = 12`` (the edge of the PNR
    transition region), ``|chi2| = 1e-3`` (the two-spin construction) and
    ``chi_perp = 1e-3`` (the remapped J frame); the model's does not."""
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
    assert _jump(OPTIONS, make, x) < 1e-6


def _mismatch(a, b):
    """``1 - <a, b> / (|a| |b|)`` of the polarizations (both, unweighted)."""
    a = np.concatenate([np.asarray(x) for x in a])
    b = np.concatenate([np.asarray(x) for x in b])
    return float(1 - np.real(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("case", ["light", "generic", "heavy"])
def test_phase_grid_close_to_the_uniform_grid(case):
    """The phase grid of the PNR interpolants (``extended_phase_grid``)
    stays well within a mismatch of 1e-3 of LAL's uniform grid."""
    total, b = {"light": (13.0, binary(mass_1=8.0, mass_2=5.0)), "generic": (65.0, binary()),
                "heavy": (250.0, binary(mass_1=150.0, mass_2=100.0))}[case]
    f = frequencies(total)
    reference = compiled(OPTIONS)(b, f)
    assert finite(reference)
    assert _mismatch(compiled(PHASE_GRID)(b, f), reference) < 1e-4


def _setup(b, f):
    """The set-up of the SpinTaylor angles (``prec, wf, grid, flags``)."""
    pnr.clear_cache()
    w, m1, m2, chi1, chi2 = model.setup_waveform(b, 0.0, f[0], f[-1])
    w.hm_release = 122019
    base, wf0 = pnr.common_setup(w, m1, m2, chi1, chi2, pnr.mode_range(model.DEFAULT_MODES))
    return model._spintaylor_setup(base, wf0, m1, m2, model.DEFAULT_MODES)


def test_alpha_continuous_where_the_inspiral_alpha_ends():
    """LAL hands the SpinTaylor ``alpha`` over to its merger-ringdown
    continuation at 0.98 of the end of the inspiral, where the two differ
    (the continuation matches the spline at 0.97 and 0.99): ``alpha`` steps
    in frequency, and the grid of the PNR interpolants samples the step
    wherever a node crosses it. The model hands over at 0.97, where the two
    agree in value and slope."""

    def step(b, f):
        prec, w, _, _ = _setup(b, f)
        at = prec.st_angles.ftrans_alpha
        mf = jnp.stack([jnp.nextafter(at, -jnp.inf), jnp.nextafter(at, jnp.inf)])
        alpha, slope = jax.jvp(lambda m: pnr.st_alpha(prec, w, m)[0], (mf,), (jnp.ones_like(mf),))
        return alpha[1] - alpha[0], (slope[1] - slope[0]) / slope[0]

    jump, kink = jax.jit(step)(binary(), frequencies(65.0))
    assert abs(float(jump)) < 1e-10 and abs(float(kink)) < 1e-8


def test_continuous_across_the_refinement_of_the_pnr_grid():
    """LAL samples the PNR angles four times more finely where L comes close
    to J (``beta_min < 0.01`` and ``beta_min / beta_max < 0.55``), and its
    waveform jumps where it starts; the model refines gradually."""
    total = 245.0

    def make(s):
        return binary(mass_1=140.0, mass_2=105.0, spin_1=(-0.18, -0.48, 0.17), spin_2=(0.24, s, 0.66),
                      f_ref=23.6), frequencies(total)

    assert _jump(OPTIONS, make, LAL_REFINEMENT) < 1e-7


def _trajectory(f, case):
    """``alpha`` (unwrapped) and ``beta`` along a path of L in the J frame:
    circling past J (``"north"``) or -J (``"south"``) at a distance of 5e-4
    near f = 60 Hz, or precessing steadily while ``beta`` goes from 0.5 to
    2.5 (``"equator"``)."""
    if case == "equator":
        return 2 * np.pi * (f - 20.0) / 10.0, 0.5 + 2.0 * (f - 20.0) / 80.0
    phase = 2 * np.pi * (f - 60.0) / 40.0
    lx, ly = 0.0205 - 0.02 * np.cos(phase), 0.02 * np.sin(phase)
    beta = np.arcsin(np.hypot(lx, ly))
    return np.unwrap(np.arctan2(ly, lx)), (beta if case == "north" else np.pi - beta)


@pytest.mark.parametrize("case", ["north", "south", "equator"])
def test_smooth_interpolants_do_not_depend_on_the_branch_of_alpha(case):
    """Where L passes close to J (or -J), ``alpha`` swings by nearly ``pi``
    between points of the PNR grid, and which way the unwrapping takes the
    swing flips with the parameters (by ``2 pi`` beyond it). The
    interpolants of the model give the same rotation either way,
    follow the swing, which is narrower than the grid's spacing, and give
    LAL's ``gamma = -int alpha' cos(beta) df`` (up to a constant) in both
    hemispheres and across the equator."""
    from waveformx.models.xpnr.pnr import PNR_GRID_CAPACITY

    n, h, i_start = 400, 0.25, 40
    f = (i_start + np.arange(PNR_GRID_CAPACITY)) * h
    alpha, beta = _trajectory(f, case)
    other = alpha + 2 * np.pi * (f > 60.0)
    valid = np.arange(PNR_GRID_CAPACITY) < n
    q = np.linspace(20.0, 100.0, 1001)
    out = []
    for a in (alpha, other):
        s = pnr._geometric_interpolants(jnp.asarray(np.where(valid, a, 0.0)), jnp.asarray(np.where(valid, beta, 0.0)),
                                        jnp.float64(i_start), jnp.float64(n), jnp.float64(h), jnp.float64(n),
                                        jnp.asarray(False))
        out.append([np.asarray(x) for x in pnr.spline_angles(s, jnp.asarray(q))])
    for k in (0, 2):
        np.testing.assert_allclose(np.exp(1j * out[0][k]), np.exp(1j * out[1][k]), atol=1e-12)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-12)
    # against the path itself, and gamma integrated finely
    fine = np.linspace(20.0, 100.0, 2_000_001)
    a_fine, b_fine = _trajectory(fine, case)
    integrand = -np.gradient(a_fine, fine) * np.cos(b_fine)
    gamma = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine))])[::2000]
    a_exact, b_exact = _trajectory(q, case)
    np.testing.assert_allclose(np.exp(1j * out[0][0]), np.exp(1j * a_exact), atol=1e-6)
    np.testing.assert_allclose(out[0][1], b_exact, atol=1e-6)
    drift = out[0][2] - out[0][2][0] - (gamma - gamma[0])
    np.testing.assert_allclose(np.exp(1j * drift), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

OPTIONS_22 = WaveformOptions(mode_array=((2, 2), (2, -2)), scaled_strain=False)
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


def test_phase_grid_gradients_finite():
    options = WaveformOptions(mode_array=((2, 2), (2, -2)), extended_phase_grid=True, scaled_strain=False)

    def loss(x):
        b = Binary(mass_1=x[0], mass_2=x[1], spin_1=x[2:5], spin_2=x[5:8], luminosity_distance=x[8],
                   inclination=x[9], phi_ref=x[10], f_ref=x[11])
        hp, hc = MODEL.polarizations(b, GRID, options)
        return jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2) * 1e42

    grad = jax.jit(jax.grad(loss))
    for which in ("generic", "aligned"):
        assert np.all(np.isfinite(np.asarray(grad(jnp.asarray(POINTS[which]))))), which


def test_vmap_matches_single_calls():
    xs = [binary(), binary(mass_1=45.0), binary(spin_2=(0.0, 0.0, 0.0))]
    batch = jax.jit(jax.vmap(lambda b: MODEL.polarizations(b, GRID, OPTIONS_22)))(
        jax.tree_util.tree_map(lambda *v: jnp.stack(v), *xs))
    single = jax.jit(lambda b: MODEL.polarizations(b, GRID, OPTIONS_22))
    for i, b in enumerate(xs):
        for a, s in zip(batch, single(b)):
            np.testing.assert_allclose(np.asarray(a[i]), np.asarray(s), rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(s))))


# ---------------------------------------------------------------------------
# vanishing spins and equal masses
# ---------------------------------------------------------------------------

def test_beta_window():
    """The window of the PNR beta: beta itself within [0, pi], LAL's window
    beyond 0.01 outside, continuous with its derivative across both."""
    from waveformx.models.xpnr import pnr_angles as pa

    inside = jnp.asarray([0.0, 1e-12, 1e-3, 0.005, 0.01, 1.0, np.pi - 0.005, np.pi])
    np.testing.assert_array_equal(np.asarray(pa.beta_window(inside)), np.asarray(inside))
    outside = jnp.asarray([-0.5, -0.1, -0.0100001, np.pi + 0.0100001, np.pi + 0.2])
    np.testing.assert_array_equal(np.asarray(pa.beta_window(outside)), np.asarray(pa.arctan_window(outside)))
    grad = jax.vmap(jax.grad(pa.beta_window))
    for edge in (0.0, -0.01, np.pi, np.pi + 0.01):
        x = jnp.asarray([edge - 1e-9, edge + 1e-9])
        values, slopes = np.asarray(pa.beta_window(x)), np.asarray(grad(x))
        assert abs(values[1] - values[0]) < 1e-8 and abs(slopes[1] - slopes[0]) < 1e-6, edge


#: The steps towards a vanishing in-plane spin, total spin or mass difference.
STEPS = (1e-4, 1e-6, 1e-8, 1e-10)


def _ray(make, base):
    """``|h(eps) - h(0)| / |h(0)|`` along ``make(eps)`` for :data:`STEPS`."""
    f = frequencies(50.0, n=256)
    h0 = compiled(OPTIONS)(base, f)
    return np.array([relative(compiled(OPTIONS)(make(eps), f), h0) for eps in STEPS])


def _assert_linear(changes):
    """The changes vanish linearly with the step: the waveform is continuous
    (LAL's floors and windows left a step of ~1e-3)."""
    rates = changes / np.asarray(STEPS)
    assert np.all(rates < 1.5 * rates[0]) and np.all(rates > rates[0] / 1.5), changes


@pytest.mark.parametrize("inclination", [0.0, 0.7, 1.5])
def test_continuous_as_in_plane_spins_vanish(inclination):
    """As the in-plane spins vanish, along several directions, the waveform
    tends linearly to that of the aligned spins: no residual opening angle
    (LAL's arctan window, the perturbed MSA roots of the single-spin system,
    the polar angles kept off the poles) along an arbitrary ``alpha``."""
    for phi in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        def make(eps):
            return binary(mass_1=30.0, mass_2=20.0, spin_1=(eps * np.cos(phi), eps * np.sin(phi), 0.3),
                          spin_2=(0.0, 0.0, -0.2), inclination=inclination)

        _assert_linear(_ray(make, make(0.0)))


#: Directions of the two spins whose aligned total spin is positive
#: (``s_tot_r > 0``).
DIRECTIONS = [((0.6, -0.2, 0.3), (0.5, 0.5, 0.1)), ((-0.3, 0.1, 0.8), (0.2, -0.7, -0.4))]


@pytest.mark.parametrize("directions", DIRECTIONS)
def test_continuous_as_all_spins_vanish(directions):
    def make(eps):
        return binary(mass_1=30.0, mass_2=20.0, spin_1=tuple(eps * np.asarray(directions[0])),
                      spin_2=tuple(eps * np.asarray(directions[1])))

    _assert_linear(_ray(make, make(0.0)))


@pytest.mark.xfail(strict=True, reason="IMRPhenomXHM: the (3,3) intermediate phase changes its collocation "
                                       "points across s_tot_r = 0")
def test_continuous_as_all_spins_vanish_negative_total_spin():
    def make(eps):
        return binary(mass_1=30.0, mass_2=20.0, spin_1=(0.0, 0.0, -eps), spin_2=(0.0, 0.0, -eps))

    _assert_linear(_ray(make, make(0.0)))


def test_continuous_at_equal_masses():
    """Across ``q = 1`` (where the bodies are swapped, and where LAL's
    estimate of the period of the two-spin oscillations is NaN, which took
    the coarse grid of the PNR interpolants rather than its limit, the
    finest)."""
    def make(eps):
        return binary(mass_1=30.0 * (1.0 + eps), mass_2=30.0, spin_1=(0.2, 0.1, 0.3), spin_2=(-0.1, 0.2, 0.3))

    base = make(0.0)
    _assert_linear(_ray(make, base))
    _assert_linear(_ray(lambda eps: make(-eps), base))


@functools.lru_cache(maxsize=None)
def _ray_function(direction):
    f = frequencies(50.0, n=128)

    def h(eps):
        b = binary(mass_1=30.0, mass_2=20.0, spin_1=(eps * direction[0], eps * direction[1], 0.3),
                   spin_2=(0.0, 0.0, -0.2))
        hp, hc = MODEL.polarizations(b, f, OPTIONS_22)
        return jnp.concatenate([hp, hc]) * 1e21

    return jax.jit(h), jax.jit(lambda eps: jax.jvp(h, (eps,), (jnp.ones_like(eps),))[1])


@pytest.mark.parametrize("phi", [0.0, np.pi])
def test_derivatives_continuous_as_in_plane_spins_vanish(phi):
    """Along a direction of the in-plane spin, the derivative agrees with
    finite differences down to tiny spins and tends to the one-sided
    derivative at zero. (The waveform has no derivative at zero: LAL's fits
    of the co-precessing modes and of the PNR angles depend on the
    magnitude of the in-plane spin, so that the one-sided derivatives along
    opposite directions differ. There the derivatives are finite.)"""
    h, dh = _ray_function((float(np.cos(phi)), float(np.sin(phi))))
    for eps in (1e-3, 1e-6, 1e-8):
        step = 0.1 * eps

        def central(s):
            return (np.asarray(h(jnp.float64(eps + s))) - np.asarray(h(jnp.float64(eps - s)))) / (2 * s)

        numeric = (4.0 * central(step / 2) - central(step)) / 3.0
        tangent = np.asarray(dh(jnp.float64(eps)))
        assert np.max(np.abs(tangent - numeric)) < 1e-3 * np.max(np.abs(numeric)), eps
    one_sided = (np.asarray(h(jnp.float64(1e-9))) - np.asarray(h(jnp.float64(0.0)))) / 1e-9
    tangent = np.asarray(dh(jnp.float64(1e-9)))
    assert np.max(np.abs(tangent - one_sided)) < 1e-4 * np.max(np.abs(one_sided))
    assert np.all(np.isfinite(np.asarray(dh(jnp.float64(0.0)))))


# ---------------------------------------------------------------------------
# the NNLO and MSA prescriptions
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _prescription(version):
    return jax.jit(lambda b, f: model.polarizations(b, f, OPTIONS, version))


@pytest.mark.parametrize("version", [102, 223])
def test_prescriptions_continuous_as_spins_vanish(version):
    """The NNLO and MSA prescriptions (``IMRPhenomXPNR_NNLO``,
    ``IMRPhenomXPNR_MSA``) tend linearly to the waveform of aligned spins
    as the in-plane spins vanish, as all spins vanish and across equal
    masses. (LAL's MSA ``beta`` is not: where the two roots of the cubic of
    ``|S|^2`` are within its rounding, ``beta`` keeps an opening angle of
    up to ``~1e-4``; the model takes the ``beta`` of a constant ``|S|``
    there, see ``pnr._msa_beta``.)"""
    function = _prescription(version)
    f = frequencies(50.0, n=256)

    def ray(make):
        h0 = function(make(0.0), f)
        return np.array([relative(function(make(eps), f), h0) for eps in STEPS])

    for phi in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        def in_plane(eps):
            return binary(mass_1=30.0, mass_2=20.0, spin_1=(eps * np.cos(phi), eps * np.sin(phi), 0.3),
                          spin_2=(0.0, 0.0, -0.2), inclination=0.7)

        _assert_linear(ray(in_plane))

    def all_spins(eps):
        return binary(mass_1=30.0, mass_2=20.0, spin_1=tuple(eps * np.asarray(DIRECTIONS[0][0])),
                      spin_2=tuple(eps * np.asarray(DIRECTIONS[0][1])))

    _assert_linear(ray(all_spins))
    for sign in (1.0, -1.0):
        def masses(eps):
            return binary(mass_1=30.0 * (1.0 + sign * eps), mass_2=30.0, spin_1=(0.2, 0.1, 0.3),
                          spin_2=(-0.1, 0.2, 0.3))

        _assert_linear(ray(masses))


@pytest.mark.parametrize("version", [102, 223])
def test_prescriptions_continuous_across_lal_switches(version):
    """Across ``chi_perp = 1e-3`` (LAL's remapped J frame) and ``q = 12``
    (the edge of the PNR transition region), as for the SpinTaylor angles."""
    function = _prescription(version)
    cases = [
        (lambda a: binary(spin_1=(0.6 * a, 0.8 * a, 0.3), spin_2=(0.0, 0.0, 0.2)), frequencies(65.0), 1e-3),
        (lambda q: binary(mass_1=120.0 * q / (1 + q), mass_2=120.0 / (1 + q), spin_1=(0.5, 0.0, 0.3),
                          spin_2=(0.0, 0.2, 0.1)), frequencies(120.0), 12.0),
    ]
    for make, f, x in cases:
        lo, hi = np.nextafter(x, -np.inf), np.nextafter(x, np.inf)
        assert relative(function(make(hi), f), function(make(lo), f)) < 1e-6
