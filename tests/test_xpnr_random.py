# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR against ``SimInspiralChooseFDWaveformSequence`` for random
binaries (opt-in: set ``WAVEFORMJAX_SLOW=1``).

The draws follow waveformrs's ``scripts/validate_against_lalsimulation.py``:
total masses log-uniform in 5 to 300 solar masses, mass ratios uniform in
1/20 to 1, isotropic spins of magnitude up to 0.99, ``f_min`` among 10, 15
and 20 Hz, ``f_ref = f_min`` or uniform up to ``3 f_min``, isotropic lines of
sight. The frequencies are 256 geometrically spaced points from ``f_min`` to
``min(0.3 / M, 2048 Hz)`` (one compilation for all draws) and the model
without the parity fix is compared with the installed LALSimulation. Where
LAL fails the model must give NaN.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import compare
from waveformjax import Binary, WaveformOptions
from waveformjax.models.xpnr import MODEL

pytestmark = pytest.mark.skipif(os.environ.get("WAVEFORMJAX_SLOW") != "1",
                                reason="slow: set WAVEFORMJAX_SLOW=1 to run")

N_DRAWS = 64
TOLERANCE = (1e-4, 1e-9)
MTSUN_SI = 4.925490947641267e-06


def draws(seed=20261020):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < N_DRAWS:
        total_mass = np.exp(rng.uniform(np.log(5.0), np.log(300.0)))
        mass_ratio = rng.uniform(1 / 20, 1.0)
        mass_1 = total_mass / (1 + mass_ratio)
        f_min = float(rng.choice([10.0, 15.0, 20.0]))
        f_max = min(0.3 / (total_mass * MTSUN_SI), 2048.0)
        if f_max <= 2 * f_min:
            continue
        spins = []
        for _ in range(2):
            magnitude = rng.uniform(0, 0.99)
            cos_theta = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * np.pi)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            spins.append(list(magnitude * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])))
        f_ref = f_min if rng.uniform() < 0.5 else rng.uniform(f_min, 3 * f_min)
        out.append(dict(
            mass_1=mass_1, mass_2=total_mass - mass_1, spin_1=spins[0], spin_2=spins[1],
            luminosity_distance=float(np.exp(rng.uniform(np.log(100), np.log(3000)))),
            inclination=float(np.arccos(rng.uniform(-1, 1))), phi_ref=float(rng.uniform(0, 2 * np.pi)),
            f_ref=float(f_ref), frequencies=np.geomspace(f_min, f_max, 256),
        ))
    return out


DRAWS = draws()
POLARIZATIONS = jax.jit(lambda b, f: MODEL.polarizations(b, f, WaveformOptions(pnr_parity_fix=False, version_fallback=True)))


@pytest.mark.parametrize("index", range(N_DRAWS))
def test_random_binary_matches_lalsimulation(lal, index):
    lal_module, lalsim = lal
    case = DRAWS[index]
    vector = lal_module.CreateREAL8Vector(len(case["frequencies"]))
    vector.data = case["frequencies"]
    binary = Binary(mass_1=jnp.float64(case["mass_1"]), mass_2=jnp.float64(case["mass_2"]),
                    spin_1=jnp.asarray(case["spin_1"]), spin_2=jnp.asarray(case["spin_2"]),
                    luminosity_distance=jnp.float64(case["luminosity_distance"]),
                    inclination=jnp.float64(case["inclination"]), phi_ref=jnp.float64(case["phi_ref"]),
                    f_ref=jnp.float64(case["f_ref"]))
    hp, hc = (np.asarray(h) for h in POLARIZATIONS(binary, jnp.asarray(case["frequencies"])))
    try:
        reference = lalsim.SimInspiralChooseFDWaveformSequence(
            case["phi_ref"], case["mass_1"] * lal_module.MSUN_SI, case["mass_2"] * lal_module.MSUN_SI,
            *case["spin_1"], *case["spin_2"], case["f_ref"], case["luminosity_distance"] * 1e6 * lal_module.PC_SI,
            case["inclination"], None, lalsim.IMRPhenomXPNR, vector,
        )
    except RuntimeError:
        assert np.all(np.isnan(hp)) and np.all(np.isnan(hc))
        return
    for test, ref in zip((hp, hc), (reference[0].data.data, reference[1].data.data)):
        error, mismatch = compare(test, ref)
        print(f"draw {index}: masses ({case['mass_1']:.2f}, {case['mass_2']:.2f}) error {error:.2e} mismatch {mismatch:.2e}")
        assert error < TOLERANCE[0] and mismatch < TOLERANCE[1], (error, mismatch)
