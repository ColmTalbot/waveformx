# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 against LALSimulation 6.2.0 for random binaries (slow, opt in).

Run with ``WAVEFORMJAX_SLOW=1``; ``NRSUR7DQ4_DRAWS`` sets the number of
draws (default 64) and ``-s`` prints the worst agreement. The draws cover
``1 <= q <= 6`` (either label heavier), spin magnitudes up to 0.99 in random
directions, total masses of 20 to 300 solar masses, sampling at 1 to 8 kHz,
``f_min = 0`` or around the surrogate's starting frequency, and ``f_ref = 0``
or above ``f_min``. Where LAL fails, waveformjax must raise too.
"""

import os

import numpy as np
import pytest

import waveformjax as wf
from conftest import compare, requires_nrsur
from test_nrsur7dq4 import MISMATCH, TOLERANCE, lal_polarizations

pytestmark = [
    requires_nrsur,
    pytest.mark.skipif(
        os.environ.get("WAVEFORMJAX_SLOW", "") in ("", "0"), reason="slow: set WAVEFORMJAX_SLOW=1"
    ),
]

DRAWS = int(os.environ.get("NRSUR7DQ4_DRAWS", "64"))


def draw(rng):
    """A random binary, ``delta_t`` and ``f_min``."""
    from waveformjax.models.nrsur7dq4 import _driver

    q = rng.uniform(1.0, 6.0)
    total = rng.uniform(20.0, 300.0)
    heavy, light = total * q / (1 + q), total / (1 + q)
    if rng.uniform() < 0.5:
        heavy, light = light, heavy

    def spin():
        direction = rng.normal(size=3)
        return tuple(rng.uniform(0.0, 0.99) * direction / np.linalg.norm(direction))

    binary = dict(
        mass_1=heavy,
        mass_2=light,
        spin_1=spin(),
        spin_2=spin(),
        luminosity_distance=rng.uniform(50.0, 2000.0),
        inclination=rng.uniform(0.0, np.pi),
        phi_ref=rng.uniform(0.0, 2 * np.pi),
        f_ref=0.0,
    )
    delta_t = 1.0 / rng.choice([1024, 2048, 4096, 8192])
    start = _driver.start_frequency(wf._binary.binary_from_keywords(binary))
    f_min = 0.0 if rng.uniform() < 0.3 else start * rng.uniform(0.95, 1.5)
    if rng.uniform() < 0.6:
        binary["f_ref"] = max(f_min, start) * rng.uniform(1.0, 2.0)
    return binary, float(delta_t), float(f_min)


def test_random_draws(lal):
    lal_module, lalsim = lal
    rng = np.random.default_rng(20261019)
    worst_error = worst_mismatch = 0.0
    compared = failed = 0
    for _ in range(DRAWS):
        binary, delta_t, f_min = draw(rng)
        try:
            hp, hc = lal_polarizations(lal_module, lalsim, binary, delta_t, f_min)
        except RuntimeError:
            with pytest.raises((wf.InvalidParameterError, wf.ModelError)):
                wf.td_polarizations("NRSur7dq4", delta_t, f_min, **binary)
            failed += 1
            continue
        result = wf.td_polarizations("NRSur7dq4", delta_t, f_min, scaled_strain=False, **binary)
        assert len(result.hplus) == hp.data.length, binary
        assert result.epoch == float(hp.epoch), binary
        for ours, theirs in ((result.hplus, hp.data.data), (result.hcross, hc.data.data)):
            error, mismatch = compare(ours, theirs)
            worst_error = max(worst_error, error)
            worst_mismatch = max(worst_mismatch, mismatch)
            assert error < TOLERANCE and mismatch < MISMATCH, (binary, delta_t, f_min, error, mismatch)
        compared += 1
    print(
        f"\nNRSur7dq4: {compared} draws compared, {failed} rejected by LAL and waveformjax; "
        f"worst relative error {worst_error:.2e}, worst mismatch {worst_mismatch:.2e}"
    )
    # most draws are inside the surrogate (about 9 in 10 with these priors)
    assert compared >= 0.75 * DRAWS
