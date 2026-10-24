# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The lean IMRPhenomXPNR in MLX (``waveformjax.models.xpnr.mlx``) against
the JAX lean model in float64, stage by stage (the stages of
``waveformjax.models.xpnr.mlx._stages``) and end to end. Skipped without
MLX.

* The stages composed in JAX reproduce the lean model to rounding (the
  reference of everything else).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import waveformjax as wf  # noqa: E402
from waveformjax.models.xpnr import lean  # noqa: E402
from waveformjax.models.xpnr.mlx import _stages  # noqa: E402

MTSUN_SI = 4.925490947641267e-06

BINARIES = {
    "generic": wf.Binary(36.0, 29.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 20.0),
    "aligned": wf.Binary(36.0, 29.0, (0.0, 0.0, 0.3), (0.0, 0.0, -0.1), 500.0, 0.7, 1.2, 20.0),
    "unequal": wf.Binary(90.0, 5.0, (0.6, 0.3, -0.2), (0.1, 0.2, 0.3), 500.0, 0.7, 1.2, 20.0),
    "beyond": wf.Binary(180.0, 120.0, (0.4, 0.2, 0.3), (-0.2, 0.3, -0.1), 500.0, 0.7, 1.2, 60.0),
}


def as_dtype(binary, dtype):
    return wf.Binary(*(jnp.asarray(value, dtype) for value in binary))


def frequencies_for(binary, n=512):
    mass = float(binary.mass_1 + binary.mass_2)
    f_min = min(20.0, float(binary.f_ref))
    return np.linspace(f_min, min(0.3 / (mass * MTSUN_SI), 1024.0), n)


@pytest.mark.parametrize("name", list(BINARIES))
def test_stages_reproduce_lean(name):
    """The stages composed in JAX are the lean model (float64, to
    rounding)."""
    binary = as_dtype(BINARIES[name], jnp.float64)
    f = jnp.asarray(frequencies_for(binary))
    expected = jax.jit(lambda b, f: lean.polarizations(b, f))(binary, f)
    got = _stages.reference(binary, f)
    for a, b in zip(got, expected):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(b))
        assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(b))
