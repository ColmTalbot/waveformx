# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""``MODEL`` of IMRPhenomXHM: the contract of :mod:`waveformx.models`.

* :meth:`XHMModel.polarizations` is pure (``jax.jit`` with ``options``
  static, ``jax.vmap``, ``jax.grad``, ``jax.jvp``);
* :meth:`XHMModel.modes` and :meth:`XHMModel.choose_fd_modes` return the
  modes in the conventions of ``SimInspiralChooseFDModes`` (two-sided
  frequencies), those whose sum is the polarizations (see
  :func:`.model.modes_and_polarizations`);
* :meth:`XHMModel.validate` applies LALSimulation's checks to concrete
  values (those of IMRPhenomXAS).

One code path: LAL's defaults (``PhenomXHMReleaseVersion = 122022``, no
multibanding) and every mode; a mode array is not supported (select modes
from :meth:`XHMModel.modes`).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..._results import FrequencyModes
from ...exceptions import UnsupportedError
from ..xas.api import check
from ..xpnr._math import MTSUN_SI
from . import core, model

__all__ = ["MODEL", "XHMModel"]


@functools.partial(jax.jit, static_argnums=(2,))
def _modes(binary, frequencies, options):
    return model.modes_and_polarizations(binary, frequencies, options)[0]


def _two_sided(modes, frequencies, inclination):
    """The modes on ``-f[::-1], f``: ``h_{l,-m}`` at the positive
    frequencies, ``h_{l,m}(-f) = (-1)^l conj(h_{l,-m}(f))`` at the negative
    ones."""
    zero = jnp.zeros_like(modes[(2, 2)])
    out = {}
    for (ell, emm), h in modes.items():
        sign = -1.0 if ell % 2 else 1.0
        out[(ell, -emm)] = jnp.concatenate([zero, h])
        out[(ell, emm)] = jnp.concatenate([(sign * jnp.conj(h))[::-1], zero])
    two_sided = jnp.concatenate([-frequencies[::-1], frequencies])
    return FrequencyModes(two_sided, dict(sorted(out.items())), float(inclination), float(np.pi / 2), 0.0)


class XHMModel:
    """IMRPhenomXHM (``SimInspiralChooseFDWaveformSequence`` and
    ``SimInspiralChooseFDModes``)."""

    domain = "frequency"
    precessing = False

    def validate(self, binary, frequencies, options):
        """LALSimulation's checks of IMRPhenomXHM for concrete values."""
        if options.mode_array is not None:
            raise UnsupportedError("IMRPhenomXHM computes every mode: select them from modes()")
        check(binary, frequencies, "IMRPhenomXHM")

    def polarizations(self, binary, frequencies, options):
        """``SimInspiralChooseFDWaveformSequence`` (see :func:`.model.polarizations`)."""
        return model.polarizations(binary, frequencies, options)

    def modes(self, binary, frequencies, options):
        """The modes at ``frequencies`` on the two-sided frequencies
        ``-f[::-1], f``."""
        frequencies = jnp.asarray(frequencies, jnp.float64)
        return _two_sided(_modes(binary, frequencies, options), frequencies, binary.inclination)

    def choose_fd_modes(self, binary, delta_f, f_min, f_max, options):
        """``SimInspiralChooseFDModes``: the modes on LAL's two-sided uniform
        grid ``k delta_f`` (zero below ``f_min`` and above ``min(f_max,
        f_cut)``, ``f_ref = 0`` meaning ``f_min``)."""
        m1, m2 = float(binary.mass_1), float(binary.mass_2)
        chi1, chi2 = float(np.asarray(binary.spin_1)[2]), float(np.asarray(binary.spin_2)[2])
        if m1 < m2:
            m1, m2, chi1, chi2 = m2, m1, chi2, chi1
        eta = m1 * m2 / (m1 + m2) ** 2
        delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
        chi_eff = 0.5 * (1.0 + delta) * chi1 + 0.5 * (1.0 - delta) * chi2
        f_cut = (0.33 if chi_eff > 0.99 else 0.3) / ((m1 + m2) * MTSUN_SI)
        f_max_prime = min(f_max if f_max != 0.0 else f_cut, f_cut)
        i_start = int(f_min / delta_f)
        i_stop = int(f_max_prime / delta_f) + 1
        last = f_max if f_max_prime < f_max else f_max_prime
        n_full = int(2.0 ** np.ceil(np.log2(int(last / delta_f)))) + 1
        n = max(n_full, i_stop)
        if float(binary.f_ref) == 0.0:
            binary = binary._replace(f_ref=f_min)
        band = jnp.arange(i_start, i_stop) * delta_f
        modes = _modes(binary, band, options)
        out = {}
        for (ell, emm), h in modes.items():
            # (the one-mode functions vanish above f_max_prime)
            full = jnp.zeros(n, jnp.complex128).at[i_start:i_stop].set(h)[:n_full]
            out[(ell, emm)] = full
        frequencies = jnp.arange(n_full) * delta_f
        two_sided = _two_sided(out, frequencies, binary.inclination)
        # (a zero frequency appears once)
        keep = np.r_[np.arange(n_full - 1), np.arange(n_full, 2 * n_full)]
        modes2 = {key: value[keep] for key, value in two_sided.modes.items()}
        return FrequencyModes(two_sided.frequencies[keep], modes2, two_sided.theta, two_sided.phi, 0.0)


MODEL = XHMModel()
