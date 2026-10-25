# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The models as the conversions see them.

* :data:`XPNR`: IMRPhenomXPNR as ``SimInspiralChooseFDWaveform`` on the
  uniform grid ``k delta_f`` (multibanding off): the set-up of
  ``IMRPhenomXSetWaveformVariables`` with ``delta_f``, ``f_min`` and
  ``f_max``, evaluated at the frequencies of the band (waveformrs's
  ``IMRPhenomXPNR::polarizations_uniform``), and the cutoff ``f_max'`` that
  fixes the band.
* :data:`NRSUR7DQ4`: NRSur7dq4 as ``SimInspiralChooseTDWaveform``, on LAL's
  output grid (host) and at given times (pure).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from . import _conditioning as c

__all__ = ["XPNR", "NRSUR7DQ4", "GENERATORS"]


_UNIFORM = {}


def _uniform_function(options):
    """:meth:`_XPNR.uniform` of ``((binary, delta_f, f_min, f_max),
    frequencies)`` for ``options``: the derivatives of the model are
    forward-mode Jacobians (:func:`waveformx.models.xpnr._transforms.differentiable`),
    here with respect to the binary and the three scalars of the set-up."""
    from ..models.xpnr import model, pnr
    from ..models.xpnr._transforms import differentiable

    def single(params, frequencies):
        binary, delta_f, f_min, f_max = params
        pnr.clear_cache()
        modes = model.coprecessing_modes(options.mode_array)
        data = model._decide(binary, frequencies, delta_f, f_min, f_max, modes,
                             phase_grid=options.extended_phase_grid)
        hp, hc = model._twist_polarizations(data, modes, frequencies)
        nan = jnp.asarray(jnp.nan + 1j * jnp.nan, jnp.complex128)
        return jnp.where(data.error, nan, hp), jnp.where(data.error, nan, hc)

    return differentiable(single)


class _XPNR:
    """IMRPhenomXPNR on LAL's uniform grid."""

    name = "IMRPhenomXPNR"
    domain = "frequency"

    @staticmethod
    def uniform(binary, frequencies, delta_f, f_min, f_max, options):
        """``(h_+, h_x)`` at ``frequencies`` with LAL's set-up for the
        uniform grid of spacing ``delta_f`` from ``f_min`` to ``f_max`` (all
        may be traced); NaN where LAL fails. The values depend on the whole
        array: the antisymmetric amplitude ratio is a moving average along
        it (see :func:`._from_fd.model_frequencies`). As
        :func:`waveformx.models.xpnr.model.polarizations`, batched under
        ``jax.vmap``, with derivatives from forward-mode Jacobians with
        respect to the binary and to ``delta_f``, ``f_min`` and ``f_max``
        (which may depend on the binary)."""
        if options not in _UNIFORM:
            _UNIFORM[options] = _uniform_function(options)
        scalars = tuple(jnp.asarray(x, jnp.float64) for x in (delta_f, f_min, f_max))
        return _UNIFORM[options]((binary,) + scalars, jnp.asarray(frequencies, jnp.float64))

    @staticmethod
    def cutoff(binary):
        """``f_max -> f_max'``: ``min(f_max, f_cut)`` (``f_cut`` for ``f_max
        = 0``) with ``f_cut = 0.3 / M`` (``0.33 / M`` for ``chi_eff >
        0.99``), as ``IMRPhenomXSetWaveformVariables`` computes it."""
        m1 = float(binary.mass_1) * c.MSUN_SI / c.MSUN_SI
        m2 = float(binary.mass_2) * c.MSUN_SI / c.MSUN_SI
        chi1 = float(np.asarray(binary.spin_1)[2])
        chi2 = float(np.asarray(binary.spin_2)[2])
        if m1 < m2:
            m1, m2, chi1, chi2 = m2, m1, chi2, chi1
        delta = abs((m1 - m2) / (m1 + m2))
        eta = min(abs(0.25 * c.HOST.fma(-delta, delta, 1.0)), 0.25)
        delta = math.sqrt(max(c.HOST.fma(-4.0, eta, 1.0), 0.0))
        chi_eff = c.HOST.fma(0.5 * (1.0 + delta), chi1, 0.5 * (1.0 - delta) * chi2)
        f_cut = (0.33 if chi_eff > 0.99 else 0.3) / ((m1 + m2) * c.MTSUN_SI)

        def f_max_prime(f_max):
            return min(f_max if f_max != 0.0 else f_cut, f_cut)

        return f_max_prime

    @staticmethod
    def validate(binary, f_min, f_max_prime, options):
        """LAL's checks of the model for the band ``[f_min, f_max']``."""
        from ..models.xpnr import MODEL

        MODEL.validate(binary, np.array([f_min, f_max_prime]), options)


class _NRSur7dq4:
    """NRSur7dq4 as ``SimInspiralChooseTDWaveform``."""

    name = "NRSur7dq4"
    domain = "time"

    @staticmethod
    def setup(binary, delta_t, f_min, options):
        """LAL's checks and output grid (see
        :func:`waveformx.models.nrsur7dq4._driver.setup`)."""
        from ..models.nrsur7dq4 import _driver

        return _driver.setup(binary, float(delta_t), float(f_min), options)

    @staticmethod
    def evaluate(grid, options):
        """The polarizations on the grid of :meth:`setup`, in a buffer of
        a bucketed length (zero-padded, so that compilations are shared)
        and their number of samples."""
        from ..models.nrsur7dq4 import _driver, _waveform
        from ..models.nrsur7dq4._data import default_data

        data = default_data()
        n = _driver.bucket(grid.n)
        hp, hc = _driver._polarizations_grid(data, _driver._as_numpy(grid.binary), grid.t0, grid.dt,
                                             _waveform.strain_scale(options), n, options.mode_array)
        inside = jnp.arange(n) < grid.n
        return jnp.where(inside, hp, 0.0), jnp.where(inside, hc, 0.0)

    @staticmethod
    def times(grid):
        """The times (s, relative to the peak) of LAL's samples."""
        return grid.m_tot_sec * (grid.t0 + grid.dt * np.arange(grid.n))

    @staticmethod
    def at(binary, times, options):
        """The polarizations at ``times`` (pure)."""
        from ..models.nrsur7dq4 import MODEL

        return MODEL.polarizations_at(binary, times, options)


XPNR = _XPNR()
NRSUR7DQ4 = _NRSur7dq4()
GENERATORS = {model.name: model for model in (XPNR, NRSUR7DQ4)}
