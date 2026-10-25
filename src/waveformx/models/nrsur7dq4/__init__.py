# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""NRSur7dq4 in JAX: the precessing numerical-relativity surrogate of Varma
et al. (2019, arXiv:1905.09300), following ``LALSimIMRPrecessingNRSur.c``
of LALSimulation 6.2.0 and its Rust translation in waveformrs.

=================  ==========================================================
module             contents
=================  ==========================================================
``_data``          reading ``NRSur7dq4_v1.0.h5`` into a pytree of arrays
``_fits``          the parameter-space fits (``NRSur7dq4_eval_fit``)
``_dynamics``      the dynamics ODE: reference time, RK4 start, AB4 steps
``_spline``        GSL's natural cubic spline and 4-point polynomial
``_frames``        coorbital -> coprecessing -> inertial frame (Wigner D)
``_waveform``      the modes and polarizations as pure JAX functions
``_driver``        LAL's checks, the output grid, batches
=================  ==========================================================

The model needs the data file ``NRSur7dq4_v1.0.h5`` (LALSuite's data, also
at https://zenodo.org/records/14999310), found through ``NRSUR7DQ4_DATA``
(the file) or ``LAL_DATA_PATH`` (its directory). It is read once.

:data:`MODEL` implements the time-domain contract of
:mod:`waveformx.models`. Its pure functions take a
:class:`~waveformx.Binary`; ``polarizations_at(binary, times, options)``
evaluates the waveform at ``times`` (s, relative to the peak), works under
``jax.jit``, ``jax.vmap`` and ``jax.grad`` with respect to every parameter
of the binary, and starts the dynamics at the start of the surrogate where
``binary.f_ref = 0`` (``td_polarizations`` replaces ``f_ref = 0`` by
``f_min``, as LAL). With the option ``reference_mf`` the binary is given
where the orbital frequency is ``pi reference_mf / M`` (``f_ref`` zero), a
reference of the same point of the inspiral for every mass, which needs no
search on the host; :mod:`.reference` converts binaries between the two
conventions with the surrogate's dynamics. It is evaluated in float32 if the binary's parameters
are float32 (with float32 data, :func:`load` with ``dtype``), else in
float64.

Every output is the strain multiplied by ``1e21`` unless the option
``scaled_strain`` is false (the strain itself, as LAL).
"""

from __future__ import annotations

from typing import Optional

from . import _driver, _waveform
from ._data import NRSurData, default_data, load

__all__ = ["MODEL", "NRSur7dq4Model", "load"]


class NRSur7dq4Model:
    """NRSur7dq4 as ``SimInspiralChooseTDWaveform`` and
    ``SimInspiralChooseTDModes`` (see :mod:`waveformx.models`)."""

    domain = "time"
    precessing = True

    @staticmethod
    def validate(binary, delta_t: float, f_min: float, options) -> None:
        """LAL's checks for concrete values (mass ratio and spin limits, the
        reference and starting frequencies, the mode array)."""
        from .. import check_reference_mf

        check_reference_mf(binary, options, "NRSur7dq4")
        _driver.setup(binary, float(delta_t), float(f_min), options)

    @staticmethod
    def td_polarizations(binary, delta_t: float, f_min: float, options):
        """``SimInspiralChooseTDWaveform``: a
        :class:`~waveformx.TimeSeriesPolarizations` of the binary's own
        length and LAL's epoch."""
        return _driver.td_polarizations(binary, float(delta_t), float(f_min), options)

    @staticmethod
    def td_modes(binary, delta_t: float, f_min: float, options):
        """``SimInspiralChooseTDModes``: a
        :class:`~waveformx.TimeSeriesModes` of the inertial modes."""
        return _driver.td_modes(binary, float(delta_t), float(f_min), options)

    @staticmethod
    def batch_td_polarizations(table, ok, delta_t: float, f_min: float, options, chunk_size: Optional[int] = None):
        """The polarizations of the rows of ``table`` with ``ok`` (``None``
        for the others), evaluated with ``jax.vmap`` in chunks."""
        return _driver.batch_td_polarizations(table, ok, float(delta_t), float(f_min), options, chunk_size)

    @staticmethod
    def polarizations_at(binary, times, options, data: Optional[NRSurData] = None):
        """``(h_+, h_x)`` at ``times`` (s, relative to the peak; zero outside
        the surrogate), pure and differentiable. ``binary.f_ref = 0`` starts
        the dynamics at the start of the surrogate (with
        ``options.reference_mf``, the binary is given at the orbital
        frequency ``pi reference_mf`` and ``f_ref`` must be zero: NaN
        otherwise, with a zero derivative). No checks are made:
        where LAL would fail to find the reference time the result is
        NaN. The precision is that of the binary's parameters (float32 or
        float64); the times, and ``data`` if given, are converted to it."""
        data = _data_for(binary, data)
        return _waveform.polarizations_at(data, binary, times, options.mode_array, _waveform.strain_scale(options),
                                          options.reference_mf)

    @staticmethod
    def modes_at(binary, times, options, data: Optional[NRSurData] = None):
        """The inertial modes ``{(l, m): h_lm}`` at ``times``, as
        :meth:`polarizations_at`."""
        data = _data_for(binary, data)
        return _waveform.modes_at(data, binary, times, options.mode_array, _waveform.strain_scale(options),
                                  options.reference_mf)


def _data_for(binary, data: Optional[NRSurData]) -> NRSurData:
    """The data in the precision of ``binary``: the default data, or
    ``data`` converted."""
    dtype = _waveform.precision(binary)
    if data is None:
        return default_data(dtype)
    return data if data.dtype == dtype else data.astype(dtype)


MODEL = NRSur7dq4Model()
