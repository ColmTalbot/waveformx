# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR and NRSur7dq4 in JAX, with the interface of waveformrs.

The models are pure JAX functions of a :class:`Binary`: they can be compiled
with ``jax.jit``, mapped over many binaries with ``jax.vmap`` and
differentiated with ``jax.grad``. The functions below wrap them with the
signatures of waveformrs's Python bindings: the binary as keyword arguments
(``mass_1``, ``mass_2`` in solar masses, ``spin_1``, ``spin_2``,
``luminosity_distance`` in Mpc, ``inclination``, ``phi_ref``, ``f_ref``) and
the options of :class:`WaveformOptions` as keywords or ``options=``.

Importing waveformx enables 64-bit floats in JAX
(``jax_enable_x64``), which the models need.

>>> import numpy as np
>>> import waveformx
>>> frequencies = np.linspace(20.0, 1024.0, 4017)
>>> hplus, hcross = waveformx.polarizations(
...     "IMRPhenomXPNR", frequencies, mass_1=36.0, mass_2=29.0,
...     spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1),
...     luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0,
... )
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import conversions, exceptions, match  # noqa: E402
from ._api import (  # noqa: E402
    approximants,
    batch_polarizations,
    batch_td_polarizations,
    choose_fd_modes,
    is_frequency_domain,
    is_precessing,
    modes,
    nrsur7dq4_data_file,
    polarizations,
    polarizations_function,
    polarizations_on_grid,
    simulate_fd,
    simulate_fd_function,
    simulate_td,
    simulate_td_function,
    td_modes,
    td_polarizations,
    td_polarizations_function,
)
from ._binary import COLUMNS, Binary, parameter_table  # noqa: E402
from ._harmonics import spin_weighted_spherical_harmonic  # noqa: E402
from ._options import REFERENCE_MF, WaveformOptions  # noqa: E402
from ._results import (  # noqa: E402
    BatchPolarizations,
    BatchTimeSeries,
    FrequencyGrid,
    FrequencyModes,
    FrequencySeriesPolarizations,
    TimeSeriesModes,
    TimeSeriesPolarizations,
)
from .exceptions import (  # noqa: E402
    DataFileError,
    DataFileNotFoundError,
    InvalidParameterError,
    ModelError,
    UnknownApproximantError,
    UnsupportedError,
    WaveformError,
)

from ._reference import to_mass_scaled_reference, to_physical_reference  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "approximants",
    "is_frequency_domain",
    "is_precessing",
    "WaveformOptions",
    "REFERENCE_MF",
    "to_mass_scaled_reference",
    "to_physical_reference",
    "Binary",
    "COLUMNS",
    "parameter_table",
    "polarizations",
    "polarizations_function",
    "FrequencyGrid",
    "polarizations_on_grid",
    "modes",
    "choose_fd_modes",
    "FrequencyModes",
    "td_polarizations",
    "td_polarizations_function",
    "td_modes",
    "simulate_td",
    "simulate_fd",
    "simulate_td_function",
    "simulate_fd_function",
    "TimeSeriesPolarizations",
    "FrequencySeriesPolarizations",
    "TimeSeriesModes",
    "nrsur7dq4_data_file",
    "batch_polarizations",
    "batch_td_polarizations",
    "BatchPolarizations",
    "BatchTimeSeries",
    "spin_weighted_spherical_harmonic",
    "conversions",
    "match",
    "exceptions",
    "WaveformError",
    "InvalidParameterError",
    "UnsupportedError",
    "UnknownApproximantError",
    "ModelError",
    "DataFileError",
    "DataFileNotFoundError",
]
