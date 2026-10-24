# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXHM, the aligned-spin model with higher modes of IMRPhenomX
(LALSimulation 6.2.0, ``PhenomXHMReleaseVersion = 122022``), written for
speed. See :mod:`waveformjax.models` for the methods of ``MODEL``.

:mod:`.reference` is LAL's computation in ``Mf``; :mod:`.core` the model as
tables of coefficients per binary and a short formula per frequency (the
(2,2) mode is IMRPhenomXAS's, :mod:`..xas.core`); :mod:`.model` the JAX
pure functions (modes and polarizations), :mod:`.api` ``MODEL`` and
:mod:`.mlx` the port to MLX (Apple GPUs).
"""

from .api import MODEL

__all__ = ["MODEL"]
