# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""TaylorF2, the post-Newtonian stationary-phase inspiral of LALSimulation
6.2.0 with its defaults (3.5PN phase with the aligned-spin terms,
leading-order amplitude, no tidal terms), written for speed. See
:mod:`waveformx.models` for the methods of ``MODEL``.

:mod:`.pn` holds the post-Newtonian phasing (shared with IMRPhenomD),
:mod:`.core` the coefficients of each binary as a table and the waveform
at the frequencies from it, :mod:`.model` the JAX pure function,
:mod:`.api` ``MODEL`` and :mod:`.mlx` the port to MLX (Apple GPUs).
"""

from .api import MODEL

__all__ = ["MODEL"]
