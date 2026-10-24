# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXAS, the aligned-spin (2,2) mode of IMRPhenomX
(LALSimulation 6.2.0), written for speed. See :mod:`waveformx.models` for
the methods of ``MODEL``.

:mod:`.core` holds the model: the coefficients of each binary as a table
and the waveform at the frequencies from it. :mod:`.model` is the JAX pure
function, :mod:`.api` ``MODEL`` and :mod:`.mlx` the port to MLX (Apple
GPUs). The fits and the coefficients are those of IMRPhenomXPNR's (2,2)
mode (:mod:`..xpnr.internals`, :mod:`..xpnr.amp22`, :mod:`..xpnr.phase22`).
"""

from .api import MODEL

__all__ = ["MODEL"]
