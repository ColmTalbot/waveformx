# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomD, the aligned-spin (2,2) inspiral-merger-ringdown model of
LALSimulation 6.2.0, written for speed. See :mod:`waveformx.models` for the
methods of ``MODEL``.

:mod:`.fits` holds the fits (with the quasi-normal-mode frequencies as
closed-form fits in the final spin, :mod:`._qnm_fit`, in place of LAL's
interpolated table), :mod:`.core` the coefficients of each binary as a
table and the waveform at the frequencies from it, :mod:`.model` the JAX
pure function, :mod:`.api` ``MODEL`` and :mod:`.mlx` the port to MLX
(Apple GPUs).
"""

from .api import MODEL

__all__ = ["MODEL"]
