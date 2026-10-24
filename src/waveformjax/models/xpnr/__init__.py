# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR in JAX (LALSimulation 6.2.0, by default with the parity fix
of lalsuite master), after waveformrs. See :mod:`waveformjax.models` for the
methods of ``MODEL``.

The modules follow LAL's files: :mod:`.internals`, :mod:`.phase22`,
:mod:`.amp22`, :mod:`.xas` (the (2,2) mode), :mod:`.hm` (the higher modes
of release 122019), :mod:`.precession`, :mod:`.nnlo`, :mod:`.msa`,
:mod:`.spintaylor` (the Euler angles of versions 102, 223 and 330),
:mod:`.pnr_angles`, :mod:`.pnr` (the PNR machinery),
:mod:`.antisymmetric`, :mod:`.model` (the polarizations and J-frame modes),
:mod:`.lean` (the lean model) and :mod:`.api` (``MODEL``). The fits are generated from waveformrs into
:mod:`._generated`.
"""

from .api import MODEL

__all__ = ["MODEL"]
