# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR in JAX: LALSimulation 6.2.0's model with the parity fix of
lalsuite !2593, after waveformrs, without LAL's numerical procedure (see
:mod:`.model`; the version that reproduces it is the tag ``lal-faithful``).
See :mod:`waveformx.models` for the methods of ``MODEL``.

The modules follow LAL's files: :mod:`.internals`, :mod:`.phase22`,
:mod:`.amp22`, :mod:`.xas` (the (2,2) mode), :mod:`.hm` (the higher modes
of release 122019), :mod:`.precession`, :mod:`.nnlo`, :mod:`.msa`,
:mod:`.spintaylor` (the SpinTaylor angles), :mod:`.pnr_angles`,
:mod:`.pnr` (the PNR machinery), :mod:`.antisymmetric`, :mod:`.model` (the
polarizations and J-frame modes), :mod:`.reference` (the reference
conventions) and :mod:`.api` (``MODEL``, and ``MODEL_NNLO`` and
``MODEL_MSA``, the approximants ``IMRPhenomXPNR_NNLO`` and
``IMRPhenomXPNR_MSA``: the PNR angles built on the NNLO and MSA angles,
LAL's ``PhenomXPrecVersion`` 102 and 223, rather than the SpinTaylor
angles). The fits are generated from waveformrs into :mod:`._generated`.
"""

from .api import MODEL, MODEL_MSA, MODEL_NNLO

__all__ = ["MODEL", "MODEL_MSA", "MODEL_NNLO"]
