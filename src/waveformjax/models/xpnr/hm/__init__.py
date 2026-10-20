# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""The co-precessing higher modes of IMRPhenomXPNR: the (2,1), (3,3), (3,2)
(with spheroidal-to-spherical mode mixing) and (4,4) modes of IMRPhenomXHM
with ``PhenomXHMReleaseVersion = 122019`` (García-Quirós et al. 2020,
arXiv:2001.10914), after waveformrs's ``imrphenomx/xhm.rs`` and
``imrphenomx/hm_*.rs``.

* :func:`mode_coefficients` (``wf``, ``phase22``, ``ell``, ``emm``): every
  frequency-independent quantity of a mode, a pytree;
* :func:`mode_h` (``coefficients``, ``mf``): ``h_{l,-m}(Mf)`` in strain units;
* :func:`frequency_sequence_one_mode`: ``XLALSimIMRPhenomXHMFrequencySequenceOneMode``;
* the quasi-normal-mode fits of the higher modes (``fring21``, ``fdamp21``,
  ..., ``fring44``, ``fdamp44`` and the (3,2) mixing coefficients
  ``re_l2m2lp2`` ... ``im_l3m2lp3``), also as the module :mod:`.qnm`, and
  :func:`~.variables.mixing_coefficients` (rounded to single precision as LAL
  stores them).
"""

from . import qnm  # noqa: F401
from .qnm import (  # noqa: F401
    fdamp21,
    fdamp32,
    fdamp33,
    fdamp44,
    fring21,
    fring32,
    fring33,
    fring44,
    im_l2m2lp2,
    im_l2m2lp3,
    im_l3m2lp2,
    im_l3m2lp3,
    re_l2m2lp2,
    re_l2m2lp3,
    re_l3m2lp2,
    re_l3m2lp3,
)
from .mode import ModeCoefficients, frequency_sequence_one_mode, mode_coefficients, mode_h, modes_coefficients  # noqa: F401
from .variables import HIGHER_MODES, hm_waveform_struct, mixing_coefficients, qnm_fits  # noqa: F401

__all__ = [
    "HIGHER_MODES",
    "ModeCoefficients",
    "mode_coefficients",
    "modes_coefficients",
    "mode_h",
    "frequency_sequence_one_mode",
    "hm_waveform_struct",
    "mixing_coefficients",
    "qnm_fits",
    "fring21", "fdamp21", "fring33", "fdamp33", "fring32", "fdamp32", "fring44", "fdamp44",
    "re_l2m2lp2", "re_l2m2lp3", "re_l3m2lp2", "re_l3m2lp3",
    "im_l2m2lp2", "im_l2m2lp3", "im_l3m2lp2", "im_l3m2lp3",
]
