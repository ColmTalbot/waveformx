# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR in JAX (not implemented yet): see :mod:`waveformjax.models` for
the methods ``MODEL`` must provide."""


class _NotImplemented:
    domain = "frequency"
    precessing = True

    def __getattr__(self, name):
        raise NotImplementedError("IMRPhenomXPNR is not implemented yet")


MODEL = _NotImplemented()
