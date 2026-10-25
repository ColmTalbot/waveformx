# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformx follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""IMRPhenomXPNR (:mod:`..model`) in MLX, for Apple GPUs, in
float32.

* :func:`polarizations`: ``(h_+, h_x)`` at given frequencies, for one
  binary or a batch, differentiable with ``mx.grad`` and ``mx.vjp``;
* :func:`polarizations_and_jacobian`: with their forward-mode Jacobian
  with respect to the twelve parameters of the binary.

The port is generated from the JAX model rather than written again:
the model is cut into stages that are functions of one binary's scalars,
or of its scalars and one point (:mod:`._stages`); each is traced to a
jaxpr, lowered to straight-line scalar code (:mod:`._program`) and run as
a Metal kernel over the batch and the points (:mod:`._metal`,
:mod:`._kernels`, :mod:`._build`), its derivatives from the JVP that JAX
writes. What does not have that form is written for MLX: the integration
of the precession equations (:mod:`._spintaylor`, a Metal kernel calling
the translated pieces of :mod:`._st_parts`), the searches and splines
(:mod:`._arrays`) and the moving average of the antisymmetric mode
(:mod:`._smooth`); :mod:`._pipeline` composes everything.

It needs the optional dependency ``mlx`` (``pip install waveformx[mlx]``)
and JAX (to trace the model, once per process and set of modes).

The functions take a :class:`~waveformx.Binary` of numbers or arrays
(converted to float32 MLX arrays): one binary, or a batch along a leading
axis of its fields (``mass_1`` of shape ``(B,)``, ``spin_1`` of shape
``(B, 3)``; fields without the axis are shared). Every output is the
strain multiplied by ``1e21`` unless ``scaled_strain`` is false;
``reference_mf`` selects the mass-scaled reference frequency
(``WaveformOptions.reference_mf``). Where the model fails (where the JAX
model returns NaN) the result is NaN.
"""

from __future__ import annotations

__all__ = ["polarizations", "polarizations_and_jacobian"]


def polarizations(binary, frequencies, mode_array=None, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` of IMRPhenomXPNR (see :func:`._api.polarizations`)."""
    from ._api import polarizations as function

    return function(binary, frequencies, mode_array, scaled_strain, reference_mf)


def polarizations_and_jacobian(binary, frequencies, mode_array=None, scaled_strain=True, reference_mf=None):
    """``(h_+, h_x)`` and their Jacobian (see :func:`._api.polarizations_and_jacobian`)."""
    from ._api import polarizations_and_jacobian as function

    return function(binary, frequencies, mode_array, scaled_strain, reference_mf)
