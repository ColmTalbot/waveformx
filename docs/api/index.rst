API
===

Everything in the first four pages is importable from ``waveformx``
directly; the others are modules of their own.

=====================================  ===================================================
Page                                   Contents
=====================================  ===================================================
:doc:`waveforms`                       the models, and their waveforms at given
                                       frequencies or on LALSimulation's grids, for one
                                       binary or many
:doc:`conditioned`                     LALSimulation's conditioned waveforms, and the
                                       conversions between the time and frequency domains
:doc:`binaries`                        the binary, the options, and the reference
                                       frequency
:doc:`jax`                             the models as pure JAX functions, for ``jax.jit``,
                                       ``jax.vmap`` and ``jax.grad``
:doc:`match`                           matches between waveforms (:mod:`waveformx.match`)
:doc:`torch`                           the models as PyTorch functions
                                       (:mod:`waveformx.torch`)
:doc:`mlx`                             the models in MLX, for Apple GPUs
                                       (``waveformx.models.<model>.mlx``)
:doc:`utilities`                       the NRSur7dq4 data file, spherical harmonics and
                                       exceptions
=====================================  ===================================================

.. toctree::
   :hidden:

   waveforms
   conditioned
   binaries
   jax
   match
   torch
   mlx
   utilities
