waveformx
=========

Gravitational-wave waveform models for compact binaries in `JAX
<https://github.com/jax-ml/jax>`_: TaylorF2, IMRPhenomD, IMRPhenomXAS,
IMRPhenomXHM, IMRPhenomXPNR (with IMRPhenomXPNR_NNLO and IMRPhenomXPNR_MSA)
and NRSur7dq4. The models can be compiled with ``jax.jit``, evaluated for
many binaries at once with ``jax.vmap`` and differentiated with
``jax.grad``. They follow `LALSimulation
<https://git.ligo.org/lscsoft/lalsuite>`_, with its names, parameters and
options, and are checked against it.

Every model can also be called from `PyTorch <https://pytorch.org>`_, with
gradients for PyTorch's autograd, and is available in `MLX
<https://github.com/ml-explore/mlx>`_, in single precision on Apple GPUs.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   models
   lalsimulation
   api/index
   development

Features
--------

* LALSimulation's interface: polarizations, modes and batches of binaries,
  in the frequency and time domains (:doc:`api/waveforms`).
* Pure JAX functions of a :class:`~waveformx.Binary` for use inside your
  own ``jit``, ``vmap`` and ``grad`` code (:doc:`api/jax`).
* LALSimulation's conditioned waveforms and its conversions between the
  time and frequency domains (``SimInspiralTD`` and ``SimInspiralFD``), also
  as pure functions that can be differentiated (:doc:`api/conditioned`).
* Single precision (float32) in the pure function of every model when the
  binary's parameters are float32, including the derivatives.
* Matches between waveforms, maximised over time and phase or over the
  detector's polarisation response, as pure functions that can be compiled,
  vectorised and differentiated (:doc:`api/match`).
* A reference frequency scaled by the binary's total mass for every model,
  with converters between the two conventions (:doc:`api/binaries`).
* PyTorch functions of every model (:doc:`api/torch`) and MLX versions of
  every model (:doc:`api/mlx`).

License
-------

waveformx is free software, licensed under the GNU General Public License
version 3 or later. It is derived from LALSuite and the GNU Scientific
Library; ``NOTICE.md`` in the repository lists the original code and its
authors.
