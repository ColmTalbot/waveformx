waveformx
=========

Gravitational-wave waveform models for compact binaries in `JAX
<https://github.com/jax-ml/jax>`_: TaylorF2, IMRPhenomD, IMRPhenomXAS,
IMRPhenomXHM, IMRPhenomXPNR (with IMRPhenomXPNR_NNLO and IMRPhenomXPNR_MSA)
and NRSur7dq4. The models can be compiled with ``jax.jit``, evaluated for
many binaries at once with ``jax.vmap`` and differentiated with
``jax.grad``. They follow `LALSimulation
<https://git.ligo.org/lscsoft/lalsuite>`_, have the interface of the Python
bindings of `waveformrs <https://github.com/ColmTalbot/waveformrs>`_, and
are also available in `MLX <https://github.com/ml-explore/mlx>`_ for Apple
GPUs.

What each model includes, and how the models compare with LALSimulation, is
in the `README <https://github.com/ColmTalbot/waveformx#readme>`_ and the
`report <https://github.com/ColmTalbot/waveformx/tree/main/report>`_.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api
   mlx
   development

waveformx is free software, licensed under the GNU General Public License
version 3 or later.
