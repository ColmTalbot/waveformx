MLX
===

Every model is also available in `MLX <https://github.com/ml-explore/mlx>`_,
in single precision on Apple GPUs (``pip install "waveformx[mlx]"``). The
MLX functions take a :class:`waveformx.Binary` of numbers or MLX arrays,
for one binary or a batch along a leading axis of its fields, and work with
MLX's own derivatives (``mx.grad``, ``mx.vjp``). Each module also has
``polarizations_and_jacobian``, which gives the waveform with its
derivatives with respect to every parameter of the binary in one call.

=================  ======================================
Model              Module
=================  ======================================
TaylorF2           :mod:`waveformx.models.taylorf2.mlx`
IMRPhenomD         :mod:`waveformx.models.phenomd.mlx`
IMRPhenomXAS       :mod:`waveformx.models.xas.mlx`
IMRPhenomXHM       :mod:`waveformx.models.xhm.mlx`
IMRPhenomXPNR      :mod:`waveformx.models.xpnr.mlx`
NRSur7dq4          :mod:`waveformx.models.nrsur7dq4.mlx`
=================  ======================================

TaylorF2
--------

.. module:: waveformx.models.taylorf2.mlx

.. autofunction:: polarizations
.. autofunction:: polarizations_and_jacobian
.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference

IMRPhenomD
----------

.. module:: waveformx.models.phenomd.mlx

.. autofunction:: polarizations
.. autofunction:: polarizations_and_jacobian
.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference

IMRPhenomXAS
------------

.. module:: waveformx.models.xas.mlx

.. autofunction:: polarizations
.. autofunction:: polarizations_and_jacobian
.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference

IMRPhenomXHM
------------

.. module:: waveformx.models.xhm.mlx

.. autofunction:: polarizations
.. autofunction:: modes
.. autofunction:: polarizations_and_jacobian
.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference

IMRPhenomXPNR
-------------

``approximant=`` selects IMRPhenomXPNR, IMRPhenomXPNR_NNLO or
IMRPhenomXPNR_MSA. ``reference_mf`` is available here too; the converters
between reference conventions are those of :doc:`binaries`.

.. module:: waveformx.models.xpnr.mlx

.. autofunction:: polarizations
.. autofunction:: polarizations_and_jacobian

NRSur7dq4
---------

The same functions as the JAX model: the pure functions at any times,
LALSimulation's waveforms on its grid, and the conditioned waveforms. It
reads the same data file (:doc:`../installation`).

.. module:: waveformx.models.nrsur7dq4.mlx

.. autofunction:: polarizations_at
.. autofunction:: modes_at
.. autofunction:: td_polarizations
.. autofunction:: td_modes
.. autofunction:: batch_td_polarizations
.. autofunction:: simulate_td
.. autofunction:: simulate_fd
.. autofunction:: simulate_td_function
.. autofunction:: simulate_fd_function
.. autofunction:: load
.. autofunction:: default_data
