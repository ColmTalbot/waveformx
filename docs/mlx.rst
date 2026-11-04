MLX
===

Every model is also available in `MLX <https://github.com/ml-explore/mlx>`_,
in single precision on Apple GPUs (``pip install "waveformx[mlx]"``). The
MLX versions take a :class:`waveformx.Binary` of numbers or MLX arrays, for
one binary or a batch along a leading axis, and work with MLX's own
derivatives (``mx.grad``, ``mx.vjp``); see the :doc:`quickstart`.

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

.. automodule:: waveformx.models.taylorf2.mlx
   :members:
.. automodule:: waveformx.models.phenomd.mlx
   :members:
.. automodule:: waveformx.models.xas.mlx
   :members:
.. automodule:: waveformx.models.xhm.mlx
   :members:
.. automodule:: waveformx.models.xpnr.mlx
   :members:
.. automodule:: waveformx.models.nrsur7dq4.mlx
   :members:
