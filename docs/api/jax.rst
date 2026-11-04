Pure JAX functions
==================

.. currentmodule:: waveformx

Each of these returns the model as a pure function of a :class:`Binary`
(and the frequencies or times), for your own ``jax.jit``, ``jax.vmap`` and
``jax.grad`` code. They make no parameter checks. A leading axis on the
fields of the binary is mapped over with ``jax.vmap``; float32 parameters
run the models that support it in single precision.

.. autofunction:: polarizations_function
.. autofunction:: td_polarizations_function
.. autofunction:: simulate_td_function
.. autofunction:: simulate_fd_function
.. autoclass:: waveformx.conversions.SimulationFunction
   :members:
