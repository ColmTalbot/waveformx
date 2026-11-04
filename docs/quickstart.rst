Quick start
===========

JAX
---

The functions of the package take the binary as keyword arguments, with the
masses in solar masses and the distance in Mpc, and LALSimulation's names
for the models (:func:`waveformx.approximants`):

.. code-block:: python

    import numpy as np
    import waveformx

    frequencies = np.linspace(20.0, 1024.0, 4017)
    hplus, hcross = waveformx.polarizations(
        "IMRPhenomXPNR", frequencies, mass_1=36.0, mass_2=29.0,
        spin_1=(0.4, 0.2, 0.3), spin_2=(-0.2, 0.3, -0.1),
        luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0,
    )

Options (:class:`waveformx.WaveformOptions`) are keywords too, e.g.
``mode_array=((2, 2), (2, -2))`` or ``scaled_strain=False`` for the strain
itself rather than the strain times ``1e21``.

For your own compiled, vectorised or differentiated code, the pure function
of a model takes a :class:`waveformx.Binary`:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    model = waveformx.polarizations_function("IMRPhenomXAS")
    binary = waveformx.Binary(
        mass_1=36.0, mass_2=29.0, spin_1=jnp.array([0.0, 0.0, 0.3]),
        spin_2=jnp.array([0.0, 0.0, -0.1]), luminosity_distance=500.0, f_ref=20.0,
    )

    def power(mass_1):
        hplus, _ = model(binary._replace(mass_1=mass_1), frequencies)
        return jnp.sum(jnp.abs(hplus) ** 2)

    derivative = jax.jit(jax.grad(power))(36.0)

``jax.vmap`` maps over a leading axis of the fields of the binary;
:func:`waveformx.batch_polarizations` does that for a table of binaries.

PyTorch
-------

The same pure functions are available to PyTorch in :mod:`waveformx.torch`
(:doc:`api/torch`). They take a :class:`~waveformx.Binary` of torch tensors
(or plain numbers), for one binary or a batch along a leading axis, and
give torch tensors that PyTorch's autograd can differentiate. The waveform
and its derivatives are computed by the JAX model:

.. code-block:: python

    import torch
    import waveformx
    import waveformx.torch

    model = waveformx.torch.polarizations_function("IMRPhenomXAS")
    frequencies = torch.linspace(20.0, 1024.0, 4017, dtype=torch.float64)
    mass_1 = torch.tensor(36.0, dtype=torch.float64, requires_grad=True)
    binary = waveformx.Binary(
        mass_1=mass_1, mass_2=29.0, spin_1=(0.0, 0.0, 0.3),
        spin_2=(0.0, 0.0, -0.1), luminosity_distance=500.0, f_ref=20.0,
    )
    hplus, hcross = model(binary, frequencies)
    (hplus.abs() ** 2).sum().backward()  # the derivative is in mass_1.grad

MLX
---

The MLX versions (:doc:`api/mlx`) take the same :class:`~waveformx.Binary`, for
one binary or a batch along a leading axis, in single precision, and work
with MLX's own derivatives:

.. code-block:: python

    import mlx.core as mx
    import numpy as np
    import waveformx
    from waveformx.models.xpnr import mlx as xpnr_mlx

    frequencies = mx.array(np.linspace(20.0, 1024.0, 4017), dtype=mx.float32)
    binaries = waveformx.Binary(
        mass_1=mx.array([36.0, 30.0, 50.0]), mass_2=mx.array([29.0, 25.0, 20.0]),
        spin_1=mx.array([[0.4, 0.2, 0.3], [0.1, 0.0, 0.5], [0.0, 0.3, -0.2]]),
        spin_2=mx.array([[-0.2, 0.3, -0.1], [0.0, 0.1, 0.2], [0.2, 0.0, 0.1]]),
        luminosity_distance=500.0, inclination=0.7, phi_ref=1.2, f_ref=20.0,
    )
    hplus, hcross = xpnr_mlx.polarizations(binaries, frequencies)

    def power(mass_1):
        hplus, _ = xpnr_mlx.polarizations(binaries._replace(mass_1=mass_1), frequencies)
        return mx.sum(mx.abs(hplus) ** 2)

    derivative = mx.grad(power)(binaries.mass_1)
