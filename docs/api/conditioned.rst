Conditioned waveforms
=====================

.. module:: waveformx.conversions

.. currentmodule:: waveformx

LALSimulation's ``SimInspiralTD`` and ``SimInspiralFD``: the waveform of a
model in either domain, with LALSimulation's tapers, high-pass filter,
lengths and start times. IMRPhenomXPNR (and IMRPhenomXPNR_NNLO and
IMRPhenomXPNR_MSA) are available in the time domain, NRSur7dq4 in the
frequency domain, and each also conditioned in its own domain. The same
functions are in :mod:`waveformx.conversions`; their pure versions are in
:doc:`jax`.

.. autofunction:: simulate_td
.. autofunction:: simulate_fd
.. autoclass:: FrequencySeriesPolarizations
   :members:
