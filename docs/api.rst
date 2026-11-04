API
===

.. currentmodule:: waveformx

Everything below is importable from ``waveformx`` (its ``__all__``), but
for :mod:`waveformx.match`, :mod:`waveformx.conversions`,
:mod:`waveformx.torch` and the ``reference`` module of each model.

Models and options
------------------

.. autofunction:: approximants
.. autofunction:: is_frequency_domain
.. autofunction:: is_precessing
.. autofunction:: nrsur7dq4_data_file

.. autoclass:: Binary

.. autofunction:: parameter_table
.. autodata:: COLUMNS
   :no-value:

.. autoclass:: WaveformOptions
   :members: lalsimulation_defaults, replace, as_dict

.. autodata:: REFERENCE_MF
   :no-value:

Frequency-domain waveforms
--------------------------

.. autofunction:: polarizations
.. autofunction:: polarizations_function
.. autofunction:: polarizations_on_grid
.. autoclass:: FrequencyGrid
   :members:
.. autofunction:: modes
.. autofunction:: choose_fd_modes
.. autoclass:: FrequencyModes
   :members:

Time-domain waveforms
---------------------

.. autofunction:: td_polarizations
.. autofunction:: td_polarizations_function
.. autofunction:: td_modes
.. autoclass:: TimeSeriesPolarizations
   :members:
.. autoclass:: TimeSeriesModes
   :members:

Batches of binaries
-------------------

.. autofunction:: batch_polarizations
.. autofunction:: batch_td_polarizations
.. autoclass:: BatchPolarizations
   :members:
.. autoclass:: BatchTimeSeries
   :members:

Conditioned waveforms
---------------------

LALSimulation's ``SimInspiralTD`` and ``SimInspiralFD``, also available as
:mod:`waveformx.conversions`.

.. autofunction:: simulate_td
.. autofunction:: simulate_fd
.. autofunction:: simulate_td_function
.. autofunction:: simulate_fd_function
.. autoclass:: FrequencySeriesPolarizations
   :members:
.. autoclass:: waveformx.conversions.SimulationFunction
   :members:

Reference frequencies
---------------------

.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference

The converters of each model, which these call:

.. automodule:: waveformx.models.taylorf2.reference
   :members:
.. automodule:: waveformx.models.phenomd.reference
   :members:
.. automodule:: waveformx.models.xas.reference
   :members:
.. automodule:: waveformx.models.xhm.reference
   :members:
.. automodule:: waveformx.models.xpnr.reference
   :members:
.. automodule:: waveformx.models.nrsur7dq4.reference
   :members:

Spherical harmonics
-------------------

.. currentmodule:: waveformx

.. autofunction:: spin_weighted_spherical_harmonic

Matches
-------

.. automodule:: waveformx.match
   :members:

PyTorch
-------

.. automodule:: waveformx.torch
   :members:

Exceptions
----------

.. automodule:: waveformx.exceptions
   :members:
   :show-inheritance:
