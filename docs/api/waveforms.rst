Waveforms
=========

.. currentmodule:: waveformx

These functions take the binary as keyword arguments (``mass_1``,
``mass_2`` in solar masses, ``spin_1``, ``spin_2``, ``luminosity_distance``
in Mpc, ``inclination``, ``phi_ref``, ``f_ref``; see :class:`Binary`) and
the options of :class:`WaveformOptions` as keywords or ``options=``. They
check the binary as LALSimulation does and raise the exceptions of
:doc:`utilities`.

Models
------

.. autofunction:: approximants
.. autofunction:: is_frequency_domain
.. autofunction:: is_precessing

Frequency domain
----------------

.. autofunction:: polarizations
.. autofunction:: polarizations_on_grid
.. autoclass:: FrequencyGrid
   :members:
.. autofunction:: modes
.. autofunction:: choose_fd_modes
.. autoclass:: FrequencyModes
   :members:

Time domain
-----------

.. autofunction:: td_polarizations
.. autofunction:: td_modes
.. autoclass:: TimeSeriesPolarizations
   :members:
.. autoclass:: TimeSeriesModes
   :members:

Many binaries
-------------

.. autofunction:: batch_polarizations
.. autofunction:: batch_td_polarizations
.. autoclass:: BatchPolarizations
   :members:
.. autoclass:: BatchTimeSeries
   :members:
