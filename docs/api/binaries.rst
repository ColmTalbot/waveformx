Binaries and options
====================

.. currentmodule:: waveformx

The binary
----------

.. autoclass:: Binary
.. autofunction:: parameter_table
.. autodata:: COLUMNS
   :no-value:

Options
-------

.. autoclass:: WaveformOptions
   :members: lalsimulation_defaults, replace, as_dict

Reference frequency
-------------------

Every model can take the binary at a reference frequency scaled by its
total mass (``reference_mf``, recommended :data:`REFERENCE_MF`) instead of
one in Hz. These functions convert a binary between the two conventions,
for any model (``approximant=``), so that both give the same waveform. They
are pure functions, which can be compiled and differentiated.

.. autodata:: REFERENCE_MF
   :no-value:
.. autofunction:: to_mass_scaled_reference
.. autofunction:: to_physical_reference
