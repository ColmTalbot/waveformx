Utilities
=========

.. currentmodule:: waveformx

Data file
---------

.. autofunction:: nrsur7dq4_data_file

Spherical harmonics
-------------------

.. autofunction:: spin_weighted_spherical_harmonic

Exceptions
----------

Every error is a subclass of :class:`WaveformError`, which also derives from
the matching built-in exception, so ``except ValueError`` catches invalid
parameters as it would for most Python code.

.. autoexception:: WaveformError
.. autoexception:: InvalidParameterError
   :show-inheritance:
.. autoexception:: UnsupportedError
   :show-inheritance:
.. autoexception:: UnknownApproximantError
   :show-inheritance:
.. autoexception:: ModelError
   :show-inheritance:
.. autoexception:: DataFileError
   :show-inheritance:
.. autoexception:: DataFileNotFoundError
   :show-inheritance:
