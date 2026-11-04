Installation
============

From PyPI, once released, or from a clone of the repository::

    pip install waveformx
    pip install .

waveformx needs Python 3.11 or later, JAX, NumPy and h5py. The optional
extras are

* ``mlx``: the MLX versions of the models (:doc:`mlx`), on Apple silicon;
* ``test``: pytest, for the tests;
* ``compare``: LALSimulation (``lalsuite``) and waveformrs, which the tests
  compare with (without them those comparisons are skipped);
* ``docs``: Sphinx, for this documentation.

For example::

    pip install "waveformx[mlx]"

NRSur7dq4 needs its data file, ``NRSur7dq4_v1.0.h5``, from LALSuite's data
(also on `Zenodo <https://zenodo.org/records/14999310>`_). Set
``NRSUR7DQ4_DATA`` to the file, or ``LAL_DATA_PATH`` to the directory that
holds it; :func:`waveformx.nrsur7dq4_data_file` says which file is found.

Importing waveformx enables 64-bit floats in JAX (``jax_enable_x64``),
which the models need.
