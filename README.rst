*********************************
xcorr - xarray based correlations
*********************************


**xcorr** is an open-source project containing tools to crosscorrelate
waveform timeseries  (`obspy.Stream` objects) and wrap all results and
metadata as a self-describing N-D labeled `xarray.Dataset` following
CF-1.9 Conventions and FAIR data guidelines. Data can be stored as
`netCDF4` and opened/postprocessed in other languages, packages,
machines or platforms.

**xcorr** contains various modules such as waveform preprocessing, a
waterfall-based client wrapping various getters from local archives as well
as remote services, unbiased frequency-domain-based crosscorrelations and
postprocessing/analysis tools.


Features
========

Main xcorr features listed per submodule:

- **core** - Main functions of xcorr are `init`, `process`. `bias_correct`,
  `read`, `write` and `merge`. All of these return or are applied on an
  `xarray.Dataset`.

- **clients** - Load waveform data from a local SeisComP Data Structure (SDS)
  archive and automatically download missing or incomplete waveforms by the
  FDSN web service and/or NMS Client.

- **preprocess**: process waveforms before correlation given a dictionary with
  operations per channel id. Channel operations and parameters are added to the
  `pair` as attribute containing id as key.

- **cc** - Crosscorrelation functions and constructors for ``lag`` time and
  unbiased ``weight`` vectors.

- **postprocess** - Postprocess correlation data arrays:

  - Utilities to demean the crosscorrelation estimate, extract lag windows of
    interest, calculate the signal-to-noise ratio of the crosscorrelation
    estimate.
  - Time-domain filter an `xarray.DataArray` using a forward-backward
    (zero-phase) digital butterworth filter using cascaded second-order
    sections.
  - Stack crosscorrelation estimates 
  - Spectrogram

- **util** - Utilities:

  - Hash utilities for python `list` and `dict` types, and entire objects such
    as `obspy.Trace`, `obspy.Stream`, `xarray.DataArray`, and `xarray.Dataset`.
  - Time utilities to convert between various datetime types and between
    timedelta and seconds. Functions to get the number of days per month and
    year depending on the calendar type.
  - Receiver utilities such as SEED-id name checks, type conversions, receiver
    pair handlers and geodetic operations.


Installation
============

Create a clone, or copy of the xcorr repository

.. code-block:: console

    git clone https://github.com/psmsmets/xcorr.git

Run ``git pull`` to update the local repository to this master repository.


Install xcorr via ``pip``:

.. code-block:: console

   cd xcorr
   pip install -e .

Required are Python version 3.5 or higher and the modules `NumPy`, `SciPy`,
`ObsPy`, `Pandas`, and `xarray`.
Old versions of `ObsPy` (<1.2.0) and `NumPy` tend to cause problems which
kills the `remove_response` function of `ObsPy`.
Create a working (non-conflicting) Python 3.7 environment in conda as follows:

.. code-block:: console

    conda env create -f xcorr.yml


Citation
========

If you publish results for which you used empymod, please give credit by citing
`Smets (2020)  <#>`_:

    Smets, P. S. M., ... (2020), title: Journal (number), pages; DOI:
    `doi <#>`_.

All releases have a Zenodo-DOI, which can be found on `Zenodo-DOI <#>`_.


License information
===================

Copyright 2020 Pieter Smets.

Licensed under the MIT License. See the ``LICENSE``- and ``NOTICE``-files or
the documentation for more information.
