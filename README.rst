*************************************
xcorr - xarray contained correlations
*************************************


**xcorr** is an open-source project existing of tools to crosscorrelate
waveform timeseries (`obspy.Stream` objects) contained in a self-describing
N-D labeled `xarray.Dataset` following CF-1.9 Conventions and FAIR data
guidelines. Data and metadata can be stored as `netCDF4` and read/postprocessed
in other languages, packages, machines or platforms.

**xcorr** exists of various submodules such as generalized waveform preprocessing,
a waterfall-based client wrapping various getters from both local archives and
remote services, an (unbiased) frequency-domain crosscorrelations algorithm and
various postprocessing/analysis tools.


Features
========

Main xcorr features listed per submodule:

- **core**: Main functions of xcorr are `init`, `process`. `bias_correct`,
  `read`, `write` and `merge`. All of these return or are applied on an
  `xarray.Dataset`.

- **clients**: Load waveform data from a local SeisComP Data Structure (SDS)
  archive and automatically download missing or incomplete waveforms by the
  FDSN web service and/or NMS Client.

- **preprocess**: Process waveforms before correlation given a dictionary with
  operations per channel id. Channel operations and parameters are added to the
  `pair` as attribute containing id as key.

- **cc**: Crosscorrelation functions and constructors for `lag` time and
  unbiased `weight` vectors.

- **signal**: Postprocess crosscorrelation estimates, or any `xarray.DataArray`
  with a (lag) time dimension.

  - ``detrend``: demean and linear detrend.
  - ``mask``: mask coordinates of interest (one or two-dimensional).
  - ``filter``: time-domain filter using a forward-backward (zero-phase) digital
    butterworth filter by cascaded second-order sections.
  - ``psd``: spectrogram.
  - ``snr``: signal-to-noise ratio.
  - ``rms``: root-mean-square.
  - ``stack``: stack over a full time period, or grouped per day, month, year,
    year-day, year-month. 
  - ``taper``: apply a ``window`` to a dimension.
  - ``window``: construct a taper window for a coordinate using any window of
    `scipy.signal.windows <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_ 
    in an `obspy.Trace.taper <https://docs.obspy.org/master/packages/autogen/obspy.core.trace.Trace.taper.html>`_
    like way.

- **util**: Various utilities.

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
