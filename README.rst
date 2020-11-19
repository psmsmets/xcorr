*************************************
xcorr - xarray contained correlations
*************************************


**xcorr** is an open-source project existing of tools to cross-correlate
waveform timeseries (`obspy.Stream` objects) contained in a self-describing
N-D labelled `xarray.Dataset` following CF-1.9 Conventions and FAIR data
guidelines. Data and metadata are stored as a `NetCDF Dataset` to be easily
read / postprocessed by other packages, using different languages or on other 
platforms.

**xcorr** exists of various submodules to create a simple workflow from fetching / 
archiving / preprocessing raw waveforms to cross-correlation functions followed by 
various postprocessing and analysis tools. 


Features
========

Main `xcorr` features listed per submodule:

- **core**: Main functions of xcorr are ``init``, ``process``, ``bias_correct``,
  ``read``, ``write`` and ``merge``. All of these return or are applied on an
  `xarray.Dataset`.
  Batch process cross-correlations by ``lazy_process``: multithreading and lazy
  scheduling using `dask <https://dask.org>`_ after first verifying both data
  availability and preprocessing.
  ``mread`` is a multi-file merge and read combining the `xcorr.read` with
  `xarray.open_mfdataset` including **Dask** features like parallelization
  and memory chunking.

- **client**: A waterfall-based client wrapping various getters from both 
  local archives and remote services. Load waveform data from a local SeisComP 
  Data Structure (SDS) archive and automatically download missing or incomplete 
  waveforms by the FDSN web service and the `CTBTO <https://www.ctbto.org>`_ 
  Verification Data Messaging System (VDMS, via `pyvdms` wrapping the command 
  line client).

- **preprocess**: Generalized waveform preprocessing using `obspy` given a dictionary
  with operations per channel id. Channel operations and parameters are added to the
  `pair` as attribute containing id as key.

- **cc**: Crosscorrelation functions and constructors for `lag` time and
  unbiased `weight` vectors.

- **signal**: Postprocess cross-correlation estimates, or any `xarray.DataArray`
  with a (lag) time dimension. Each signal routine is **Dask** capable for
  large datasets.

  - ``beamform``: Least-squares plane wave beamforming.
  - ``correlate``: frequency domain correlation estimator with zero-padding 1d or 2d data.
  - ``detrend``: demean and linear detrend.
  - ``fft``: apply forward and inverse fourier transform along a single-dimension.
  - ``filter``: time-domain filter using a forward-backward (zero-phase) digital
    butterworth filter by cascaded second-order sections.
  - ``mask``: mask coordinates of interest (one or two-dimensional).
  - ``normalize``: broadcast normalize a vector or 2d matrix.
  - ``rms``: root-mean-square.
  - ``snr``: signal-to-noise ratio.
  - ``spectrogram``: compute the power spectrogram (density or spectrum).
    year-day, year-month. 
  - ``taper``: apply a ``window`` to a dimension.
  - ``timeshift``: apply a ``timeshift`` using fft to a dimension.
  - ``trigger``: perform a coincidence trigger on a precomputed characteristic
    function, for example, the signal-to-noise ratio.
  - ``unbias``: bias correct the correlation estimate.
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

- **scripts**: Core scripts to (re)produce the manuscript results. See entry points.


Entry points
============

Core scripts to (re)produce the manuscript results are implemented as entry points
and can be accessed from the command line:

- ``xcorr-snr``: Signal-to-noise ratio estimation of cross-crorrelationss.
- ``xcorr-ct``: Coincidence triggers of cross-crorrelations signal-to-noise ratios.
- ``xcorr-psd``: Spectrogram estimation of signal-to-noise ratio triggered periods.
- ``xcorr-timelapse`` Two-dimensional cross-correlation of cross-correlation spectrograms.
- ``xcorr-beamform`` Plane wave beamforming of cross-correlation functions.


Installation
============

Create a clone, or copy of the xcorr repository

.. code-block:: console

    git clone https://gitlab.com/psmsmets/xcorr.git

Run ``git pull`` to update the local repository to this master repository.


Install xcorr via ``pip``:

.. code-block:: console

   cd xcorr
   pip install -e .


Required are Python version 3.6 or higher and the modules `NumPy`, `SciPy`,
`ObsPy`, `Pandas`, and `Xarray`.

Create a Python 3.8 environment in conda as follows:

.. code-block:: console

    conda env create -f environment.yml


Acknowledgements
================

If you publish results for which you used xcorr, please give credit by citing
`Smets et al. (2020)  <#>`_:

    Smets, P. S. M., ... (2020),
    Long-range hydroacoustic observations of the Monowai Volcanic Centre:
    a proxy for variations in deep-ocean temperature,
    Journal (number), pages, DOI: `doi <#>`_.

All stable releases have a Zenodo-DOI, which can be found on `Zenodo-DOI <#>`_.


Contributing
============

Only accepts pull requests that fixes bugs / fixes typos / improves existing content.


License
=======

Copyright 2020 Pieter Smets.

Licensed under the GNU GPLv3 License. See the ``LICENSE``- and ``NOTICE``-files
or the documentation for more information.
