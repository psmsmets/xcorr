*************************************
xcorr - xarray contained correlations
*************************************

.. image:: https://zenodo.org/badge/450085786.svg
   :target: https://zenodo.org/badge/latestdoi/450085786

**xcorr** provides tools for reusable and reproducible time-series
cross-correlation analysis. Cross-correlation functions of waveform time-series
(``obspy.Stream`` objects) are contained in a self-describing N-D labelled
``xarray.Dataset`` following CF-1.9 Conventions and FAIR data guidelines. Data
and metadata are stored as a ``NetCDF Dataset`` to be easily read and post-processed
by other packages, using different languages or on other platforms.


Features
========

Main `xcorr` features listed per submodule:

- **core**: initiate and process an N-D labelled
  cross-correlation dataset via accessors.
  Batch processing is handled by `Dask <https://dask.org>`_ after verifying both the waveform
  availability and stream preprocessing operations.
  Output files are structered in an SDS-like folder structure
  (year/pair/ncfile) and can easily be read and merged in parallel.

- **stream**: fetch and archive waveforms using the waterfall-based ``Client`` class,
  wrapping various getters from both local archives and remote services.
  Generalized SEED-channel based waveform processing.

- **signal**: signal processing of an `xarray.DataArray` timeseries.
  Each signal routine is **Dask** capable supporting large datasets.
  Functions are available as `.signal` accessor.


See the examples, scripts and documentation for the full functionality.


Entry points
============

Core scripts to (re)produce the manuscript results are implemented as entry points
and can be accessed from the command line:

- ``xcorr-ct``: Coincidence triggers of cross-crorrelations signal-to-noise ratios.
- ``xcorr-beamform`` Plane wave beamforming of cross-correlation functions.
- ``xcorr-psd``: Spectrogram estimation of signal-to-noise ratio triggered periods.
- ``xcorr-snr``: Signal-to-noise ratio estimation of cross-crorrelations.
- ``xcorr-timelapse`` Two-dimensional cross-correlation of cross-correlation spectrograms.


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


Required are Python3.7 or higher and the modules `NumPy`, `SciPy`,
`ObsPy`, `Pandas`, and `Xarray`.
`Dask` is required to run the scripts but needed for the base functionality.

Create a conda environment named ``xcorr`` with Python 3.9 and all required packages:

.. code-block:: console

    conda env create -f environment.yml


Acknowledgements
=====

If you publish results for which you used `xcorr <https://github.com/psmsmets/xcorr>`_
or `xcorr-cookbook-data <https://github.com/psmsmets/xcorr-cookbook-data>`_, 
please give credit by citing the applicable DOI and the manuscript
`Smets et al. (2022)  <https://doi.org/10.1029/2022JC018451>`_ :

    Smets, Weemstra and Evers (2022),
    Hydroacoustic travel time variations as a proxy for passive deep-ocean
    thermometry - a cookbook,
    Journal of Geophysical Research Oceans, DOI: `10.1029/2022JC018451 <https://doi.org/10.1029/2022JC018451>`_.

All stable releases have a dedicated Zenodo-DOI:

- xcorr: `https://doi.org/10.5281/zenodo.5883341 <https://doi.org/10.5281/zenodo.5883341>`_
- xcorr-cookbook-data: `https://doi.org/10.5281/zenodo.5902705 <https://doi.org/10.5281/zenodo.5902705>`_


Contributing
============

Only accepts pull requests that fixes bugs / fixes typos / improves existing content.


License
=======

Copyright 2022 Pieter Smets.

Licensed under the GNU GPLv3 License. See the
`LICENSE <https://github.com/psmsmets/xcorr/blob/master/LICENSE>`_
file or the documentation for more information.
