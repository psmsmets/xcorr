r"""

:mod:`io.read` -- Read
======================

Read an xcorr N-D labelled set of data arrays from a netCDF4 file.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import os
try:
    import h5netcdf
except ModuleNotFoundError:
    h5netcdf = False
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..io.validate import validate, validate_list


__all__ = ['read', 'mfread']


def read(
    path: str, extract: bool = False, engine: str = None, verb: int = 0,
    **kwargs
):
    """
    Read an xcorr N-D labelled set of data arrays from a netCDF4 file.

    Parameters
    ----------
    path : `str`
        NetCDF4 filename.

    extract : `bool`, optional
        Mask cross-correlation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to the :func:`validate`.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays read from netCDF4.

    """
    # open if exists
    if not os.path.isfile(path):
        return None

    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # open and validate
    dataset = validate(xr.open_dataset(path, engine=engine),
                       verb=verb, **kwargs)

    # early return?
    if dataset is None:
        return

    # verbose status
    if verb > 0:
        src = (dataset.encoding['source'] if 'source' in dataset.encoding
               else '[memory]')
        print('{s} #(status==1): {n} of {m}'.format(
            s=src,
            n=np.sum(dataset.status.data == 1),
            m=dataset.time.size,
        ))

    # extract valid data
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    return dataset


def mfread(
    paths, extract: bool = False, preprocess: callable = None,
    engine: str = None, parallel: bool = True, chunks=None,
    naive: bool = False, verb: int = 0, **kwargs
):
    """
    Read multiple xcorr N-D labelled files as a single dataset using
    :func:`xarray.open_mfdataset`.

    Parameters
    ----------
    paths : `str` or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open. Paths can be given as strings or as pathlib
        Paths.

    extract : `bool`, optional
        Mask cross-correlation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    preprocess : `callable`, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding['source']``.
        If `None` (default) the :func:`validate` is used
        ``quick_and_dirty=True``.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    parallel : `bool`, optional
        Enabled parallellization if `True` (defaults). Requires Dask.

    chunks : `int` or `dict` , optional
        Dictionary with keys given by dimension names and values given by
        chunk sizes. In general, these should divide the dimensions of each
        dataset. If int, chunk each dimension by chunks. By default, chunks
        will be chosen to load entire input files into memory at once. This
        has a major impact on performance: see :func:`xarray.open_mfdataset`
        for more details.

    naive : `bool`, optional
        If `True`, ``paths`` is directly passed on to
        :func:`xarray.open_mfdataset` disabling :func:`validate_list`.
        Defaults to `False`.

    Any additional keyword arguments will be passed to the :func:`validate`.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays read from netCDF4.

    """
    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # get a list of validated datasets
    if naive:
        validated = paths
    else:
        validated = validate_list(paths, keep_opened=False, paths_only=True,
                                  engine=engine, verb=verb, **kwargs)
        if verb > 3:
            print('validated :', validated)

    # validate wrapper to pass arguments
    def _validate(ds):
        return validate(ds, quick_and_dirty=True)

    # open multiple files using dask
    dataset = xr.open_mfdataset(
        paths=validated,
        chunks=chunks,
        combine='by_coords',
        preprocess=preprocess or _validate,
        engine=engine,
        lock=False,
        data_vars='minimal',
        parallel=parallel,
        join='outer',
    )

    # extract valid data
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    # update some global attrs
    if 'sha256_hash' in dataset.attrs:
        del dataset.attrs['sha256_hash']

    strt0 = pd.to_datetime(dataset.time.values[0]).strftime('%Y.%j')
    strt1 = pd.to_datetime(dataset.time.values[-1]).strftime('%Y.%j')
    dataset.attrs['title'] = (
        (dataset.attrs['title']).split(' - ')[0] +
        ' - ' +
        strt0 +
        ' to {}'.format(strt1) if strt0 != strt1 else ''
    ).strip()

    dataset.attrs['history'] = 'Merged @ {}'.format(pd.to_datetime('now'))

    # hashes?
    if naive:
        dataset.attrs['sha_hash'] = None
        dataset.attrs['sha256_hash_metadata'] = None

    return dataset
