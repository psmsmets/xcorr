r"""

:mod:`io.validate` -- Validate
==============================

Validate an xcorr N-D labelled set of data arrays.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import os
from glob import glob
try:
    import h5netcdf
except ModuleNotFoundError:
    h5netcdf = False
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from .. import util
from .utils import preprocess_operations_to_dict


__all__ = ['validate', 'validate_list']


def validate(
    dataset: xr.Dataset, fast: bool = True, quick_and_dirty: bool = False,
    metadata_hash: str = None, preprocess_hash: str = None,
    xcorr_version: str = None, timedelta_to_float_seconds: bool = True,
    verb: int = 0
):
    """
    Validate an xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays to be validated.

    fast : `bool`, optional
        Omit verifying the `sha256_hash` if `True`. Default is `False`.

    quick_and_dirty : `bool`, optional
        Omit verifying both the `sha256_hash` and `sha256_hash_metadata`
        if `True`. Default is `False`.

    metadata_hash : `str`, optional
        Provide a template metadata sha256 hash to filter data arrays.
        A mismatch in ``dataset.sha256_hash_metadata`` will return `None`.
        Defaults to `None`.

    preprocess_hash : `str`, optional
        Provide a template pair preprocess sha256 hash to filter data arrays.
        A mismatch in ``dataset.pair.attrs['preprocess']['sha256_hash']`` will
        return `None`.
        Defaults to `None`.

    xcorr_version : `str`, optional
        Provide a template xcorr version number to filter data arrays.
        A mismatch in ``dataset.xcorr_version`` will return `None`.
        Defaults to `None`.

    timedelta_to_float_seconds : `bool`, optional
        Convert variable of dtype timedelta64 to float seconds if `True`.
        Default is `True`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    validated : :class:`xarray.Dataset`
        The validated `xcorr` N-D labelled set of data arrays.

    """

    # cannot be empty
    if not isinstance(dataset, xr.Dataset):
        return None

    # check existance of main attributes
    if (
        'xcorr_version' not in dataset.attrs or
        'sha256_hash_metadata' not in dataset.attrs
    ):
        dataset.close()
        return None

    # force fast?
    if 'sha256_hash' not in dataset.attrs:
        fast = True

    # dataset fixes single-elements represented as np.arrays (h5netcdf)
    for var in dataset.variables:
        for attr in dataset[var].attrs.keys():
            if (
                isinstance(dataset[var].attrs[attr], np.ndarray) and
                len(dataset[var].attrs[attr]) == 1
            ):
                dataset[var].attrs[attr] = dataset[var].attrs[attr].item()

    # extract source
    src = (dataset.encoding['source'] if 'source' in dataset.encoding
           else '[memory]')

    # Verify metadata_hash input
    if metadata_hash:
        if not isinstance(metadata_hash, str):
            dataset.close()
            raise TypeError('``metadata_hash`` should be a string.')
        if not len(metadata_hash) == 64:
            dataset.close()
            raise ValueError('``metadata_hash`` should be of length 64.')

    # check if at least pair and time variables exist
    if not ('pair' in dataset.coords and 'time' in dataset.coords):
        if verb > 0:
            warnings.warn('Dataset contains no pair and time coordinate.',
                          UserWarning)
        dataset.close()
        return None

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)

    # calculate some hashes
    if not quick_and_dirty:

        sha256_hash_metadata = util.hasher.hash_Dataset(
            dataset, metadata_only=True, debug=verb > 3
        )

        if sha256_hash_metadata != dataset.sha256_hash_metadata:
            if verb > 0:
                warnings.warn(
                    f'Dataset metadata sha256 hash in {src} is inconsistent.',
                    UserWarning
                )
            if verb > 1:
                print('source :', src)
                print(
                    'sha256 hash metadata in ncfile :',
                    dataset.sha256_hash_metadata
                )
                print(
                    'sha256 hash metadata computed  :',
                    sha256_hash_metadata
                )
            dataset.close()
            return None

    if not (quick_and_dirty or fast):
        sha256_hash = util.hasher.hash_Dataset(
            dataset, metadata_only=False, debug=verb > 3
        )
        if sha256_hash != dataset.sha256_hash:
            if verb > 0:
                warnings.warn(f'Dataset sha256 hash in {src} is inconsistent.',
                              UserWarning)
            if verb > 1:
                print('source :', src)
                print('sha256 hash in ncfile :', dataset.sha256_hash)
                print('sha256 hash computed  :', sha256_hash)
            dataset.close()
            return None

    if not quick_and_dirty:
        # compare metadata_hash with template
        if metadata_hash and dataset.sha256_hash_metadata != metadata_hash:
            if verb > 0:
                warnings.warn('Dataset metadata hash does not match.',
                              UserWarning)
            dataset.close()
            return None

        # compare preprocess_hash with template
        if (
            preprocess_hash and
            dataset.pair.attrs['preprocess']['sha256_hash'] != preprocess_hash
        ):
            if verb > 0:
                warnings.warn('Dataset preprocess hash does not match.',
                              UserWarning)
            dataset.close()
            return None

        # compare xcorr_version with template
        if xcorr_version and dataset.xcorr_version != xcorr_version:
            if verb > 0:
                warnings.warn('Dataset xcorr version does not match.',
                              UserWarning)
            dataset.close()
            return None

    # pair as utf-8
    try:
        dataset = dataset.assign_coords({
            'pair': dataset.pair.str.decode('utf-8')
        })
    except AttributeError:
        pass

    # timedelta64 to float seconds
    if timedelta_to_float_seconds:
        for var in dataset.variables:
            if dataset[var].dtype == np.dtype('timedelta64[ns]'):
                dataset[var] = dataset[var] / pd.to_timedelta('1s')
                dataset[var].attrs['units'] = 's'

    return dataset


def validate_list(
    datasets, strict: bool = False, paths_only: bool = False,
    keep_opened: bool = False, engine: str = None,
    parallel: bool = True, compute: bool = True, compute_args: dict = {},
    verb: int = 0, **kwargs
):
    """
    Validate a list of xcorr N-D labelled datasets.

    Parameters
    ----------
    datasets : `str` or `list`
        A glob string, or a list of either glob strings or a
        :class:`xr.Dataset` containing the `xcorr` N-D labelled data arrays.

    strict : `bool`, optional
        If `True`, do not merge data arrays with different `xcorr` versions.
        Defaults to `False`.

    paths_only : `bool`, optional
        If `True`, ``datasets`` can only be glob strings. Defaults to `False`.

    keep_opened : `bool`, optional
        If `True`, do not close the file after opening. Defaults to `False`.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    parallel : `bool`, optional
        Enabled parallellization if `True` (default). Requires Dask.

    compute : `bool`, optional
        Execute Dask compute if `True` (default) and ``parallel`` is enabled.
        Otherwise the delayed object is returned.

    compute_args : `dict`, optional
        Provide a dictionary with arguments for Dask compute.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to :func:`validate`.

    Returns
    -------
    validated : :class:`xarray.Dataset`
        The validated list of `xcorr` N-D labelled sets of data arrays.
        If ``datasets`` was a glob string the expanded file list is returned.

    """

    valErr = ('Datasets should either be a list of paths or xarray.Dataset '
              '(when paths_only is False) but not both.')

    if isinstance(datasets, str):
        datasets = [datasets]

    if not isinstance(datasets, list):
        raise TypeError('datasets should be either a string or a list.')

    parallel = dask and parallel

    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # expand path list with glob
    sources = []

    # expand glob strings and check for unique type in list
    for source in datasets:
        if isinstance(source, str):
            sources += glob(source)
        elif isinstance(source, xr.Dataset):
            if len(sources) > 0 or paths_only:
                raise ValueError(valErr)
        else:
            raise ValueError(valErr)

    isFile = len(sources) > 0
    sources = sorted(sources) if isFile else datasets
    if verb > 3:
        print('validate sources:', sources)
    validated = []

    # get dataset wrapper
    def get_dataset(source):
        if not isFile:
            return source
        if not os.path.isfile(source):
            if verb > 0:
                warnings.warn(f'Datasets item "{source}" does not exists.',
                              UserWarning)
            return None
        return xr.open_dataset(source, engine=engine)

    # get output wrapper
    def get_output(ds):
        if not isinstance(ds, xr.Dataset):
            return None
        if isFile and not keep_opened:
            ds.close()
            return ds.encoding['source']
        else:
            return ds

    # find first validated dataset
    for i, source in enumerate(sources):
        ds = get_dataset(source)
        if ds is None:
            continue
        ds = validate(ds, verb=verb-4, **kwargs)
        if ds is not None:
            break

    # any valid dataset?
    if ds is None:
        raise RuntimeError('No valid dataset found.')

    # append first valid ds
    validated.append(get_output(ds))

    # set validate args based on first valid ds
    validate_args = {
        'metadata_hash': ds.attrs['sha256_hash_metadata'],
        'preprocess_hash': ds.pair.attrs['preprocess']['sha256_hash'],
        'xcorr_version': ds.attrs['xcorr_version'] if strict else None,
    }
    if verb > 3:
        print('validate_args:', validate_args)

    # evaluate
    if parallel:
        for source in sources[i+1:]:
            ds = dask.delayed(get_dataset)(source)
            ds = dask.delayed(validate)(ds, **validate_args, **kwargs)
            validated.append(dask.delayed(get_output)(ds))
        if compute:
            validated = dask.compute(validated, **compute_args)[0]
            validated = [ds for ds in validated if ds is not None]
    else:
        for source in sources[i+1:]:
            ds = get_dataset(source)
            ds = validate(ds, verb=verb-1, **validate_args, **kwargs)
            if ds is not None:
                validated.append(get_output(ds))

    return validated
