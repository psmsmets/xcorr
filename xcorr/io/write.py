r"""

:mod:`io.write` -- Write
========================

Write an xcorr N-D labelled set of data arrays to a netCDF4 file.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import os
import shutil
try:
    import h5netcdf
except ModuleNotFoundError:
    h5netcdf = False

# Relative imports
from .. import util
from .utils import (preprocess_operations_to_dict,
                    preprocess_operations_to_json)


__all__ = ['write']


def write(
    data, path: str, close: bool = True,
    force_write: bool = False, engine: str = None,
    variable_encoding: dict = None,
    hash_data: bool = True, verb: int = 1, **kwargs
):
    """
    Write an xcorr N-D labelled set of data arrays to a netCDF4 file using a
    temporary file and replacing the final destination.

    Before writing the data, metadata and data hash hashes are verified and
    updated if necessary. This changes the data attributes in place.

    The preferred engine is set to h5netcdf if the module can be found.

    Parameters
    ----------
    data : :class:`xarray.Dataset` or :class:`xarray.DataArray`
        The `xcorr` N-D labelled data array or set of arrays.

    path : `str`
        The netCDF4 filename.

    close : `bool`, optional
        Close the data `True` (default).

    force_write : `bool`, optional
        Always write file if `True` even if its empty. Default is `False`.

    engine : `str`, optional
        Set the xarray engine to read the file. Defaults to h5netcdf if the
        module is found instead of netcdf4.

    variable_encoding : `dict`, optional
        Set the same encoding for all non-coordinate variables.

    hash_data : `bool`, optional
        Hash the data of each variable in the dataset or dataarray.
        Defaults to `True`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    Any additional keyword arguments will be passed to
    :func:`xarray.to_netcdf`.

    """
    # check
    isdataset = isinstance(data, xr.Dataset)

    if not (isdataset or isinstance(data, xr.DataArray)):
        raise TypeError('data should be an xarray DataArray or Dataset.')

    # set the engine
    engine = engine or ('h5netcdf' if h5netcdf else None)

    # metadata hash
    metadata_hash = util.hasher.hash(data, metadata_only=True, debug=verb > 3)

    if 'sha256_hash_metadata' not in data.attrs:
        data.attrs['sha256_hash_metadata'] = metadata_hash

    if metadata_hash != data.attrs['sha256_hash_metadata']:
        data.attrs['sha256_hash_metadata'] = metadata_hash
        if verb > -1:
            warnings.warn(
                'Data metadata sha256 hash is updated.',
                UserWarning
            )

    # check status?
    if isdataset and 'status' in data.variables:
        if (
            not force_write and not np.any(data.status.data == 1)
        ):
            warnings.warn(
                'Dataset contains no data. No need to save it.',
                UserWarning
            )
            return

    # start verbose
    if verb > 0:
        print('Write data as "{}"'.format(path), end=': ')

    # folders and tmp name
    abspath, file = os.path.split(os.path.abspath(path))

    if not os.path.exists(abspath):
        os.makedirs(abspath)

    tmp = os.path.join(
        abspath,
        '{f}.{t}'.format(f=file, t=int(pd.to_datetime('now').timestamp()*1e3))
    )

    # close data
    if close:
        if verb > 0:
            print('Close', end='. ')
        data.close()

    # calculate data hash
    if hash_data:
        if verb > 0:
            print('Hash', end='. ')
        data.attrs['sha256_hash'] = util.hasher.hash(data, metadata_only=False,
                                                     debug=verb > 3)
    else:
        if 'sha256_hash' in data.attrs:
            del data.attrs['sha256_hash']

    # convert preprocess operations
    if verb > 1:
        print('Operations to json.', end='. ')

    if 'pair' in data.dims:
        preprocess_operations_to_json(data.pair)

    # update encoding on all variables
    if isinstance(variable_encoding, dict):
        if isinstance(data, xr.Dataset):
            for var in data.variables:
                if var in data.dims:
                    continue
                data[var].encoding = variable_encoding
        else:
            data.encoding = variable_encoding

    # write to temporary file
    if verb > 0:
        print('To temporary netcdf', end='. ')
    data.to_netcdf(
        path=tmp, mode='w', format='netcdf4',
        **{'engine': engine, **kwargs},
    )

    # Replace file
    if verb > 0:
        print('Replace', end='. ')
    shutil.move(tmp, os.path.join(abspath, file))

    # convert preprocess operations
    if verb > 1:
        print('Operations to dict.', end='. ')

    if 'pair' in data.dims:
        preprocess_operations_to_dict(data.pair)

    # final message
    if verb > 0:
        print('Done.')

    return
