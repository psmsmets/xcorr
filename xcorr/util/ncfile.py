r"""

:mod:`util.path` -- Path Utilities
==================================

Utilities for ``xcorr`` for file naming.

"""


# Mandatory imports
import pandas as pd
import xarray as xr
import os


# Relative imports
from ..util.receiver import check_receiver, split_pair, receiver_to_str


__all__ = ['ncfile']


_pair_type_error = ('``pair`` should be a string or tuple with the '
                    'receiver couple SEED-ids.')


def ncfile(pair, time: pd.Timestamp, root: str = None,
           verify_receiver: bool = True, **kwargs):
    r"""Return the netCDF filename and path.

    Parameters
    ----------
    pair : `tuple` or `str`
        Receiver pair couple as a string or as `tuple` of length 2 specifying
        the receiver SEED-ids as either `str` or `dict`.

    time : `pd.Timestamp`
        Time of the output file.

    root : `str`, optional

    verify_receiver : `bool`, optional
        If `True` (default), verify each receiver for valid SEED-id naming
        without wildcards.

    Returns
    -------
    path : `str`
        Path to the ncfile.

    """
    # check pair
    if isinstance(pair, xr.DataArray):
        if pair.size == 1:
            pair = str(pair.values)
        else:
            raise ValueError('pair should be a single element')
    if verify_receiver:
        if isinstance(pair, str):
            pair = split_pair(pair, to_dict=False)
        elif isinstance(pair, tuple) and len(pair) == 2:
            pair = list(pair)
        else:
            raise TypeError(_pair_type_error)
        for i, rec in enumerate(pair):
            if isinstance(rec, dict):
                pair[i] = receiver_to_str(rec)
            elif not isinstance(rec, str):
                raise TypeError(_pair_type_error)
            if not check_receiver(pair[i], allow_wildcards=False):
                raise TypeError(_pair_type_error)
        pair = '-'.join(pair)

    # check time
    if isinstance(time, xr.DataArray):
        if time.size == 1:
            time = pd.to_datetime(time.values)
        else:
            raise ValueError('time should be a single element')
    if not isinstance(time, pd.Timestamp):
        raise TypeError('``time`` should be of type `pd.Timestamp`!')

    # check root
    if root and not isinstance(root, str):
        raise TypeError('``root`` should be of type `str`!')

    # join
    year = f'{time.year:04d}'
    root = os.path.join(root, year, pair) if root else os.path.join(year, pair)

    # path and filename
    ncfile = os.path.join(root, '{p}.{y:04d}.{d:03d}.nc'.format(
        p=pair, y=time.year, d=time.dayofyear
    ))

    return ncfile
