r"""

:mod:`util.path` -- Path Utilities
==================================

Utilities for ``xcorr`` for file naming.

"""


# Mandatory imports
import pandas as pd
import os


# Relative imports
from ..util.receiver import (check_receiver, split_pair, receiver_to_str)


__all__ = ['ncfile']


_pair_type_error = ('``pair`` should be a string or tuple with the '
                    'receiver couple SEED-ids.')


def ncfile(pair, time: pd.Timestamp, root: str = None):
    r"""Return the netCDF filename and path.

    Parameters
    ----------
    pair : `tuple` or `str`
        Receiver pair couple as a string or as `tuple` of length 2 specifying
        the receiver SEED-ids as either `str` or `dict`.

    time : `pd.Timestamp`
        Time of the output file.

    root : `str`, optional

    Returns
    -------
    path : `str`
        Path to the ncfile.

    """
    # check pair
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

    # check time
    if not isinstance(time, pd.Timestamp):
        raise TypeError('``time`` should be of type `pd.Timestamp`!')

    # check root
    if root and not isinstance(root, str):
        raise TypeError('``root`` should be of type `str`!')

    # join
    pair = '-'.join(pair)
    root = os.path.join(root, pair) if root else pair

    # path and filename
    ncfile = os.path.join(root, '{p}.{y:04d}.{d:03d}.nc'.format(
        p=pair, y=time.year, d=time.dayofyear
    ))

    return ncfile
