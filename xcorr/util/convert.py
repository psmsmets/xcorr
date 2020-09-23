r"""

:mod:`util.convert` -- Xarray conversions
=========================================

Utilities for ``xarray`` to ``ObsPy`` conversions.

"""


# Mandatory imports
from xarray import DataArray
from obspy import Trace, Stream
import numpy as np
import pandas as pd


# Relative imports
from ..util.time import to_datetime, to_UTCDateTime


__all__ = ['to_trace', 'to_stream']


def to_trace(x: DataArray, dim: str = None, starttime: np.datetime64 = None,
             name: str = None, **kwargs):
    """
    Convert a one-dimensional dataarray to a trace object

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        One-dimensional N-D labelled data array.

    Returns
    -------
    tr : :class:`obspy.Trace`
        Obspy trace object.

    """
    if not isinstance(x, DataArray):
        raise TypeError('x should be an xarray.DataArray.')

    if len(x.dims) != 1:
        raise ValueError('x should be a one-dimensional data array.')

    dim = dim or x.dims[-1]
    if dim not in x.dims:
        raise ValueError(f'x has no dimension {dim}')

    if x[dim].dtype.type == np.datetime64:
        starttime = to_datetime(x[dim][0])
        x[dim] = x[dim] - x[dim][0]
    else:
        starttime = to_datetime(starttime or 'now')
        x[dim] = x[dim]*pd.Timedelta('1s')

    delta = (x[dim].diff('lag').mean()/pd.Timedelta('1s')).item()

    # trace header
    header = {
        'network': 'xr',
        'station': '',
        'location': '',
        'channel': name or dim,
        'starttime': to_UTCDateTime(starttime),
        'delta': delta,
        **kwargs
    }

    # resample to have a continuous time interval
    x = x.resample({dim: f'{delta}S'}).nearest(tolerance=f'{delta/2}S')

    # get masked array
    data = x.to_masked_array()
    np.ma.set_fill_value(data, -1.)

    return Trace(data=data, header=header)


def to_stream(x: DataArray, dim: str = None, **kwargs):
    """
    Convert a two-dimensional dataarray to a stream object

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        Two-dimensional N-D labelled data array.

    dim : `str`, optional
        Set the trace time dimension. Defaults to the last dimension.

    Returns
    -------
    st : :class:`obspy.Stream`
        Obspy stream object.

    """
    if not isinstance(x, DataArray):
        raise TypeError('x should be an xarray.DataArray.')

    if len(x.dims) != 2:
        raise ValueError('x should be a two-dimensional data array.')

    dim = dim or x.dims[-1]
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    dim0 = x.isel({dim: 0}).dims[0]

    st = Stream()

    for i, d in enumerate(x[dim0]):
        st += to_trace(
            x=x.loc[{dim0: d}],
            **{**kwargs, 'station': f'{dim}_{i}', dim: dim}
        )

    return st
