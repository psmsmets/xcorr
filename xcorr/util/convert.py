r"""

:mod:`util.convert` -- Xarray conversions
=========================================

Utilities for ``xarray`` to ``ObsPy`` conversions.

"""


# Mandatory imports
from xarray import DataArray
from obspy import Trace, Stream
import numpy as np


# Relative imports
from ..util.time import to_UTCDateTime


__all__ = ['to_trace', 'to_stream']


def to_trace(da: DataArray, **kwargs):
    """
    Convert a one-dimensional dataarray to a trace object

    Parameters
    ----------
    da : :class:`xarray.DataArray`
        One-dimensional N-D labelled data array.

    Returns
    -------
    tr : :class:`obspy.Trace`
        Obspy trace object.

    """
    assert isinstance(da, DataArray), 'da should be an xarray.DataArray.'
    assert len(da.dims) == 1, 'da should be a one-dimensional data array.'

    dim = 'time'
    assert dim in da.dims, f'da has no dimension {dim}'

    assert 'window_length' in da[dim].attrs, (
        'da time coord has no attribute "window_length"'
    )

    assert 'window_overlap' in da[dim].attrs, (
        'da time coord has no attribute "window_overlap"'
    )

    dt = da[dim].attrs['window_length'] * (
        1 - da[dim].attrs['window_overlap']
    )

    # trace header
    header = {
        'network': 'xr',
        'station': '',
        'location': '',
        'channel': da.name,
        'starttime': to_UTCDateTime(da[dim][0].values),
        'delta': dt,
        **kwargs
    }

    # resample to have a continuous time interval
    da = da.resample(time=f'{dt}S').nearest(tolerance=f'{dt/2}S')

    # get masked array
    data = da.to_masked_array()
    np.ma.set_fill_value(data, -1.)

    return Trace(data=data, header=header)


def to_stream(da: DataArray):
    """
    Convert a two-dimensional dataarray to a stream object

    Parameters
    ----------
    da : :class:`xarray.DataArray`
        Two-dimensional N-D labelled data array.

    Returns
    -------
    st : :class:`obspy.Stream`
        Obspy stream object.

    """
    assert isinstance(da, DataArray), 'da should be an xarray.DataArray.'
    assert len(da.dims) == 2, 'da should be a two-dimensional data array.'

    st = Stream()

    dim = da.dims[-2]

    for i, d in enumerate(da[dim]):

        obj = da.loc[{dim: d}]

        st += to_trace(obj, **{'station': f'{dim}_{i}', dim: d.values})

    return st
