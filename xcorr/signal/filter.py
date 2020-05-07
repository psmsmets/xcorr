r"""

:mod:`signal.filter` -- Filter
==============================

Filter an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
from scipy import signal


# Relative imports
from ..util.history import historicize


__all__ = ['filter']


def filter(
    x: xr.DataArray, frequency, btype: str, order: int = 2,
    dim: str = 'lag', inplace: bool = False, **kwargs
):
    """
    Butterworth filter an N-D labeled array of data.

    Implementation of :func:`scipy.signal.butter` and
    :func:`scipy.signal.sosfiltfilt` to a :class:`xarray.DataArray`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to be filtered.

    frequency : `float` or `tuple`
        The corner frequency (pair) of the filter, in Hz.

    btype : `str` {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
       The type of filter. Default is ‘lowpass’.

    order : `int`, optional
        The order of the filter. Default is 2.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is 'lag'.

    inplace : `bool`, optional
        If `True`, filter in place and avoid a copy. Default is `False`.

    Returns
    -------
    y : :class:`xarray.DataArray` or `None`
        The windowed output of ``x`` if ``inplace`` is `False`.
    """
    assert dim in x.dims, (
        'x has no dimension "{}"!'.format(dim)
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension "{}" has no attribute "sampling_rate"!'.format(dim)
    )
    assert (
        isinstance(frequency, float) or
        (isinstance(frequency, tuple) and len(frequency) == 2)
    ), 'Corner frequency should be a `float` or tuple-pair with (min, max)!'

    y = x if inplace else x.copy()

    sos = signal.butter(
        N=order,
        Wn=frequency,
        btype=btype,
        output='sos',
        fs=y[dim].attrs['sampling_rate']
    )

    y.data = signal.sosfiltfilt(
        sos, y.data, axis=y.dims.index(dim)
    ).astype(y.dtype)

    historicize(y, f='filter', a={
        'x': x.name,
        'frequency': frequency,
        'btype': btype,
        'order': order,
        'dim': dim,
        'inplace': inplace,
        '**kwargs': kwargs,
    })

    return None if inplace else y
