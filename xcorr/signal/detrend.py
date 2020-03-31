r"""

:mod:`signal.detrend` -- Detrend
================================

Detrend an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
from scipy import signal


# Relative imports
from ..util.history import historicize


__all__ = ['detrend', 'demean']


def detrend(
    x: xr.DataArray, type: str = 'constant', bp=0, dim: str = 'lag',
    inplace: bool = True
):
    r"""Detrend an N-D labeled array of data.

    Implementation of :func:`scipy.signal.detrend` to a
    :class:`xarray.DataArray`.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be detrended.

    type : `str` {‘constant’, ‘linear’}, optional
       The type of detrending. If type == 'constant' (default), only the
       mean of ``x`` is subtracted. If type == 'linear', the result of a
       linear least-squares fit to ``x`` is subtracted from ``x``.

    bp : `array_like of ints`, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of ``dim`` between two break points. Break
        points are specified as indices of ``x`` into ``dim``.

    dim : `str`, optional
        The coordinates name of ``x`` to be detrended over. Default is 'lag'.

    inplace : `bool`, optional
        If `True` (default), detrend in place and avoid a copy.

    Returns:
    --------
    y : `None` or :class:`xarray.DataArray`
        The detrended output of ``x`` if ``inplace`` is `False`.
    """
    assert dim in x.dims, 'Dimension not found!'

    y = x if inplace else x.copy()
    y.data = signal.detrend(
        x=y.data,
        axis=y.dims.index(dim),
        type='linear',
        bp=0
    )

    historicize(y, f='detrend', a={
        'x': x.name,
        'type': type,
        'bp': bp,
        'dim': dim,
        'inplace': inplace,
    })

    return None if inplace else y


def demean(x: xr.DataArray, dim: str = 'lag', inplace: bool = True):
    r"""Demean  an N-D labeled array of data.

    Wrapper function for :func:`xcorr.signal.detrend` with arguments
    ``type``='constant' and ``bp``=0.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be detrended.

    dim : `str`, optional
        The coordinates name of ``x`` to be demeaned over. Default is 'lag'.

    inplace : `bool`, optional
        If `True` (default), demean in place and avoid a copy.

    Returns:
    --------
    y : `None` or :class:`xarray.DataArray`
        The demeaned output of ``x`` if ``inplace`` is `False`.
    """
    assert dim in x.dims, 'Dimension not found!'
    return detrend(x, type='constant', dim=dim, inplace=inplace)
