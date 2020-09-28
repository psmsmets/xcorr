r"""

:mod:`signal.detrend` -- Detrend
================================

Detrend an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['detrend', 'demean']

_types = ('demean', 'linear')


def detrend(x: xr.DataArray, dim: str = None, type: str = None,
            skipna: bool = True):
    """
    Linear detrend an N-D labelled array of data.

    Implementation of :func:`scipy.stats.linregress` to a
    :class:`xarray.DataArray` using :func:`xarray.apply_ufunc`.
    `NaN` is discarged to estimate the linear fit.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be detrended linearly.

    dim : `str`, optional
        The coordinates name of ``x`` to be detrended over. Defaults to the
        last dimension of ``x``.

    type : {'constant', 'linear'}, optional
        The type of detrending. If type == 'constant' (default), only the mean
        of data is subtracted.  If type == 'linear' (default), the result of a
        linear least-squares fit to data is subtracted from data.

    skipna : `bool`, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or skipna=True has not been
        implemented (object, datetime64 or timedelta64).

    Returns
    -------
    y : `None` or :class:`xarray.DataArray`
        The detrended data array.

    """

    # dim
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # type
    type = type or _types[0]
    if not isinstance(type, str):
        raise TypeError('type should be a string')
    if type not in _types:
        raise KeyError(f'type should be any of "{"|".join(_types)}"')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[x.dtype])

    # func
    def linear_detrend(x, y):
        def linear_detrend_axis(y, x):
            not_nan_ind = ~np.isnan(y)
            m, b, r_val, p_val, std_err = stats.linregress(
                x[not_nan_ind] if skipna else x,
                y[not_nan_ind] if skipna else y
            )
            return y - (m*x + b)
        if (x.dtype.type == np.datetime64):
            x = x - x[0]
        if (x.dtype.type == np.timedelta64):
            x = x/pd.Timedelta('1s')
        return np.apply_along_axis(linear_detrend_axis, -1, y, x)

    if type == 'demean':
        y = x - x.mean(dim=dim, skipna=skipna, keep_attrs=True)
        y.attrs = x.attrs
    elif type == 'linear':
        y = xr.apply_ufunc(linear_detrend, x[dim], x,
                           input_core_dims=[[dim], [dim]],
                           output_core_dims=[[dim]],
                           keep_attrs=True,
                           vectorize=False,
                           **dargs)

    # log workflow
    historicize(y, f='detrend', a={
        'x': y.name,
        'dim': dim,
        'type': type,
        'skipna': skipna,
    })

    return y


def demean(x: xr.DataArray, **kwargs):
    r"""Demean  an N-D labelled array of data.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be detrended.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`detrend`.

    Returns:
    --------
    y : `None` or :class:`xarray.DataArray`
        The demeaned array of data.

    """
    return detrend(x, type='constant', **kwargs)
