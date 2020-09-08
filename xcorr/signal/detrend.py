r"""

:mod:`signal.detrend` -- Detrend
================================

Detrend an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
from scipy import signal
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['detrend', 'demean']


def detrend(x: xr.DataArray, dim: str = None, **kwargs):
    """
    Detrend an N-D labelled array of data.

    Implementation of :func:`scipy.signal.detrend` to a
    :class:`xarray.DataArray` using :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be detrended.

    dim : `str`, optional
        The coordinates name of ``x`` to be detrended over. Defaults to the
        last dimension of ``x``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`scipy.signal.detrend`.

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

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[x.dtype])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(signal.detrend, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': -1, **kwargs})

    # log workflow
    historicize(y, f='detrend', a={
        'x': y.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def demean(x: xr.DataArray, dim: str = None, **kwargs):
    r"""Demean  an N-D labelled array of data.

    Wrapper function for :func:`xcorr.signal.detrend` with arguments
    ``type``='constant' and ``bp``=0.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be detrended.

    dim : `str`, optional
        The coordinates name of ``x`` to be demeaned over. Defaults to the
        last dimension of ``x``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`xarray.apply_ufunc`.

    Returns:
    --------
    y : `None` or :class:`xarray.DataArray`
        The demeaned array of data.

    """
    return detrend(x, type='constant', dim=dim, **kwargs)
