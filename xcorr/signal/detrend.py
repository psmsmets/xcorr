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


def detrend(
    x: xr.DataArray, type: str = 'constant', dim: str = None,
    **kwargs
):
    """
    Detrend an N-D labelled array of data.

    Implementation of :func:`scipy.signal.detrend` to a
    :class:`xarray.DataArray` using :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be detrended.

    type : `str` {‘constant’, ‘linear’}, optional
       The type of detrending. If type == 'constant' (default), only the
       mean of ``x`` is subtracted. If type == 'linear', the result of a
       linear least-squares fit to ``x`` is subtracted from ``x``.

    dim : `str`, optional
        The coordinates name of ``x`` to be detrended over. Defaults to the
        last dimension of ``x``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`xarray.apply_ufunc`.

    Returns
    -------
    y : `None` or :class:`xarray.DataArray`
        The detrended data array.

    """

    # dim
    dim = dim or x.dims[-1]
    assert dim in x.dims, f'x has no dimension "{dim}"!'

    # get index of dim
    axis = x.dims.index(dim)
    axis = -1 if axis == len(x.dims) - 1 else axis

    # detrend wrapper to simplify ufinc input
    def _detrend(obj):
        return signal.detrend(
            data=obj.data,
            axis=axis,
            type=type,
        )

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='parallelized', output_dtypes=[x.dtype])

    # apply sosfiltfilt as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_detrend, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # log workflow
    historicize(y, f='detrend', a={
        'x': y.name,
        'type': type,
        'dim': dim,
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
