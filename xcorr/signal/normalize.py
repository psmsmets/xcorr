r"""

:mod:`signal.normalize` -- Normalize
====================================

Normalize an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['norm', 'norm1d', 'norm2d']


def norm(*args, **kwargs):
    """Wrapper to norm1d.
    """
    return norm1d(*args, **kwargs)


def norm1d(x: xr.DataArray, dim: str = None, **kwargs):
    """
    Vector norm of an N-D labelled array of data.

    Implementation of :func:`np.linalg.norm` to a
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
        :func:`np.linalg.norm`.

    Returns
    -------
    y : `None` or :class:`xarray.DataArray`
        The normalized data array.

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
    n = xr.apply_ufunc(np.linalg.norm, x,
                       input_core_dims=[[dim]],
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': -1, **kwargs})

    # normalize
    y = x/n

    # propagate attributes
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='norm', a={
        'x': x.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def norm2d(x: xr.DataArray, dims: tuple = None, **kwargs):
    """
    Matrix norm (two-dimensional) of an N-D labelled array of data.

    Implementation of :func:`np.linalg.norm` to a
    :class:`xarray.DataArray` using :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be detrended.

    dims : `tuple`, optional
        A tuple pair with the coordinates name of ``x``. Defaults to the last
        two dimensions of ``x``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`np.linalg.norm`.

    Returns
    -------
    y : `None` or :class:`xarray.DataArray`
        The normalized data array.

    """

    # dim
    dims = dims or x.dims[-2:]
    if not isinstance(dims, tuple) or len(dims) != 2:
        raise TypeError('dims should be a tuple of length 2')

    if dims[0] not in x.dims or dims[1] not in x.dims:
        raise ValueError(f'x has no dimensions "{dims}"')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[x.dtype])

    # apply ufunc (and optional dask distributed)
    n = xr.apply_ufunc(np.linalg.norm, x,
                       input_core_dims=[dims],
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': (-2, -1), **kwargs})

    # normalize
    y = x/n

    # propagate attributes
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='norm2d', a={
        'x': x.name,
        'dims': dims,
        '**kwargs': kwargs,
    })

    return y
