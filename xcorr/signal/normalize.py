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
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='parallelized', output_dtypes=[x.dtype])

    # apply ufunc (and optional dask distributed)
    n = xr.apply_ufunc(np.linalg.norm, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': -1, **kwargs})

    # normalize
    y = x/n

    # propagat attributes
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='norm', a={
        'x': x.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def norm2d(x: xr.DataArray, dim: tuple = None, **kwargs):
    """
    Matrix norm (two-dimensional) of an N-D labelled array of data.

    Implementation of :func:`np.linalg.norm` to a
    :class:`xarray.DataArray` using :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be detrended.

    dim : `tuple`, optional
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
    dim = dim or x.dims[-2:]
    if not isinstance(dim, tuple) or len(dim) != 2:
        raise TypeError('dim should be a tuple of length 2')

    if dim[0] not in x.dims or dim[1] not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='parallelized', output_dtypes=[x.dtype])

    # apply ufunc to get norms (and optional dask distributed)
    n = xr.apply_ufunc(np.linalg.norm, x,
                       input_core_dims=[dim],
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': (-2, -1), **kwargs})

    # normalize
    y = x/n

    # propagat attributes
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='norm2d', a={
        'x': x.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y