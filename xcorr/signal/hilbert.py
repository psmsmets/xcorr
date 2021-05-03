r"""

:mod:`signal.hilbert` -- Hilbert
================================

Hilbert transform of an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
import scipy as sp
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize
from .absolute import absolute


__all__ = ['hilbert', 'envelope']


def hilbert(
    x: xr.DataArray, dim: str = None, **kwargs
):
    """
    Compute the analytic signal, using the Hilbert transform, of an N-D
    labelled array of data.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to be transformed.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is the
        last dimension.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`scipy.signal.hilbert`.

    Returns
    -------
    y : :class:`xarray.DataArray` or `None`
        The Hilbert transform of ``x``.

    """
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[np.complex128])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(sp.signal.hilbert, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': -1, **kwargs})

    # restore attributes
    y.name = x.name
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='hilbert', a={
        'x': x.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def envelope(
    x: xr.DataArray, dim: str = None, **kwargs
):
    """
    Compute the amplitude envelope of an N-D labelled array of data.
    The amplitude envelope is the magnitude of the analytic signal, estimated
    by the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to compute the amplitude envelope for.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is the
        last dimension.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`scipy.signal.hilbert`.

    Returns
    -------
    y : :class:`xarray.DataArray` or `None`
        The amplitude envelope of ``x``.

    """
    return absolute(hilbert(x, dim, **kwargs))
