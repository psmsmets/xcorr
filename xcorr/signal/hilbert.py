r"""

:mod:`signal.hilbert` -- Hilbert
================================

Hilbert transform of an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
from scipy import signal
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['hilbert']


def hilbert(
    x: xr.DataArray, dim: str = None, **kwargs
):
    """
    Compute the analytic signal, using the Hilbert transform, of an N-D
    labeled array of data.

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
        :func:`signal.hilbert`.

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
    y = xr.apply_ufunc(signal.hilbert, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={'axis': -1, **kwargs})

    # log workflow
    historicize(y, f='hilbert', a={
        'x': x.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
