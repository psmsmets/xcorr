r"""

:mod:`signal.absolute` -- Absolute
==================================

Hilbert transform of an N-D labeled array of data.

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


__all__ = ['abs', 'absolute']


def absolute(
    x: xr.DataArray, **kwargs
):
    r"""Calculate the absolute value element-wise of an N-D labeled
    array of data.

    ``abs`` is a shorthand for this function.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`np.absolute`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        An N-D labeled array of data containing the absolute value of each
        element in ``x``.  For complex input, ``a + ib``, the absolute value
        is :math:`\sqrt{ a^2 + b^2 }`.
        This is a scalar if ``x`` is a scalar.
    """
    if not isinstance(x, xr.DataArray):
        raise TypeError('x should be an xarray.DataArray')

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[np.complex128])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(np.absolute, x,
                       input_core_dims=[[]],
                       output_core_dims=[[]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={**kwargs})

    # log workflow
    historicize(y, f='absolute', a={
        'x': x.name,
        '**kwargs': kwargs,
    })

    return y


def abs(x: xr.DataArray, **kwargs):
    r"""Calculate the absolute value element-wise.

    ``abs`` is a shorthand for this function.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to calculate the absolute value.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`np.absolute`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        An ndarray containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\sqrt{ a^2 + b^2 }`.
        This is a scalar if `x` is a scalar.
    """
    return absolute(x, **kwargs)
