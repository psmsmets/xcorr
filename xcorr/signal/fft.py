r"""

:mod:`signal.fft` -- FFT
========================

Forward and inverse FFT an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
import json
try:
    from pyfftw.interfaces import numpy_fft as fftlib
except ModuleNotFoundError:
    from numpy import fft as fftlib
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['fft', 'ifft']


_recip_name = '__reciprocal_name__'
_recip_attr = '__reciprocal_attr__'


def fft(
    x: xr.DataArray, dim: str = None,
    new_dim: str = None, new_dim_attrs: dict = None,
    dtype: np.dtype = None, **kwargs
):
    """
    FFT an N-D labelled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to apply the FFT.

    dim : `str`, optional
        Coordinate name of ``x`` to apply the FFT.
        Defaults to the last dimension of ``x``.

    new_dim : `str`, optional
        Coordinate name of ``x`` after applying the FFT.
        Defaults to `freq`.

    new_dim_attrs : `dict`, optional
        Dictionary with coordinate attributes.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the complex dtype of ``x`` is used.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`np.fft.fft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the FFT of ``x``.

    """

    # dim
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    # inverse attributes?
    old_dim, old_dim_attrs = _load_reciprocal_attrs(x[dim])

    # new dim
    new_dim = new_dim or old_dim or 'freq'
    if not isinstance(new_dim, str):
        raise TypeError('new_dim should be a string')
    if new_dim in x.dims:
        raise ValueError(f'x already has a dimension "{new_dim}"')

    new_dim_attrs = new_dim_attrs or old_dim_attrs or dict()
    if not isinstance(new_dim_attrs, dict):
        raise TypeError('new_dim_attrs should be a dictionary')

    # dtype
    dtype = np.dtype(dtype or np.complex_(real=x.dtype.name))
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'complex' not in dtype.name:
        raise TypeError('dtype should be complex.')
    dtype = dtype.name

    # delta and frequency
    delta = np.abs(x[dim][1] - x[dim][0]).values.item()
    freq = fftlib.fftshift(fftlib.fftfreq(x[dim].size, d=delta))

    # set axis
    ax = -1

    # wrapper to simplify ufunc input
    def _fft(f, **kwargs):
        return fftlib.fftshift(fftlib.fft(f, axis=ax, **kwargs), axes=ax)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[dtype])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_fft, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # add new dim
    y = y.assign_coords({
        new_dim: xr.DataArray(
            data=freq,
            dims=(new_dim,),
            name=new_dim,
            attrs={
                **x[dim].attrs,
                'long_name': 'Frequency',
                'standard_name': 'frequency',
                'units': 's-1',
                **new_dim_attrs,
                **_dump_reciprocal_attrs(x[dim]),
            },
        ),
    })

    # log workflow
    historicize(y, f='fft', a={
        'x': x.name,
        'dim': dim,
        'new_dim': new_dim,
        'dtype': dtype,
        '**kwargs': kwargs,
    })

    return y


def ifft(
    x: xr.DataArray, dim: str = None,
    new_dim: str = None, new_dim_attrs: dict = None,
    dtype: np.dtype = None, **kwargs
):
    """
    IFFT an N-D labelled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to apply the inverse FFT.

    dim : `str`, optional
        Coordinate name of ``x`` to apply the FFT.
        Defaults to the last dimension of ``x``.

    new_dim : `str`, optional
        Coordinate name of ``x`` after applying the FFT.
        Defaults to `time`.

    new_dim_attrs : `dict`, optional
        Dictionary with coordinate attributes.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the complex dtype of ``x`` is used.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`np.fft.ifft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the inverse FFT of ``x``.

    """

    # dim
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    # inverse attributes?
    old_dim, old_dim_attrs = _load_reciprocal_attrs(x[dim])

    # new dim
    new_dim = new_dim or old_dim or 'time'
    if not isinstance(new_dim, str):
        raise TypeError('new_dim should be a string')
    if new_dim in x.dims:
        raise ValueError(f'x already has a dimension "{new_dim}"')

    new_dim_attrs = new_dim_attrs or old_dim_attrs or dict()
    if not isinstance(new_dim_attrs, dict):
        raise TypeError('new_dim_attrs should be a dictionary')

    # dtype
    dtype = np.dtype(dtype or np.float_(x=x.dtype.name))
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'float' not in dtype.name:
        raise TypeError('dtype should be float.')
    dtype = dtype.name

    # delta
    delta = (x[dim][1] - x[dim][0]).values.item()
    time = np.linspace(
        start=0.,
        stop=(x[dim].size-1)/np.round(x[dim].size*delta, decimals=5),
        num=x[dim].size,
    )

    # set axis
    ax = -1

    # wrapper to simplify ufunc input
    def _ifft(F, **kwargs):
        return np.real(
            fftlib.ifft(
                fftlib.fftshift(F, axes=ax),
                axis=ax,
                **kwargs
            )
        ).astype(dtype)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[dtype])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_ifft, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # add new dim
    y = y.assign_coords({
        new_dim: xr.DataArray(
            data=time,
            dims=(new_dim,),
            name=new_dim,
            attrs={
                **x[dim].attrs,
                'long_name': 'Time',
                'standard_name': 'time',
                'units': 's',
                **new_dim_attrs,
                **_dump_reciprocal_attrs(x[dim]),
            },
        ),
    })

    # log workflow
    historicize(y, f='ifft', a={
        'x': x.name,
        'dim': dim,
        'new_dim': new_dim,
        'dtype': dtype,
        '**kwargs': kwargs,
    })

    return y


def _load_reciprocal_attrs(x: xr.DataArray):
    """Extract fft inverse attributes
    """
    name = x.attrs[_recip_name] if _recip_name in x.attrs else None
    attr = json.loads(x.attrs[_recip_attr]) if _recip_attr in x.attrs else None
    return name, attr


def _dump_reciprocal_attrs(x: xr.DataArray):
    """
    """
    attrs = dict()
    for attr in ['long_name', 'standard_name', 'units']:
        if attr in x.attrs:
            attrs[attr] = x.attrs[attr]
    return {
        _recip_name: x.name,
        _recip_attr: json.dumps(attrs),
    }
