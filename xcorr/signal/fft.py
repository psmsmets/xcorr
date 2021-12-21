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


__all__ = ['fft', 'ifft', 'rfft', 'irfft']


_recip_name = '__reciprocal_name__'
_recip_attr = '__reciprocal_attr__'
_recip_zero = '__reciprocal_zero__'


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
        Set the complex dtype. If `None` (default), `np.complex128` is used.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`fftlib.fft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the FFT of ``x`` as a complex dtype.

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
    dtype = np.dtype(dtype or 'complex128')
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'complex' not in dtype.name:
        raise TypeError('dtype should be complex.')
    dtype = dtype.name

    # delta and frequency
    delta = np.round(
        np.fabs(x[dim][1] - x[dim][0]).values.item(),
        decimals=10
    )
    freq = fftlib.fftshift(fftlib.fftfreq(x[dim].size, d=delta))

    # wrapper to simplify ufunc input
    def _fft(f, **kwargs):
        F = fftlib.fft(f.astype(dtype), n=x[dim].size, axis=-1, **kwargs)
        return fftlib.fftshift(F, axes=-1)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed')

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_fft, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       output_dtypes=[dtype],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs=kwargs)

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
        The data array to apply the IFFT.

    dim : `str`, optional
        Coordinate name of ``x`` to apply the IFFT.
        Defaults to the last dimension of ``x``.

    new_dim : `str`, optional
        Coordinate name of ``x`` after applying the IFFT.
        Defaults to `time`.

    new_dim_attrs : `dict`, optional
        Dictionary with coordinate attributes.

    dtype : :class:`np.dtype`, optional
        Set the complex dtype. If `None` (default), `np.complex128` is used.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`fftlib.ifft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the IFFT of ``x`` as a complex dtype.

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
    dtype = np.dtype(dtype or 'complex128')
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'complex' not in dtype.name:
        raise TypeError('dtype should be complex.')
    dtype = dtype.name

    # delta
    delta = np.round(
        np.fabs(x[dim][1] - x[dim][0]).values.item(),
        decimals=10
    )
    time = np.linspace(
        start=0.,
        stop=(x[dim].size-1)/np.round(x[dim].size*delta, decimals=6),
        num=x[dim].size,
    )
    if _recip_zero in x[dim].attrs:
        time += x[dim].attrs[_recip_zero]

    # wrapper to simplify ufunc input
    def _ifft(F, **kwargs):
        F = fftlib.fftshift(F, axes=-1)
        return fftlib.ifft(F.astype(dtype), n=x[dim].size, axis=-1, **kwargs)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed')

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_ifft, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       output_dtypes=[dtype],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs=kwargs)

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


def rfft(
    x: xr.DataArray, dim: str = None,
    new_dim: str = None, new_dim_attrs: dict = None,
    dtype: np.dtype = None, **kwargs
):
    """
    RFFT an N-D labelled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The real data array to apply the RFFT.

    dim : `str`, optional
        Coordinate name of ``x`` to apply the RFFT.
        Defaults to the last dimension of ``x``.

    new_dim : `str`, optional
        Coordinate name of ``x`` after applying the RFFT.
        Defaults to `freq`.

    new_dim_attrs : `dict`, optional
        Dictionary with coordinate attributes.

    dtype : :class:`np.dtype`, optional
        Set the float dtype of ``x``. If `None` (default), ``x`` will be cast
        to `np.float64`.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`fftlib.rfft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the RFFT of ``x`` as a complex dtype with the
        same alignment as ``dtype``.

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
    dtype = np.dtype(dtype or 'float64')
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'float' not in dtype.name:
        raise TypeError('dtype should be float.')
    dtype_x = dtype.name
    dtype_y = f'complex{dtype.alignment * 16}'

    # samples, period and frequency
    n = x[dim].size - 1 if x[dim].size & 0x1 else x[dim].size
    fN = np.round(n//2/(x[dim][n-1] - x[dim][0]).item(), decimals=0)
    f = np.linspace(0., fN, n//2 + 1)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed')

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(fftlib.rfft, x.astype(dtype_x),
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       output_dtypes=[dtype_y],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={**kwargs, 'n': n, 'axis': -1})

    # add new dim
    y = y.assign_coords({
        new_dim: xr.DataArray(
            data=f,
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
    historicize(y, f='rfft', a={
        'x': x.name,
        'dim': dim,
        'new_dim': new_dim,
        'dtype': dtype,
        '**kwargs': kwargs,
    })

    return y


def irfft(
    x: xr.DataArray, dim: str = None,
    new_dim: str = None, new_dim_attrs: dict = None,
    dtype: np.dtype = None, **kwargs
):
    """
    IRFFT an N-D labelled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to apply the IRFFT.

    dim : `str`, optional
        Coordinate name of ``x`` to apply the IRFFT.
        Defaults to the last dimension of ``x``.

    new_dim : `str`, optional
        Coordinate name of ``x`` after applying the IRFFT.
        Defaults to `time`.

    new_dim_attrs : `dict`, optional
        Dictionary with coordinate attributes.

    dtype : :class:`np.dtype`, optional
        Set the float dtype of ``x``. If `None` (default), ``x`` will be cast
        to `np.float64`.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`fftlib.irfft`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data real array containing the IRFFT of ``x``.

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
    dtype = np.dtype(dtype or 'float64')
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    if 'float' not in dtype.name:
        raise TypeError('dtype should be float.')
    dtype_x = f'complex{dtype.alignment * 16}'
    dtype_y = dtype.name

    # samples, Nyquist and time
    n = 2 * x[dim].size - 2
    fN = x[dim][-1].item()
    time = np.linspace(0., (n-1)/2/fN, n)
    if _recip_zero in x[dim].attrs:
        time += x[dim].attrs[_recip_zero]

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed')

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(fftlib.irfft, x.astype(dtype_x),
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim]],
                       output_dtypes=[dtype_y],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       kwargs={**kwargs, 'n': None, 'axis': -1})

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
        _recip_zero: x[0].item(),
    }
