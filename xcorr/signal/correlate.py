r"""

:mod:`signal.correlate` -- Correlate
====================================

Correlate an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
try:
    from pyfftw.interfaces import numpy_fft as fft
except ModuleNotFoundError:
    from numpy import fft
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..signal.normalize import norm1d, norm2d
from ..util.history import historicize


__all__ = ['correlate1d', 'correlate2d']


def correlate1d(
    in1: xr.DataArray, in2: xr.DataArray, normalize: bool = True,
    dtype: np.dtype = None, dim: str = None, **kwargs
):
    """
    One-dimensional crosscorrelate two N-D labelled arrays of data.

    Parameters
    ----------
    in1 : :class:`xarray.DataArray`
        The first data array.

    in2 : :class:`xarray.DataArray`
        The second data array.

    normalize : `bool`, optional
        If `True` (default), ``in1`` and ``in2`` are normalized before
        crosscorrelation.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the dtype of ``in1`` is used.

    dim : `str`, optional
        Coordinates name of ``in1`` and ``in2`` to crosscorrelate.
        Defaults to the last dimension of ``in1``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`xarray.apply_ufunc`.

    Returns
    -------
    cc : :class:`xarray.DataArray`
        Data array containing the discrete linear crosscorrelation of
        ``in1`` with ``in2``.

    """

    # dim
    dim = dim or in1.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')

    if dim not in in1.dims:
        raise ValueError(f'in1 has no dimensions "{dim}"')

    if dim not in in2.dims:
        raise ValueError(f'in2 has no dimensions "{dim}"')

    if in1.shape != in2.shape:
        raise ValueError('in1 and in2 should have the same shape!')

    # check regular spacing
    _regular_dim(in1[dim])

    # dtype
    dtype = dtype or in1.dtype
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    dtype = np.dtype(dtype).type

    # new dim
    new_dim = f'delta_{dim}'

    # pad
    npad = 2*in1[dim].size-1

    # set axis
    ax = -1

    # normalize?
    if normalize:
        in1 = norm1d(in1, dim)
        in2 = norm1d(in2, dim)

    # wrapper to simplify ufunc input
    def _correlate1d(f, g):
        F = fft.fft(f, s=npad, axis=ax)
        G = fft.fft(g, s=npad, axis=ax)
        FG = F * np.conjugate(G)
        cc = fft.fftshift(np.real(fft.ifft(FG, axis=ax)), axes=ax)
        return cc

    # dask collection?
    dargs = {}
    if dask and (dask.is_dask_collection(in1) or dask.is_dask_collection(in2)):
        dargs = dict(dask='allowed', output_dtypes=[in1.dtype])

    # apply ufunc (and optional dask distributed)
    cc = xr.apply_ufunc(_correlate1d, in1, in2,
                        input_core_dims=[[dim], [dim]],
                        output_core_dims=[[new_dim]],
                        keep_attrs=False,
                        vectorize=False,
                        **dargs,
                        **kwargs)

    # add new dim
    cc = cc.assign_coords({
        new_dim: _new_coord(in1[dim]),
    })

    # set attributes
    cc.name = 'cc'
    cc.attrs = {
        **cc.attrs,
        'long_name': 'Crosscorrelation Estimate',
        'standard_name': 'crosscorrelation_estimate',
        'units': '-',
        'add_offset': dtype(0.),
        'scale_factor': dtype(1.),
        'valid_range': dtype([-1., 1.]),
        'normalize': np.byte(normalize),
        'bias_correct': np.byte(0),
        'unbiased': np.byte(0),
        'history_in1': in1.attrs['history'] if 'history' in in1.attrs else '',
        'history_in2': in2.attrs['history'] if 'history' in in2.attrs else '',
    }

    # log workflow
    historicize(cc, f='correlate1d', a={
        'in1': in1.name,
        'in2': in2.name,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return cc


def correlate2d(
    in1: xr.DataArray, in2: xr.DataArray, normalize: bool = True,
    dtype: np.dtype = None, dims: tuple = None, **kwargs
):
    """
    Two-dimensional crosscorrelate two N-D labelled arrays of data.

    Parameters
    ----------
    in1 : :class:`xarray.DataArray`
        The first data array.

    in2 : :class:`xarray.DataArray`
        The second data array.

    normalize : `bool`, optional
        If `True` (default), ``in1`` and ``in2`` are normalized before
        crosscorrelation.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the dtype of ``in1`` is used.

    dims : `tuple`, optional
        A tuple pair with the coordinates name of ``in1`` and ``in2`` to
        crosscorrelate. Defaults to the last two dimensions of ``in1``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`xarray.apply_ufunc`.

    Returns
    -------
    cc : :class:`xarray.DataArray`
        Data array containing the discrete linear crosscorrelation of
        ``in1`` with ``in2``.

    """

    # dim
    dims = dims or in1.dims[-2:]
    if not isinstance(dims, tuple) or len(dims) != 2:
        raise TypeError('dims should be a tuple of length 2')

    for d in dims:
        if d not in in1.dims:
            raise ValueError(f'in1 has no dimensions "{dim}"')
        if d not in in2.dims:
            raise ValueError(f'in2 has no dimensions "{dim}"')

    if in1.shape != in2.shape:
        raise ValueError('in1 and in2 should have the same shape!')

    # check regular spacing
    _regular_dim(in1[dims[0]])
    _regular_dim(in1[dims[1]])

    # dtype
    dtype = dtype or in1.dtype
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    dtype = np.dtype(dtype).type

    # new dims
    new_dims = (f'delta_{dims[0]}',  f'delta_{dims[1]}')

    # pad
    npad = [2*in1[d].size-1 for d in dims]

    # set axes
    ax = (-2, -1)

    # normalize?
    if normalize:
        in1 = norm2d(in1, dims)
        in2 = norm2d(in2, dims)

    # wrapper to simplify ufunc input
    def _correlate2d(f, g):
        F = fft.fft2(f, s=npad, axes=ax)
        G = fft.fft2(g, s=npad, axes=ax)
        FG = F * np.conjugate(G)
        cc = fft.fftshift(np.real(fft.ifft2(FG, axes=ax)), axes=ax)
        return cc

    # dask collection?
    dargs = {}
    if dask and (dask.is_dask_collection(in1) or dask.is_dask_collection(in2)):
        dargs = dict(dask='allowed', output_dtypes=[in1.dtype])

    # apply ufunc (and optional dask distributed)
    cc = xr.apply_ufunc(_correlate2d, in1, in2,
                        input_core_dims=[dims, dims],
                        output_core_dims=[new_dims],
                        keep_attrs=False,
                        vectorize=False,
                        **dargs,
                        **kwargs)

    # add new dim
    cc = cc.assign_coords({
        new_dims[0]: _new_coord(in1[dims[0]]),
        new_dims[1]: _new_coord(in1[dims[1]]),
    })

    # set attributes
    cc.name = 'cc2'
    cc.attrs = {
        **cc.attrs,
        'long_name': 'Two-dimensional Crosscorrelation Estimate',
        'standard_name': '2d_crosscorrelation_estimate',
        'units': '-',
        'add_offset': dtype(0.),
        'scale_factor': dtype(1.),
        'valid_range': dtype([-1., 1.]),
        'normalize': np.byte(normalize),
        'bias_correct': np.byte(0),
        'unbiased': np.byte(0),
        'history_in1': in1.attrs['history'] if 'history' in in1.attrs else '',
        'history_in2': in2.attrs['history'] if 'history' in in2.attrs else '',
    }

    # log workflow
    historicize(cc, f='correlate2d', a={
        'in1': in1.name,
        'in2': in2.name,
        'dims': dims,
        '**kwargs': kwargs,
    })

    return cc


def _new_coord(old):
    """Private helper to construct the new dimension.
    """
    n = 2*old.size-1
    s = (old.max()-old.min()).item()/(old.size-1)

    new = xr.DataArray(
        data=fft.fftshift(fft.fftfreq(n, 1/n/s)),
        dims=(f'delta_{old.name}'),
        name=f'delta_{old.name}',
        attrs={
            'long_name': f'Delta {old.long_name}',
            'standard_name': f'delta_{old.standard_name}',
            'units': old.units,
        }
    )

    return new


def _regular_dim(dim):
    """Private helper to verify the dimension.
    """
    if not np.all(np.abs(dim.diff(dim.name, 2)) < 1e-10):
        raise ValueError(f'coordinate "{dim.name}" should be regularly spaced')
