r"""

:mod:`signal.filter` -- Filter
==============================

Filter an N-D labelled array of data.

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


__all__ = ['filter']


def filter(
    x: xr.DataArray, frequency, btype: str, order: int = 2,
    dim: str = None, **kwargs
):
    """
    Butterworth filter an N-D labelled array of data.

    Implementation of :func:`scipy.signal.butter` and
    :func:`scipy.signal.sosfiltfilt` to a :class:`xarray.DataArray` using
    :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be filtered.

    frequency : `float` or `tuple`
        The corner frequency (pair) of the filter, in Hz.

    btype : `str` {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
       The type of filter. Default is ‘lowpass’.

    order : `int`, optional
        The order of the filter. Default is 2.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Defaults to the
        last dimension of ``x``.

    **kwargs :
        Additional parameters provided to :func:`xarray.apply_ufunc`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The filtered data array.

    """

    # dim
    dim = dim or x.dims[-1]
    assert dim in x.dims, f'x has no dimension "{dim}"!'

    # check
    assert 'sampling_rate' in x[dim].attrs, (
        f'Dimension "{dim}" has no attribute "sampling_rate"!'
    )
    assert (
        isinstance(frequency, float) or
        (isinstance(frequency, tuple) and len(frequency) == 2)
    ), 'Corner frequency should be a `float` or tuple-pair with (min, max)!'

    # construct digital sos filter coefficients
    sos = signal.butter(
        N=order,
        Wn=frequency,
        btype=btype,
        output='sos',
        fs=x[dim].attrs['sampling_rate']
    )

    # get index of dim
    axis = x.dims.index(dim)
    axis = -1 if axis == len(x.dims) - 1 else axis

    # sosfiltfilt wrapper to simplify ufunc input
    def _filter(obj):
        return signal.sosfiltfilt(sos, obj, axis=axis)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='parallelized', output_dtypes=[x.dtype])

    # apply sosfiltfilt as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_filter, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # log workflow
    historicize(x, f='filter', a={
        'x': y.name,
        'frequency': frequency,
        'btype': btype,
        'order': order,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
