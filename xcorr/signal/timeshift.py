r"""

:mod:`signal.timeshift` -- Timeshift
====================================

Timeshift an N-D labelled array of data using the FFT.

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
from ..util.history import historicize


__all__ = ['timeshift']


def timeshift(
    x: xr.DataArray, delay: float, pad: bool = True,
    dtype: np.dtype = None, dim: str = None, **kwargs
):
    """
    Timeshift an N-D labelled array of data in the frequency domain.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be timeshifted.

    delay : `float`
        The delay (seconds) to timeshift ``x``.

    pad : `bool`, optional
        If `True` (default), ``x`` is zero-padded to 2*n-1 samples to avoid
        phase wrapping.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the dtype of ``x`` is used.

    dim : `str`, optional
        Coordinates name of ``x`` to timeshift.
        Defaults to the last dimension of ``x``.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`xarray.apply_ufunc`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        Data array containing the timeshift of ``x``.

    """

    # dim
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    if 'sampling_rate' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{sampling_rate}"!')

    if 'delta' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{delta}"!')

    # check regular spacing
    if not np.all(np.abs(x[dim].diff(dim, 2)) < 1e-10):
        raise ValueError(f'coordinate "{dim}" should be regularly spaced')

    # dtype
    dtype = dtype or x.dtype
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    dtype = np.dtype(dtype).type

    # pad
    if pad:
        n = 2 * x[dim].size - 1
        edge = np.int(np.round((x[dim].size - 1)/2))
        npad = [(edge, x[dim].size - 1 - edge)]
        pargs = dict(mode='constant', constant_values=0)
        indices = np.arange(edge, edge + x[dim].size, 1)
    else:
        n = x[dim].size
        npad = [(0, 0)]
        pargs = {}

    # phase shifts
    df = 1/n/x[dim].attrs['sampling_rate']
    phase_shift = np.exp(2 * np.pi * 1j * fft.fftfreq(n, df))

    # set axis
    ax = -1

    # correlate2d wrapper to simplify ufunc input
    def _timeshift(y):
        if pad:
            _npad = [(0, 0)] * (len(y.shape)-1) + npad
            Y = fft.fft(np.pad(y, pad_width=_npad, **pargs), axis=ax)
        else:
            Y = fft.fft(y, axis=ax)
        y = fft.fftshift(np.real(fft.ifft(Y * phase_shift, axis=ax)), axes=ax)
        if pad:
            y = np.take(y, indices, axis=ax)
        return y

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[x.dtype])

    # apply _correlate2d as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_timeshift, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # log workflow
    historicize(y, f='timeshift', a={
        'x': x.name,
        'delay': delay,
        'pad': pad,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
