r"""

:mod:`signal.timeshift` -- Timeshift
====================================

Timeshift an N-D labelled array of data using the FFT.

"""


# Mandatory imports
import numpy as np
import xarray as xr

# Relative imports
from ..util.history import historicize
from .fft import fft, ifft, rfft, irfft


__all__ = ['timeshift']


def timeshift(
    x: xr.DataArray, delay: float, dtype: np.dtype = None,
    dim: str = None, fast: bool = True, **kwargs
):
    """
    Timeshift an N-D labelled array of data in the frequency domain.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array to be timeshifted.

    delay : `float` or :class:`xarray.DataArray`
        The delay to timeshift ``x``.

    dtype : :class:`np.dtype`, optional
        Set the dtype. If `None` (default), the dtype of ``x`` is used.

    dim : `str`, optional
        Coordinates name of ``x`` to timeshift.
        Defaults to the last dimension of ``x``.

    fast : `bool`, optional
        Make use of the faster rfft and irfft (default). Note that if ``dim``
        has an odd number of samples this will trim one sample.
        It is recommended to apply a taper to ``x`` to avoid phase-wrapping
        artefacts.

    **kwargs :
        Any additional keyword arguments will be passed to `(i)(r)fft`.

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

    # check regular spacing
    if not np.all(np.abs(x[dim].diff(dim, 2)) < 1e-10):
        raise ValueError(f'coordinate "{dim}" should be regularly spaced')

    # delay
    if isinstance(delay, xr.DataArray):
        if dim in delay.dims:
            raise ValueError(f'delay cannot depend on dim "{dim}"')
        for d in delay.dims:
            if d not in x.dims:
                raise ValueError(f'delay dim "{d}" not existing in x')
    elif not isinstance(delay, float):
        raise TypeError('delay should be a float or DataArray')

    # dtype
    dtype = np.dtype(dtype or x.dtype)
    if not isinstance(dtype, np.dtype):
        raise TypeError('dtype should be a numpy.dtype')
    dtype = np.dtype(dtype).type

    # Make use of xarray wrapped fft and ifft to allow delay broadcasting!
    # rfft
    X = rfft(x, **kwargs) if fast else fft(x, **kwargs)

    # phase shift
    X = X * np.exp(-1j * 2 * np.pi * delay * X.freq)

    # irfft
    y = irfft(X, **kwargs) if fast else xr.ufuncs.real(ifft(X, **kwargs))

    # restore dtype
    y = y.astype(x.dtype)

    # restore coordinate
    y = y.assign_coords({dim: x[dim][0:y[dim].size]})

    # restore attributes
    y.name = x.name
    y.attrs = x.attrs

    # log workflow
    historicize(y, f='timeshift', a={
        'x': x.name,
        'delay': delay if isinstance(delay, float) else delay.name,
        'dim': dim,
        'fast': fast,
        '**kwargs': kwargs,
    })

    return y
