r"""

:mod:`signal.spectrogram` -- Spectrogram
========================================

Generate a spectrogram of an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
#import pandas as pd
import numpy as np
from scipy import signal


# Relative import
from ..util import to_seconds


__all__ = ['psd', 'psd_f_t']


def psd(
    x: xr.DataArray, duration: float = None, padding_factor: int = 2,
    dim: str = 'lag', **kwargs
):
    r"""Compute an N-D labelaled spectrogram with consecutive Fourier
    transforms.

    Implementation of :func:`scipy.signal.spectrogram` to a
    :class:`xarray.DataArray`.

    The dimension ``dim`` for which to compute the spectrogram should contain
    both sampling attributes ``sampling_rate`` and ``delta``.
    The computed spectrogram contains the coordinates of ``x`` and a new
    coordinate 'frequency' with sample frequencies ranging from dc up to
    Nyquist ``dim.sampling_rate/2``. Spectrogram time segments correspond to
    the samples in ``dim``, with `NaN` at the outer edges (``duration/2``). 

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be filtered.

    duration : `float`
       The duration of each segment, in seconds.

    padding_factor : `int`, optional
        Factor applied to duration, if a zero padded FFT is desired. If `None`,
        the FFT length is duration. Defaults to None. Default is 2.

    dim : `str`, optional
        The coordinates name if ``x`` to be filtered over. Default is 'lag'.

    **kwargs
        Extra arguments passed on to :func:`scipy.signal.spectrogram`.

    Returns:
    --------
    y : :class:`xarray.DataArray`
        The computed spectrogram for ``x``.
    """
    assert dim in x.dims, (
        'x has no dimension "{}"!'.format(dim)
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension has no attribute "{sampling_rate}"!'
    )
    assert 'delta' in x[dim].attrs, (
        'Dimension has no attribute "{delta}"!'
    )

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    duration = duration if (
        duration and duration > x[dim].delta
    ) else x[dim].delta

    win_len = int(duration * x[dim].sampling_rate)

    assert win_len >= 16, 'Change duration to have at least 16 sample points!'

    f, t, Sxx = signal.spectrogram(
        x=x.values,
        fs=x[dim].sampling_rate,
        nperseg=win_len,
        noverlap=win_len-1,
        nfft=int(padding*win_len),
        scaling='density',
        mode='psd',
        axis=x.dims.index(dim),
        **kwargs
    )

    # construct coordinates (lag last!)
    coords = {}
    for d in x.dims:
        if d != dim:
            coords[d] = x[d]
    coords['freq'] = xr.DataArray(
        data=f,
        name='freq',
        dims=('freq'),
        coords=[f],
        attrs={
            'long_name': 'Frequency',
            'standard_name': 'frequency',
            'units': 'Hz',
        }
    )
    coords[dim] = x[dim]  # should be last dim!

    # construct output
    y = xr.DataArray(
        data=np.zeros(
            [len(coord) for name, coord in coords.items()],
            dtype=np.dtype(x.dtype)
        ),
        dims=coords.keys(),
        coords=coords,
        name='psd',
        attrs={
            'long_name': 'Power Spectral Density',
            'standard_name': 'power_spectral_density',
            'units': 'Hz**-1',
            'from_variable': x.name,
            'scaling': 'density',
            'mode': 'psd',
            'duration': duration,
            'padding_factor': padding_factor,
            'centered': np.byte(True),
            **kwargs
        },
    )

    # fill accordingly
    edge = int(np.rint(win_len/2-1))

    y.data.fill(np.nan)
    y.loc[{'lag':y[dim][edge:-1-edge]}] = Sxx

    historicize(y, f='psd', a={
        'x': x.name,
        'duration': duration,
        'padding_factor': padding_factor,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def psd_f_t(
    x: xr.DataArray, duration: float = None, padding_factor: int = None,
    overlap: float = None, dim: str = 'lag', **kwargs
):
    r"""Compute an N-D labelaled spectrogram with consecutive Fourier
    transforms with manual time segment control.

    Implementation of :func:`scipy.signal.spectrogram` to a
    :class:`xarray.DataArray`.

    The dimension ``dim`` for which to compute the spectrogram should contain
    both sampling attributes ``sampling_rate`` and ``delta``.

    The computed spectrogram contains the coordinates of ``x`` minus ``dim``,
    and two new coordinates, 'psd_f' and 'psd_t`, with sample frequencies ranging
    from dc up to Nyquist ``dim.sampling_rate/2`` and with time segments depending
    on ``dim``, ``duration`` and ``overlap``, respectively. 

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be filtered.

    duration : `float`
       The duration of each segment, in seconds.

    padding_factor : `int`, optional
        Factor applied to duration, if a zero padded FFT is desired. If `None`,
        the FFT length is duration. Defaults to None. Default is 2.

    overlap : `float`
       The decimal overlap between two consecutive segment: [0,1).

    dim : `str`, optional
        The coordinates name if ``x`` to be filtered over. Default is 'lag'.

    **kwargs
        Extra arguments passed on to :func:`scipy.signal.spectrogram`.

    Returns:
    --------
    y : :class:`xarray.DataArray`
        The computed spectrogram for ``x``.
    """
    assert dim in x.dims, (
        'x has no dimension "{}"!'.format(dim)
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension has no attribute "{sampling_rate}"!'
    )
    assert 'delta' in x[dim].attrs, (
        'Dimension has no attribute "{delta}"!'
    )

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    duration = duration if (
        duration and duration > x[dim].delta
    ) else x[dim].delta

    win_len = int(duration * x[dim].sampling_rate)

    assert win_len >= 16, 'Change duration to have at least 16 sample points!'

    overlap = overlap if overlap and (0. < overlap < 1.) else .9
    win_shift = int(win_len*overlap)

    f, t, Sxx = signal.spectrogram(
        x=x.values,
        fs=x[dim].sampling_rate,
        nperseg=win_len,
        noverlap=win_shift,
        nfft=win_len*padding_factor,
        scaling='density',
        mode='psd',
        axis=x.dims.index(dim),
        **kwargs
    )

    t += to_seconds(x[dim].values[0])

    coords = {}
    for d in x.dims:
        if d != dim:
            coords[d] = x[d]
    coords['psd_f'] = (
        'psd_f',
        f,
        {
            'long_name': 'Frequency',
            'standard_name': 'frequency',
            'units': 'Hz',
        }
    )
    coords['psd_t'] = (
        'psd_t',
        t,
        {
            'long_name': 'Time',
            'standard_name': 'time',
        }
    )

    y = xr.DataArray(
        data=Sxx,
        dims=coords.keys(),
        coords=coords,
        name='psd',
        attrs={
            'long_name': 'Power Spectral Density',
            'standard_name': 'power_spectral_density',
            'units': 'Hz**-1',
            'from_variable': x.name,
            'scaling': 'density',
            'mode': 'psd',
            'duration': duration,
            'padding_factor': padding_factor,
            'overlap': overlap,
            'centered': np.byte(True),
            **kwargs
        }, 
    )

    historicize(y, f='psd_f_t', a={
        'x': x.name,
        'duration': duration,
        'padding_factor': padding_factor,
        'overlap': overlap,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
