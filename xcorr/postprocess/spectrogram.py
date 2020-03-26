# -*- coding: utf-8 -*-
"""
Python module with crosscorrelation processing, waveform preprocessing and
crosscorrelation postprocessing routines.

.. module:: process

:author:
    Pieter Smets (P.S.M.Smets@tudelft.nl)

:copyright:
    Pieter Smets

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""


# Mandatory imports
import xarray as xr
import pandas as pd
import numpy as np
from scipy import signal


# Relative import
from ..util import to_seconds


__all__ = ['psd', 'psd_f_t']


def psd(
    darray: xr.DataArray, duration: float = None, padding: int = None,
    lag_to_seconds: bool = False, **kwargs
):
    """
    PSD of a `xr.DataArray`.
    """
    padding = padding if padding and padding >= 2 else 2
    duration = duration if (
        duration and duration > darray.lag.delta
    ) else darray.lag.delta

    win_len = int(duration * darray.lag.sampling_rate)

    f, t, Sxx = signal.spectrogram(
        x=darray.values,
        fs=darray.lag.sampling_rate,
        nperseg=win_len,
        noverlap=win_len-1,
        nfft=int(padding*win_len),
        scaling='density',
        mode='psd',
        axis=darray.dims.index('lag'),
        **kwargs
    )

    # construct coordinates (lag last!)
    coords = {}
    for dim in darray.dims:
        if dim != 'lag':
            coords[dim] = darray[dim]
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
    coords['lag'] = darray['lag']  # should be last dim!

    # construct output
    darray = xr.DataArray(
        data=np.zeros(
            [len(coord) for name, coord in coords.items()],
            dtype=np.dtype(darray.dtype)
        ),
        dims=coords.keys(),
        coords=coords,
        name='psd',
        attrs={
            'long_name': 'Power Spectral Density',
            'standard_name': 'power_spectral_density',
            'units': 'Hz**-1',
            'from_variable': darray.name,
            'scaling': 'density',
            'mode': 'psd',
            'duration': duration,
            'padding': padding,
            'centered': np.byte(True),
            **kwargs
        },
    )

    # fill accordingly
    edge = int(np.rint(win_len/2-1))

    darray.data.fill(np.nan)
    darray.loc[{'lag':darray.lag[edge:-1-edge]}] = Sxx

    if lag_to_seconds:
        darray.lag.data = to_seconds(darray.lag.data)
        darray.lag.attrs['units'] = 's'

    return darray


def psd_f_t(
    darray: xr.DataArray, duration: float = None, padding: int = None,
    overlap: float = None, **kwargs
):
    """
    PSD of a `xr.DataArray`.
    """
    padding = padding if padding and padding >= 2 else 2
    duration = duration if (
        duration and duration > darray.lag.delta
    ) else darray.lag.delta
    overlap = overlap if overlap and (0. < overlap < 1.) else .9

    f, t, Sxx = signal.spectrogram(
        x=darray.values,
        fs=darray.lag.sampling_rate,
        nperseg=int(duration * darray.lag.sampling_rate),
        noverlap=int(duration * darray.lag.sampling_rate * overlap),
        nfft=int(padding * duration * darray.lag.sampling_rate),
        scaling='density',
        mode='psd',
        axis=darray.dims.index('lag'),
        **kwargs
    )

    t += to_seconds(darray.lag.values[0])

    coords = {}
    for dim in darray.dims:
        if dim != 'lag':
            coords[dim] = darray[dim]
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
        t,  # pd.to_timedelta(t,unit='s'),
        {
            'long_name': 'Time',
            'standard_name': 'time',
        }
    )

    return xr.DataArray(
        data=Sxx,
        dims=coords.keys(),
        coords=coords,
        name='psd',
        attrs={
            'long_name': 'Power Spectral Density',
            'standard_name': 'power_spectral_density',
            'units': 'Hz**-1',
            'from_variable': darray.name,
            'scaling': 'density',
            'mode': 'psd',
            'overlap': overlap,
            'duration': duration,
            'padding': padding,
            **kwargs
        },
    )
