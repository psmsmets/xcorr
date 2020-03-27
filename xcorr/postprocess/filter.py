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
import numpy as np
import xarray as xr
from scipy import signal


# Relative imports
from ..util.history import historicize


__all__ = ['filter']


def filter(
    x: xr.DataArray, frequency, btype: str, order: int = 2,
    dim: str = 'lag', **kwargs
):
    r"""Butterworth filter an N-D labeled array of data.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be filtered.

    btype : `str` {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
       The type of filter. Default is ‘lowpass’.

    order : `int`, optional
        The order of the filter. Default is 2.

    dim : `str`, optional
        The coordinates name if ``x`` to be filtered over. Default is 'lag'.

    Returns:
    --------
    y : :class:`xarray.DataArray`
        The filtered output of ``x``.
    """
    assert dim in x.dims, (
        'x has no dimension "{}"!'.format(dim)
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension "{}" has no attribute "sampling_rate"!'.format(dim)
    )
    assert (
        isinstance(frequency, float) or
        (isinstance(frequency, list) and len(frequency) == 2)
    ), 'Corner frequency should be a `float` or tuple-pair with (min, max)!'

    sos = signal.butter(
        N=order,
        Wn=frequency,
        btype=btype,
        output='sos',
        fs=x[dim].sampling_rate
    )
    fun = lambda x, sos: signal.sosfiltfilt(sos, x, **kwargs)

    y = xr.apply_ufunc(fun, x, sos)

    # add and update attributes
    y.attrs = x.attrs
    historicize(y, f='filter', a={
        'frequency': frequency,
        'btype': btype,
        'order': order,
        'dim': dim,
        '**kwargs': kwargs,
    })
    
    return y
