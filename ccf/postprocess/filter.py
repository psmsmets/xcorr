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


__all__ = ['butterworth_filter']


def butterworth_filter(
    darray: xr.DataArray, order: int, btype: str, frequency: float,
    **kwargs
):
    """
    Butterworth filter a `xr.DataArray`.
    """
    sos = signal.butter(
        N=order,
        Wn=frequency,
        btype=btype,
        output='sos',
        fs=darray.lag.sampling_rate
    )
    fun = lambda x, sos: signal.sosfiltfilt(sos, x)

    darray_filt = xr.apply_ufunc(fun, darray, sos)
    darray_filt.attrs = {
        **darray.attrs,
        'filtered': np.int8(True),
        'filter_design': 'butterworth',
        'filter_method': 'cascaded second-order sections (sos)',
        'filter_zerophase': np.int8(True),
        'filter_order': order,
        'filter_btype': btype,
        'filter_frequency': frequency,
    }
    return darray_filt
