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


__all__ = ['demean_cc', 'lag_window', 'rms', 'snr']


def demean_cc(
    dataset: xr.Dataset
):
    """
    Pair-wise demean the `xr.Dataset`.
    """
    dims = dataset.cc.dims
    if 'pair' in dims:
        for p in dataset.pair:
            for t in dataset.time:
                dataset.cc.loc[:, t, p] -= dataset.cc.loc[:, t, p].mean()
    else:
        for t in dataset.time:
            dataset.cc.loc[:, t] -= dataset.cc.loc[:, t].mean()
    return dataset


def lag_window(
    darray: xr.DataArray, window, scalar: float = 1., **kwargs
):
    """
    Return the trimmed lag window of the given DataArray
    (with dim 'lag').
    """
    assert isinstance(window, list) or isinstance(window, tuple), (
        'Window should be list or tuple of length 2!'
    )
    assert len(window) == 2, (
        'Window should be list or tuple of length 2!'
    )
    assert window[1]*scalar > window[0]*scalar, (
        'Window start should be greater than window end!'
    )
    return darray.where(
        (
            (darray.lag >= window[0]*scalar) &
            (darray.lag <= window[1]*scalar)
        ),
        drop=True
    )


def rms(
    darray: xr.DataArray, dim: str = 'lag', keep_attrs: bool = True
):
    """
    Return the root-mean-square of the DataArray.
    """
    darray = xr.ufuncs.square(darray)  # square
    return xr.ufuncs.sqrt(
        darray.mean(dim=dim, skipna=True, keep_attrs=keep_attrs)
    )  # mean and root


def snr(
    darray: xr.DataArray, signal_lag_window,
    noise_percentages: tuple = (.2, .8), **kwargs
):
    """
    Return the signal-to-noise ratio of the DataArray.
    """
    signal = Postprocess.lag_window(
        darray,
        window=signal_lag_window,
    )
    noise = Postprocess.lag_window(
        darray,
        window=noise_percentages,
        scalar=signal_lag_window[0],
    )
    snr = Postprocess.rms(signal) / Postprocess.rms(noise)
    snr.attrs = {
        'long_name': 'signal-to-noise ratio',
        'standard_name': 'signal_to_noise_ratio',
        'units': '-',
        'signal_lag_window': tuple(signal_lag_window),
        'noise_lag_window': tuple(
            noise_percentages[0]*signal_lag_window[0],
            noise_percentages[1]*signal_lag_window[0]
        ),
        'noise_percentages': tuple(noise_percentages),
    }
    return snr
