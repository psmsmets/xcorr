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
from obspy import Trace, Stream


__all__ = ['running_rms', 'running_rms_stream', 'running_rms_trace']


def running_rms(signal, **kwargs):
    """
    Returns a new :class:`obspy.Stream` or :class:`obpsy.Trace` with the running
    root-mean-square amplitude per `window` (seconds).
    """
    if isinstance(signal, Stream):
        return running_rms_stream(signal, **kwargs)
    elif isinstance(signal, Trace):
        return running_rms_trace(signal, **kwargs)
    else:
        raise TypeError('Signal should be of type obspy.Trace or obspy.Stream!')


def running_rms_stream(stream: Stream, **kwargs):
    """
    Returns a new `obspy.Stream` with the running root-mean-square
    amplitude per `window` (seconds).
    """
    rms_stream = Stream()

    for trace in stream:
        rms_stream += running_rms_trace(trace, **kwargs)

    return rms_stream


def running_rms_trace(trace: Trace, window: float = 1.):
    """
    Returns a new `obspy.Trace` with the running root-mean-square
    amplitude per `window` (seconds).
    """
    npts = int(trace.stats.endtime-trace.stats.starttime) / window
    rms_trace = Trace(data=np.zeros(int(npts), dtype=np.float64))
    rms_trace.stats.network = trace.stats.network
    rms_trace.stats.station = trace.stats.station
    rms_trace.stats.location = trace.stats.location
    if window >= 100.:
        band = 'U'
    elif window >= 10:
        band = 'V'
    else:
        band = 'L'
    rms_trace.stats.channel = band + trace.stats.channel[1:]
    rms_trace.stats.delta = window
    rms_trace.stats.starttime = trace.stats.starttime + window/2
    rms_trace.stats.npts = npts

    for index, windowed_trace in enumerate(
        trace.slide(window_length=window, step=window)
    ):
        rms_trace.data[index] = np.sqrt(
            np.sum(
                np.power(windowed_trace.data, 2)
            )/windowed_trace.stats.npts
        )

    return rms_trace
