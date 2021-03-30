r"""

:mod:`stream.process.running_rms` -- Running RMS
================================================

Running root-mean-square amplitude of a :class:`obspy.Trace` and
:class:`obspy.Stream`.

"""

# Mandatory imports
import numpy as np
from obspy import Trace, Stream


__all__ = ['running_rms', 'running_rms_stream', 'running_rms_trace']


def running_rms(signal, **kwargs):
    r"""Running root-mean-square amplitude.

    Parameters
    ----------
    signal : :class:`obspy.Stream` or :class:`obspy.Trace`
        Input signal to apply the running root-mean-square amplitude.

    kwargs : optional
        Arguments passed to :func:`running_rms_trace`.

    Returns
    -------
    rms_signal : same type as ``signal``
        Output signal after the running root-mean-square amplitude with the
        same type as ``signal``.

    """
    if isinstance(signal, Stream):
        return running_rms_stream(signal, **kwargs)
    elif isinstance(signal, Trace):
        return running_rms_trace(signal, **kwargs)
    else:
        raise TypeError(
            'Signal should be of type :class:`obspy.Trace` or '
            ':class:`obspy.Stream!`'
        )


def running_rms_stream(stream: Stream, **kwargs):
    r"""Stream running root-mean-square amplitude.

    Parameters
    ----------
    stream : :class:`obspy.Stream`
        Input stream to apply the running root-mean-square amplitude.

    kwargs : optional
        Arguments passed to :func:`running_rms_trace`.

    Returns
    -------
    rms_stream : :class:`obspy.Trace`
        Output stream after the running root-mean-square amplitude.

    """
    rms_stream = Stream()

    for trace in stream:
        rms_stream += running_rms_trace(trace, **kwargs)

    return rms_stream


def running_rms_trace(trace: Trace, window: float = 1.):
    r"""Trace running root-mean-square amplitude.

    Parameters
    ----------
    trace : :class:`obspy.Trace`
        Input trace to apply the running root-mean-square amplitude.

    window : `float`, optional
        Set the running window duration, in seconds. Defaults to 1s.

    Returns
    -------
    rms_trace : :class:`obspy.Trace`
        Output trace after the running root-mean-square amplitude.

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
