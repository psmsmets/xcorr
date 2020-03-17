# -*- coding: utf-8 -*-
"""
Python module for fetching waveform data and station metadata.

.. module:: datafetch

:author:
    Shahar Shani-Kadmiel (S.Shani-Kadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import os
import sys
import warnings
import numpy as np
from fnmatch import fnmatch

from obspy import read, Stream, Trace

# Pattern of the SDS filesystem structure
# year, network, station,  channel, julday
sdsPattern = os.path.join('*{}', '{}', '{}', '{}.*', '*.{}')

# number of seconds per day
one_day = 86400  # 24 * 60 * 60


def stream2SDS(
    stream: Stream, sds_path: str, datatype: str = 'D',
    out_format: str = 'MSEED', force_steim2: bool = False,
    force_override: bool = False, extra_samples: int = 10,
    sampling_precision: int = 2, verbose: bool = True
):
    """
    A convenient function to write a :class:`~obspy.Stream` object to a
    local SDS filesystem.

    Note
    ----
    Muli-day streams are split up into single days.

    Parameters
    ----------
    stream : :class:`~obspy.Stream`
        A :class:`~obspy.Stream` object containing one or more traces.

    sds_path : str
        Path (relative or absolute) to the local SDS filesystem path.

    datatype : str
        1 characters indicating the data type, recommended types are:
           'D' - Waveform data (default)
           'E' - Detection data
           'L' - Log data
           'T' - Timing data
           'C' - Calibration data
           'R' - Response data
           'O' - Opaque data

    out_format : str
        One of the obspy IO supported formats. The default and
        recommended format is 'MSEED'.

    force_override : bool
        By default, a check is performed to validate that the time span
        of the new data extends the time span of the old data on file.
        Set to ``False`` to bypass this check and override the data on
        disk. **Use with caution**.

    extra_samples : int
        Number of extra sample points at the end of each
        day. These will overlap with the subsequent day.

    sampling_precision : int
        Number of decimal digets to validate sampling precision of traces.
        If sampling rate is not the same it will be rounded to the specified
        precision (default is 2) and give a warning.

    force_steim2 : bool
        Force int32 as datatype and steim2 as encoding.

    verbose : bool
        By default, some information about the process is printed to the
        screen. Set to ``False`` to suppress output.
    """
    stream = slice_days(stream, extra_samples, sampling_precision)
    for tr in stream:
        if tr.stats.endtime - tr.stats.starttime < 60:
            continue
        starttime = tr.stats.starttime

        if force_steim2:
            tr.data = tr.data.astype(np.int32)

        net = tr.stats.network
        sta = tr.stats.station

        ch = tr.stats.channel
        ch_type = '{}.{}'.format(ch, datatype)

        path = os.path.join(sds_path, starttime.strftime('%Y'),
                            net, sta, ch_type)

        if not os.path.exists(path):
            os.makedirs(path)

        out_fn = os.path.join(
            path, '{}.{}.{}'.format(tr.id, datatype,
                                    starttime.strftime('%Y.%j')))
        if force_override or override(out_fn, tr, out_format):
            if verbose:
                print('Writing file {} as {} file...'.format(
                    out_fn, out_format))
                sys.stdout.flush()

            tr.split().write(out_fn, format=out_format, flush=True)


def slice_days(stream: Stream, extra: int = 10, sampling_precision: int = 2):
    """
    Slice traces in ``stream`` into day segments starting at the first
    sample after midnight till ``extra`` samples after midnight
    of the next day.

    Parameters
    ----------
    stream : :class:`~obspy.core.Stream`
        Obspy Stream object to split by days.

    extra : int
        Number of extra sample points at the end of each
        day. These will overlap with the subsequent day.

    sampling_precision : int
        Number of decimal digets to validate sampling precision of traces.
        If sampling rate is not the same it will be rounded to the specified
        precision (default is 2) and give a warning.

    force_rounding: bool
        Force sample rate rounding regardless if

    Returns
    -------
    st : :class:`~obspy.core.Stream`
        A stream with the sliced traces.
    """
    if extra < 1:
        raise ValueError('``extra`` must be larger than 1')
    st = Stream()
    try:
        stream.merge(method=0)
    except Exception:
        warnings.warn(
            'Sampling rate was rounded to {} decimal digits precision.'
            .format(sampling_precision)
        )
        for trace in stream:
            trace.stats.sampling_rate = round(
                trace.stats.sampling_rate, sampling_precision
            )
        stream.merge(method=0)
    for tr in stream:
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        delta = tr.stats.delta

        days = int(1 + (endtime - starttime) / one_day)
        for i in range(days):
            ti = starttime + i * one_day - extra * delta
            tf = ti + one_day + extra * delta
            st += tr.slice(ti, tf, nearest_sample=False)

    return st


def override(filename: str, trace: Trace, format: str = 'MSEED'):
    """
    Check if data limits in trace are more than data limits of data in
    filename.

    Returns ``True`` if data in trace expands data on file. Otherwise,
    returns ``False``.
    """
    try:
        existing = read(filename, format, True)
        existing.sort(keys=['starttime'])
        ti = existing[0].stats.starttime
        tf = existing[-1].stats.endtime
    except IOError:
        return True

    t0 = trace.stats.starttime
    t1 = trace.stats.endtime

    if ((t0 < ti and t1 > tf) or
            (t0 < ti and t1 >= tf) or
            (t0 <= ti and t1 > tf)):
        return True
    else:
        print(('\nData on file {} with:\n'
               'startime: {} - endtime: {} contains more data than {} with:\n'
               'startime: {} - endtime: {}\n\n'
               'To override the data on file remove it manually or pass '
               '``force_override=True`` to {}.\n').format(
            filename, ti, tf, trace.id, t0, t1, stream2SDS))
        return False


def exists(sds_path, starttime, net, sta, channels, channel_, ):
    """
    Check if waveform data already exists on the local filesystem.
    """
    files = []
    for root, dirs, files_ in os.walk(
        os.path.join(sds_path, str(starttime.year),
                     net, sta)):
        for f in files_:
            files += [os.path.join(root, f)]

    for ch in channels:
        if channel_[0] != '*' and ch.code not in channel_:
            print('{} not in {}'.format(ch.code, channel_)),
            sys.stdout.flush()
            continue

        for f in files:
            if fnmatch(f, sdsPattern.format(
                    starttime.year, net, sta,
                    ch.code, starttime.julday)):

                print('--> {} alredy exists. Skipping...'.format(f))
                sys.stdout.flush()
                return True
    return False
