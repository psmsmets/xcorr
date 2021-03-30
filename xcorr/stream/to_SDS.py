# -*- coding: utf-8 -*-
"""
Python module for saving a stream in an SDS filesystem structure.

.. module:: to_SDS

:author:
    Shahar Shani-Kadmiel (S.Shani-Kadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""


# Mandatory imports
import os
import warnings
from obspy import UTCDateTime, read, Stream
from obspy.core.util.testing import streams_almost_equal


__all__ = ['to_SDS']


# Pattern of the SDS filesystem structure
# year, network, station,  channel, julday
sdsPattern = os.path.join('*{}', '{}', '{}', '{}.*', '*.{}')


def to_SDS(
    stream: Stream, sds_path: str, datatype: str = 'D',
    out_format: str = 'MSEED', method: str = 'merge',
    min_seconds: float = 3600., extra_samples: int = 10,
    sampling_precision: int = 2, verbose: bool = True
):
    """
    A convenient function to write a :class:`~obspy.Stream` object to a
    local SDS filesystem.

    Note
    ----
    Stream is split up every midnight.

    Parameters
    ----------
    stream : :class:`obspy.Stream`
        A :class:`~obspy.Stream` object containing one or more traces.

    sds_path : `str`
        Path (relative or absolute) to the local SDS filesystem path.

    datatype : `str`
        1 characters indicating the data type, recommended types are:
           'D' - Waveform data (default)
           'E' - Detection data
           'L' - Log data
           'T' - Timing data
           'C' - Calibration data
           'R' - Response data
           'O' - Opaque data

    out_format : `str`
        One of the obspy IO supported formats. The default and
        recommended format is 'MSEED'.

    method : {'merge', 'overwrite'}
        By default, data in ``stream`` is merged with data in the SDS
        archive if it adds more data and fills any existing gaps. This
        is slow but safe. Set to ``overwrite`` to replace the
        existing data with the new data. **Use with caution**.

    extra_samples : `int`
        Number of extra sample points at the end of each
        day. These will overlap with the subsequent day.

    sampling_precision : `int`
        Number of decimal digits to validate sampling precision of traces.
        If sampling rate is not the same it will be rounded to the specified
        precision (default is 2) and give a warning.

    verbose : `bool`
        Information about the process is printed to stdout if `True`.
        Set to ``False`` (default) to suppress output.
    """

    # check method
    if method not in ('merge', 'overwrite'):

        raise ValueError('Method should be "merge" or "overwrite".')

    # get merged (masked) stream per seedid per day
    stream = _slice_days(stream, extra_samples, sampling_precision)

    # write masked day traces
    for tr in stream:

        starttime = tr.stats.starttime

        if tr.stats.endtime - starttime < min_seconds:

            continue

        id = tr.id

        net = tr.stats.network
        sta = tr.stats.station

        ch = tr.stats.channel
        ch_type = '{}.{}'.format(ch, datatype)

        path = os.path.join(sds_path, starttime.strftime('%Y'),
                            net, sta, ch_type)

        if not os.path.exists(path):

            os.makedirs(path)

        out_fn = os.path.join(
            path, '{}.{}.{}'.format(id, datatype,
                                    starttime.strftime('%Y.%j')))

        if method == 'merge':

            tr = tr.split()

            try:

                existing = read(out_fn, out_format)

                # compare the existing data to the new data
                if streams_almost_equal(tr, existing):

                    warnings.warn(
                        f'File {out_fn} already contains the data for {id}.'
                        ' Set ``method="overwrite"`` to force overwriting.'
                    )

                    # skip merging and writing if data is the same
                    continue

                # add two streams
                tr += existing

                # merge both streams, potentially filling in gaps
                tr.merge(method=1)

                if verbose:

                    print(f'Writing file {out_fn} as {out_format} file...')

                # split on remaining gaps
                tr.split().write(out_fn, out_format, flush=True)

                continue

            except FileNotFoundError:

                # file does not exist, write it
                if verbose:

                    print(f'Writing file {out_fn} as {out_format} file...')

                tr.write(out_fn, out_format, flush=True)

                continue

        else:

            if verbose:

                print(f'Overwriting file {out_fn} as {out_format} file...')

            tr.split().write(out_fn, out_format, flush=True)


def _slice_days(stream, extra=None, sampling_precision=2):
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
        Number of decimal digits to validate sampling precision of traces.
        If sampling rate is not the same it will be rounded to the specified
        precision (default is 2) and give a warning.

    Returns
    -------
    st : :class:`~obspy.core.Stream`
        A stream with the sliced traces.
    """
    extra = extra or 0

    if extra < 0:

        raise ValueError('``extra`` must be larger than 0')

    st = Stream()

    try:

        stream.merge(method=1)

    except Exception:

        warnings.warn(
            'Sampling rate was rounded to {} '
            'decimal digits precision.'.format(sampling_precision)
        )

        for trace in stream:

            trace.stats.sampling_rate = round(
                trace.stats.sampling_rate, sampling_precision
            )

        stream.merge(method=1)

    for tr in stream:

        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        delta = tr.stats.delta

        ti = UTCDateTime(year=starttime.year, julday=starttime.julday)

        while ti < endtime:

            tf = ti + 86400.
            st += tr.slice(
                starttime=ti,
                endtime=tf + extra * delta,
                nearest_sample=False
            )
            ti = tf

    return st
