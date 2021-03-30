r"""

:mod:`util.stream.duration` -- Stream utilities
===============================================

Common stream methods.

"""


# Mandatory imports
from obspy import Stream

__all__ = ['duration']


def duration(stream: Stream, receiver: str = None,
             sampling_rate: float = None):
    """Returns a dictionary with the total duration per receiver SEED-id,
    optionally filtered for a dedicated sampling rate.
    """
    if not isinstance(stream, Stream):
        raise TypeError('``stream`` should be a :class:`obspy.Stream`.')

    if receiver and not isinstance(receiver, str):
        raise TypeError('``receiver`` should be a `str`.')

    if sampling_rate and not isinstance(sampling_rate, float):
        raise TypeError('``sampling_rate`` should be float Hz.')

    duration = dict()

    if receiver:
        duration[receiver] = dict(gaps=[], npts=0, time=0.,
                                  starttime=None, endtime=None)

    for trace in stream:
        if sampling_rate and trace.stats.sampling_rate != sampling_rate:
            continue

        if receiver is not None and trace.id != receiver:
            continue

        if trace.id in duration:
            prev = duration[trace.id]
        else:
            prev = dict(gaps=[], npts=0, time=0.,
                        starttime=None, endtime=None)

        prev['npts'] += trace.stats.npts
        prev['time'] += trace.stats.npts * trace.stats.delta
        if trace.stats.starttime < prev['starttime'] or not prev['starttime']:
            prev['starttime'] = trace.stats.starttime
        if trace.stats.endtime < prev['endtime'] or not prev['endtime']:
            prev['endtime'] = trace.stats.endtime

        duration[trace.id] = prev

    for gap in stream.get_gaps():
        duration['.'.join(gap[:4])]['gaps'] += [gap[4:]]

    for rec in duration:
        npts_overlap = 0
        time_overlap = 0.
        for gap in duration[rec]['gaps']:
            if gap[-1] > 0:
                npts_overlap += gap[-1]
                time_overlap += gap[-2]

        duration[rec]['npts'] += npts_overlap
        duration[rec]['time'] += time_overlap

    return duration[receiver] if receiver else duration
