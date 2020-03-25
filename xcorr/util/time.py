r"""

:mod:`util.time` -- Time Utilities
==================================

Utilities for ``xcorr`` for time conversions.

"""


# Mandatory imports
import numpy as np
from xarray import DataArray
from datetime import datetime
from pandas import to_datetime, to_timedelta
from obspy import UTCDateTime


__all__ = ['_one_second', 'to_seconds', 'to_UTCDateTime',
           '_dpm', 'leap_year', 'get_dpm', 'get_dpy']


_one_second = to_timedelta(1, unit='s')


def to_seconds(time):
    r"""Convert dtype timedelta64[ns] to float seconds

    Parameters
    ----------
    time : any
        Dtype timedelta64[ns].

    Returns
    -------
    time : any
        Dtype float seconds.

    """
    if time.dtype == np.dtype('timedelta64[ns]'):
        return time / _one_second
    else:
        return time


def to_UTCDateTime(time):
    r"""Convert various datetime formats to obspy UTC-based datetime object.

    Parameters
    ----------
    time : mixed
        A string or various datetime object.

    Returns
    -------
    time : :class:`obspy.UTCDateTime`
        Obspy UTC-based datetime object.

    """
    if isinstance(time, UTCDateTime):
        return time
    elif isinstance(time, str) or isinstance(time, datetime):
        return UTCDateTime(time)
    elif isinstance(time, np.datetime64):
        return UTCDateTime(to_datetime(time))


_calendars = ['standard', 'gregorian', 'proleptic_gregorian', 'julian']


_d360 = [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
_d365 = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_d366 = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


_dpm = {
    'noleap': _d365,
    '365_day': _d365,
    'standard': _d365,
    'gregorian': _d365,
    'proleptic_gregorian': _d365,
    'all_leap': _d366,
    '366_day': _d366,
    '360_day': _d360,
}


def leap_year(year: int, cal: str = 'standard'):
    r"""Determine if year is a leap year

    Parameters
    ----------
    year : `int`
        Integer year.

    cal : `str`, {'standard', 'gregorian', 'proleptic_gregorian', 'julian'}
        Type of calendar specifying the leap year rule. Defaults to 'standard'.

    Returns
    -------
    leap : `bool`
        `True` if year is a leap year, otherwise `False`.

    """
    leap = False
    if (cal in _calendars) and (year % 4 == 0):
        leap = True
        if (
            (cal == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)
        ):
            leap = False
        elif (
            (cal in ['standard', 'gregorian']) and
            (year % 100 == 0) and (year % 400 != 0) and
            (year < 1583)
        ):
            leap = False
    return leap


def get_dpm(time: DataArray, cal: str = 'standard'):
    r"""Return the number of days per month.

    Parameters
    ----------
    time : :class:`xarray.DataArray`
        Input time object.

    cal : `str`, {'standard', 'gregorian', 'proleptic_gregorian', 'julian'}
        Type of calendar specifying the leap year rule. Defaults to 'standard'.

    Returns
    -------
    dpm : :class:`xarray.DataArray`
        Days per month for each element in ``time``.

    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = _dpm[cal]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=cal):
            month_length[i] += 1
    return month_length


def get_dpy(
    time: DataArray, cal: str = 'standard'
):
    r"""Return the number of days per year.

    Parameters
    ----------
    time : :class:`xarray.DataArray`
        Input time object.

    cal : `str`, {'standard', 'gregorian', 'proleptic_gregorian', 'julian'}
        Type of calendar specifying the leap year rule. Defaults to 'standard'.

    Returns
    -------
    dpy : :class:`xarray.DataArray`
        Days per year for each element in ``time``.

    """
    year_length = np.zeros(len(time), dtype=np.int)

    for i, year in enumerate(time.year):
        year_length[i] = 365
        if leap_year(year, calendar=cal):
            year_length[i] += 1
    return year_length
