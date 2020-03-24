r"""

:mod:`util.time` -- Time Utilities
==================================

Utilities for ``ccf`` for time conversions.

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


def leap_year(year: int, calendar: str = 'standard'):
    r"""Determine if year is a leap year

    Parameters
    ----------
    year : int
        Integer year.

    calendar : str, {'standard', 'gregorian', 'proleptic_gregorian', 'julian'}
        Type of calendar specifying the leap year rule. If empty,
        `calendar` = 'standard' (default).

    Returns
    -------
    leap : bool
        True if year is a leap year, else False.

    """
    leap = False
    if (calendar in _calendars) and (year % 4 == 0):
        leap = True
        if (
            (calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ['standard', 'gregorian']) and
            (year % 100 == 0) and (year % 400 != 0) and
            (year < 1583)
        ):
            leap = False
    return leap


def get_dpm(time: DataArray, calendar: str = 'standard'):
    r"""Determine if year is a leap year

    Parameters
    ----------
    year : int
        Integer year.

    calendar : str, {'standard', 'gregorian', 'proleptic_gregorian', 'julian'}
        Type of calendar specifying the leap year rule. If empty,
        `calendar` = 'standard' (default).

    Returns
    -------
    leap : bool
        True if year is a leap year, else False.

    """

    """
    Return an array of days per month corresponding to the months
    provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = _dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def get_dpy(
    time: DataArray, calendar: str = 'standard'
):
    """
    Return an array of days per year corresponding to the years
    provided in `years`
    """
    year_length = np.zeros(len(time), dtype=np.int)

    for i, year in enumerate(time.year):
        year_length[i] = 365
        if leap_year(year, calendar=calendar):
            year_length[i] += 1
    return year_length
