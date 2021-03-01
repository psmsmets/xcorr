r"""

:mod:`util.time` -- Time Utilities
==================================

Utilities for ``xcorr`` for time conversions.

"""


# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from obspy import UTCDateTime


__all__ = ['_one_second', 'to_seconds', 'to_datetime', 'to_UTCDateTime',
           'dataset_timedelta64_to_seconds', 'get_dates', '_dpm',
           'leap_year', 'get_dpm', 'get_dpy', 'update_lag_indices']


_one_second = pd.to_timedelta(1, unit='s')


def to_seconds(time):
    """
    Convert dtype timedelta64[ns] to float seconds

    Parameters
    ----------
    time : any
        Dtype timedelta64[ns].

    Returns
    -------
    time : any
        Dtype float seconds.

    """
    if not hasattr(time, 'dtype'):
        return time
    if time.dtype != np.dtype('timedelta64[ns]'):
        return time
    else:
        return time / _one_second


def to_datetime(time):
    """
    Extends :meth:`pandas.to_datetime` with some more date time format
    conversions.

    Parameters
    ----------
    time : mixed
        A string or various datetime object.

    Returns
    -------
    time : :class:`pandas.Timestamp`
        Pandas datetime object.

    """
    if time is None:
        return
    if isinstance(time, xr.DataArray):
        time = time.values
    elif isinstance(time, object) and hasattr(time, 'datetime'):
        time = time.datetime

    return pd.to_datetime(time)


def to_UTCDateTime(time):
    """
    Convert various datetime formats to obspy UTC-based datetime object.

    Parameters
    ----------
    time : mixed
        A string or various datetime object.

    Returns
    -------
    time : :class:`obspy.UTCDateTime`
        Obspy UTC-based datetime object.

    """
    return UTCDateTime(to_datetime(time))


def dataset_timedelta64_to_seconds(ds: xr.Dataset):
    """Updates dataset variables of type timedelta64[ns] to float64 in seconds
    """
    for var in ds.variables:
        if ds[var].dtype == np.dtype('timedelta64[ns]'):
            ds[var] = ds[var] / _one_second 
            ds[var].attrs['units'] = 's'
    return ds


def update_lag_indices(lag: xr.DataArray):
    """
    """
    for attr in ['sampling_rate', 'delta', 'npts', 'index_min', 'index_max']:
        if attr not in lag.attrs:
            raise KeyError(f'Lag has no attribute "{attr}"!')

    if lag.units != 's':
        raise ValueError('Lag time unit should be seconds.')

    lag_max = (lag.attrs['npts']-1)*lag.attrs['delta']

    srate = lag.attrs['sampling_rate']
    lag.attrs['index_min'] = int(round((lag.values[0]+lag_max)*srate))
    lag.attrs['index_max'] = int(round((lag.values[-1]+lag_max)*srate+1))


def get_dates(start: pd.Timestamp, end: pd.Timestamp):
    """
    Get the dates for the outer span of days from start to end time.

    Parameters:
    -----------
    start : :class:`pd.Timestamp`
        Start date and time.

    end : :class:`pd.Timestamp`
        End date and time.

    Returns:
    --------
    dates : :class:`pandas.DatetimeIndex`
        All dates with days from start to end, with time at midnight.

    """
    return pd.date_range(
        start=start + pd.offsets.DateOffset(0, normalize=True),
        end=end + pd.offsets.DateOffset(1, normalize=True),
        name='days',
        freq='D',
        closed='left'
    )


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
