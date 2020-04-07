r"""

:mod:`util.receiver` -- Receiver Utilities
==========================================

Receiver utilities for ``xcorr`` such as checks, type conversions,
and receiver pair geodetic operations.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
from re import match
from obspy import Inventory
from pyproj import Geod


# Relative imports
from ..util.time import to_UTCDateTime


__all__ = ['check_receiver', 'split_pair', 'receiver_to_dict',
           'receiver_to_str', 'get_receiver_channel', 'get_pair_inventory',
           'get_receiver_coordinates', 'get_pair_distance']


_regex_seed_id = (
    r'^([A-Z]{2})\.([A-Z,0-9]{3,5})\.([0-9]{0,2})\.([A-Z]{2}[0-9,A-Z]{1})'
)
_regex_seed_id_wildcards = (
    r'^([A-Z,?*]{1,2})\.([A-Z,0-9,?*]{1,5})\.'
    r'([0-9,?*]{0,2})\.([0-9,A-Z,?*]{1,3})'
)


def check_receiver(
    receiver: str, allow_wildcards: bool = False, raise_error: bool = False
):
    r"""Check receiver SEED-id string.

    Check if the receiver string matches the SEED-id regex pattern.

    This check-function is called by routines in :mod:`clients`.

    Parameters
    ----------
    receiver : `str`
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    allow_wildcards : `bool`, optional
        Allow * or ? wildcards in the SEED-id string, else wildcards are
        not allowed (default).

    raise_error : `bool`, optional
        Raise a `ValueError` when failing the regular expression,
        else returns `False` (default).

    Returns
    -------
    match : `bool`
        `True` if the receiver string matches the SEED-id pattern,
        else `False`.

    """
    if allow_wildcards is False:
        if '*' in receiver or '?' in receiver:
            if raise_error:
                raise ValueError(
                    'Receiver SEED-id cannot contain wildcards (? or *)! '
                    'Be specific.'
                )
            return False
        if not match(_regex_seed_id, receiver):
            if raise_error:
                raise ValueError(
                    'Receiver SEED-id is not of valid format '
                    '"network.station.location.channel".'
                )
            return False
    else:
        if not match(_regex_seed_id_wildcards, receiver):
            if raise_error:
                raise ValueError(
                    'Receiver SEED-id is not of valid format '
                    '"network.station.location.channel".'
                )
            return False
    return True


def split_pair(
    pair, separator: str = '-', substitute: bool = False,
    three_components: str = None, to_dict: bool = False
):
    r"""Split a receiver pair string into receivers SEED-ids.

    Parameters
    ----------
    pair : `str` or :mod:`~xarray.DataArray`
        Receiver couple separated by ``separator``. Each receiver is specified
        by a SEED-id string: '{network}.{station}.{location}.{channel}'.

    separator : `str`, optional
        Receiver pair separator:, defaults to '-'.

    substitute : `bool`, optional
        If `True`, convert radial 'R' and transverse 'T' rotated orientation
        codes automatically to ``three_components``. Defaults to `False`.

    three_components: {'12Z', 'NEZ'}, optional
        Set the three-component orientation characters for ``substitute``.
        Defaults to '12Z'.

    to_dict : `bool`, optional
        Return the SEED-id string if `False` (default). If `True`
        each receiver SEED-id string is converted into a dictionary using
        :func:`receiver_to_dict`.

    Returns
    -------
    receivers : `list`
        A list of SEED-id strings if `to_dict` is `False` (default), else a
        list of dictionaries. If separator is not found a single element is
        returned (a receiver). If ``substitute`` is `True`, the list can
        contain up to six elements (two times three substituted channels).

    """
    if isinstance(pair, xr.DataArray):
        pair = str(pair.values)
    elif isinstance(pair, np.ndarray):
        pair = str(pair)
    assert isinstance(pair, str), (
        'Pair should be either a string, numpy.ndarray or '
        'an xarray.DataArray'
    )
    three_components = three_components or '12Z'
    assert three_components == '12Z' or three_components == 'NEZ', (
        '``three_components`` should be either "12Z" or "NEZ"!'
    )

    # list of receivers
    receivers = pair.split(separator)

    # substitute R and T to 12Z or NEZ
    if substitute:
        tmp = []
        for r in receivers:
            if r[-1] in 'RT':
                tmp += [r[:-1] + c for c in three_components]
            else:
                tmp += [r]
        receivers = tmp

    return [receiver_to_dict(r) for r in receivers] if to_dict else receivers


def receiver_to_dict(receiver: str):
    r"""Split a receiver SEED-id string into a dictionary.

    Parameters
    ----------
    receiver : `str`
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    Returns
    -------
    receiver : `dict`
        Receiver dict with SEED-id keys:
        ['network', 'station', 'location', 'channel'].

    """
    if isinstance(receiver, xr.DataArray):
        receiver = str(receiver.values)
    elif isinstance(receiver, np.ndarray):
        receiver = str(receiver)
    assert isinstance(receiver, str), (
        'Receiver should be either a string, numpy.ndarray or '
        'an xarray.DataArray'
    )
    return dict(zip(
        ['network', 'station', 'location', 'channel'],
        receiver.split('.')
    ))


def receiver_to_str(receiver: dict):
    r"""Merge a receiver SEED-id dictionary into a string.

    Parameters
    ----------
    receiver : `dict`
        Receiver dictionary with SEED-id keys:
        ['network', 'station', 'location', 'channel'].

    Returns
    -------
    receiver : `str`
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    """
    return '{net}.{sta}.{loc}.{cha}'.format(
        net=receiver['network'],
        sta=receiver['station'],
        loc=receiver['location'] if receiver['location'] else '',
        cha=receiver['channel'],
    )


def get_receiver_channel(receiver):
    r"""Split a receiver SEED-id string into a dictionary.

    Parameters
    ----------
    receiver : `str` or `dict`
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'
        or dictionary with keys ['network', 'station', 'location', 'channel'].

    Returns
    -------
    channel : `str`
        Receiver channel code.

    """
    if isinstance(receiver, dict):
        return receiver['channel']
    elif isinstance(receiver, str):
        return receiver.split('.')[3]
    else:
        raise TypeError('``receiver`` should be of type `str` or `dict`!')


def get_pair_inventory(
    pair, inventory: Inventory, times=None, separator: str = '-'
):
    r"""Filter the obspy inventory object for a receiver pair.

    Parameters
    ----------
    pair : `str` or :class:`xarray.DataArray`
        Receiver couple separated by ``separator``. Each receiver is specified
        by a SEED-id string: '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

    times : `pd.DatetimeIndex` or `xarray.DataArray`

    separator : `str`, optional
        Receiver pair separator:, defaults to '-'.

    Returns
    -------
    inventory : :class:`obspy.Inventory`
        Returns a new obspy inventory object filtered for the
        receiver ``pair``.

    """
    if times is not None:
        if not (
            isinstance(times, pd.DatetimeIndex) or
            isinstance(times, xr.DataArray)
        ):
            raise TypeError(
                '``time`` should be of type `pandas.DatetimeIndex` '
                'or `xarray.DataArray`!'
            )
        t0 = to_UTCDateTime(times[0])
        t1 = to_UTCDateTime(times[-1])
    else:
        t0 = None
        t1 = None
    if isinstance(pair, list) or isinstance(pair, xr.DataArray):
        inv = Inventory([], [])
        rr = []
        for p in pair:
            rr += split_pair(p, separator, to_dict=False)
        for r in set(rr):
            d = receiver_to_dict(r)
            if d['channel'][-1] == 'R':
                for h in ['1', '2', 'N', 'E']:
                    c = d['channel'][:-1] + h
                    inv += inventory.select(
                        **{**d, 'channel': c}, starttime=t0, endtime=t1
                    )
            else:
                inv += inventory.select(**d, starttime=t0, endtime=t1)
        return inv
    else:
        r = split_pair(pair, separator, to_dict=True)
        return (
            inventory.select(**r[0], starttime=t0, endtime=t1) +
            inventory.select(**r[1], starttime=t0, endtime=t1)
        )


def get_receiver_coordinates(receiver: str, inventory: Inventory):
    r"""Retrieve the receiver coordinates from the obspy inventory.

    Parameters
    ----------
    receiver : `str`
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

    Returns
    -------
    coordinates : `dict`
        The extracted receiver coordinates from the ``inventory`` object.

    """
    if receiver[-1] == 'R' or receiver[-1] == 'T':
        receiver = receiver[:-1] + 'Z'
    return inventory.get_coordinates(receiver)


def get_pair_distance(
    pair, inventory: Inventory, ellipsoid: str = 'WGS84', poi: dict = None,
    separator: str = '-', km: bool = True
):
    r"""Calculate the receiver pair geodetic distance.

    Parameters
    ----------
    pair : `str` or :class:`xarray.DataArray`
        Receiver couple separated by ``separator``. Each receiver is specified
        by a SEED-id string: '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

    ellipsoid : `str`, optional
        Specify the ellipsoid for :class:`pyproj.Geod`. Defaults to 'WGS84'.

    poi : `dict`, optional
        Specify a point-of-interest `dict` with keys ['longitude','latitude']
        in decimal degrees to obtain a relative distance.

    separator : `str`, optional
        Receiver pair separator, defaults to '-'.

    km : `bool`, optional
        Return the geodetic distance in kilometre if `True` (default),
        else in metre.

    Returns
    -------
    d : `float`
        Distance in kilometre if ``km`` is `True` (default), else in metre.

    """
    g = Geod(ellps=ellipsoid)

    r = split_pair(pair, separator)
    c = [get_receiver_coordinates(i, inventory) for i in r]

    if poi:
        az12, az21, d0 = g.inv(
            poi['longitude'], poi['latitude'],
            c[0]['longitude'], c[0]['latitude']
        )
        az12, az21, d1 = g.inv(
            poi['longitude'], poi['latitude'],
            c[1]['longitude'], c[1]['latitude']
        )
        d = d0 - d1
    else:
        az12, az21, d = g.inv(
            c[0]['longitude'], c[0]['latitude'],
            c[1]['longitude'], c[1]['latitude']
        )
    return d*1e-3 if km else d