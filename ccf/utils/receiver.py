r"""

:mod:`utils.receiver` -- Receiver Utilities
===========================================

Receiver utilities for ``ccf`` such as checks, type conversions,
and receiver pair geodetic operations.

"""

# Mandatory imports
from xarray import DataArray
from numpy import ndarray
from re import match
from obspy import Inventory
from pyproj import Geod


__all__ = ['check_receiver', 'split_pair', 'receiver_to_dict',
           'receiver_to_str', 'get_pair_inventory',
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
    receiver : str
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    allow_wildcards : bool, optional
        Allow * or ? wildcards in the SEED-id string, else wildcards are
        not allowed (default).

    raise_error : bool, optional
        Raise a ValueError when failing the regular expression,
        else return False (default).

    Returns
    -------
    match : bool
        True if the receiver string matches the SEED-id pattern,
        else False.

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


def split_pair(pair, separator: str = '-', to_dict: bool = False):
    r"""Split a receiver pair string into two SEED-id strings.

    Parameters
    ----------
    pair : str or :mod:`~xarray.DataArray`
        Receiver pair couple separated by `separator`.
        Each receiver is specified by a SEED-id string:
        '{network}.{station}.{location}.{channel}'.

    separator : str, optional
        Receiver pair separator: '-' (default).

    to_dict : bool, optional
        Return the SEED-id string if False (default), else split each
        each receiver SEED-id string into a dict using :mod:.

    Returns
    -------
    pair : list
        Two-element list of SEED-id strings if `split_receiver` is
        False (default), else a two-element list of SEED-id dictionaries.

    """
    if isinstance(pair, DataArray):
        pair = pair.str
    elif isinstance(pair, ndarray):
        pair = str(pair)
    assert isinstance(pair, str), (
        'Pair should be either a string, numpy.ndarray or '
        'an xarray.DataArray'
    )

    return (
        [receiver_to_dict(p) for p in pair.split(separator)]
        if to_dict else pair.split(separator)
    )


def receiver_to_dict(receiver: str):
    r"""Split a receiver SEED-id string into a dictionary.

    Parameters
    ----------
    receiver : str
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    Returns
    -------
    receiver : dict
        Receiver dict with SEED-id keys:
        ['network', 'station', 'location', 'channel'].

    """
    return dict(zip(
        ['network', 'station', 'location', 'channel'],
        receiver.split('.')
    ))


def receiver_to_str(receiver: dict):
    r"""Merge a receiver SEED-id dictionary into a string.

    Parameters
    ----------
    receiver : dict
        Receiver dict with SEED-id keys:
        ['network', 'station', 'location', 'channel'].

    Returns
    -------
    receiver : str
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
    receiver : str or dict
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'
        or dict with keys ['network', 'station', 'location', 'channel'].

    Returns
    -------
    channel : str 
        Receiver channel code.

    """
    if isinstance(receiver, dict):
        return receiver['channel']
    elif isinstance(receiver, str):
        return receiver.split('.')[3]
    else:
        raise TypeError('Receiver should be of type `str` or `dict`!')


def get_pair_inventory(pair, inventory: Inventory, separator: str = '-'):
    r"""Filter the obspy inventory object for a receiver pair.

    Parameters
    ----------
    pair : str or :class:`xarray.DataArray`
        Receiver pair couple separated by `separator`.
        Each receiver is specified by a SEED-id string:
        '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

        separator : str, optional
        Receiver pair separator: '-' (default).

    Returns
    -------
    inventory : :class:`obspy.Inventory`
        Returns a new obspy inventory object for the receiver pair.

    """
    r = split_pair(pair, separator)
    return inventory.select(**r[0]) + inventory.select(**r[1])


def get_receiver_coordinates(receiver: str, inventory: Inventory):
    r"""Retrieve the receiver coordinates from the obspy inventory.

    Parameters
    ----------
    receiver : str
        Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

    Returns
    -------
    coordinates : dict
        The extracted receiver coordinates from the inventory object.

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
    pair : str or :class:`xarray.DataArray`
        Receiver pair couple separated by `separator`.
        Each receiver is specified by a SEED-id string:
        '{network}.{station}.{location}.{channel}'.

    inventory : :class:`obspy.Inventory`
        Inventory object.

    ellipsoid : str, optional
        Specify the ellipsoid for :class:`pyproj.Geod`:
        `ellipsoid`='WGS84' (default).

    poi : dict, optional
        Specify a point-of-interest dict with keys ['longitude','latitude']
        in decimal degrees to obtain a relative distance.

    separator : str, optional
        Receiver pair separator: '-' (default).

    km : bool, optional
        Return the geodetic distance in kilometre if True (default),
        else in metre.

    Returns
    -------
    d : float
        Distance in kilometre if `km` is True (default), else in metre.

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
