r"""

:mod:`utils` -- Utilites
========================

Utilities for ``ccf`` such as time conversions and receiver (pair) checks
and operations.
This module consists of two groups of functions in a single class:
   1. Time related
   2. Receiver related

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import re
from obspy import UTCDateTime, Inventory
from pyproj import Geod


class Utils:

    # 1. Time related

    _one_second = pd.to_timedelta(1, unit='s')

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
            return time / Utils._one_second
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
        elif (
            isinstance(time, str) or
            isinstance(time, datetime.datetime) or
            isinstance(time, pd.datetime)
        ):
            return UTCDateTime(time)
        elif isinstance(time, np.datetime64):
            return UTCDateTime(pd.to_datetime(time))

    # 2. Receiver related

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
            if not re.match(Utils._regex_seed_id, receiver):
                if raise_error:
                    raise ValueError(
                        'Receiver SEED-id is not of valid format '
                        '"network.station.location.channel".'
                    )
                return False
        else:
            if not re.match(Utils._regex_seed_id_wildcards, receiver):
                if raise_error:
                    raise ValueError(
                        'Receiver SEED-id is not of valid format '
                        '"network.station.location.channel".'
                    )
                return False
        return True

    def split_pair(pair, separator: str = '-', todict: bool = False):
        r"""Split a receiver pair string into two SEED-id strings.

        Parameters
        ----------
        pair : str or :mod:`~xarray.DataArray`
            Receiver pair couple separated by `separator`.
            Each receiver is specified by a SEED-id string:
            '{network}.{station}.{location}.{channel}'.

        separator : str, optional
            Receiver pair separator: '-' (default).

        todict : bool, optional
            Return the SEED-id string if False (default), else split each
            each receiver SEED-id string into a dict using :mod:.


        Returns
        -------

        pair : list
            Two-element list of SEED-id strings if `split_receiver` is
            False (default), else a two-element list of SEED-id dictionaries.

        """
        if isinstance(pair, xr.DataArray):
            pair = pair.str
        elif isinstance(pair, np.ndarray):
            pair = str(pair)
        assert isinstance(pair, str), (
            'Pair should be either a string, numpy.ndarray or '
            'an xarray.DataArray'
        )

        return (
            [Utils.receiver_todict(p) for p in pair.split(separator)]
            if todict else pair.split(separator)
        )

    def receiver_todict(receiver: str):
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

    def receiver_tostr(receiver: dict):
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
        r = Utils.split_pair(pair, separator)
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

        r = Utils.split_pair(pair, separator)
        c = [Utils.get_receiver_coordinates(i, inventory) for i in r]

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
