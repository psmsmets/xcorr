# -*- coding: utf-8 -*-
"""
Python module with various crosscorrelation helper functions.

.. module:: helpers 

:author:
    Pieter Smets (P.S.M.Smets@tudelft.nl)

:copyright:
    Pieter Smets

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import numpy as np
import xarray as xr
import pandas as pd
import re
from obspy import UTCDateTime, Inventory
from pyproj import Geod


class Helpers:

    one_second = pd.to_timedelta(1, unit='s')
    regex_seed_id = (
        r'^([A-Z]{2})\.([A-Z,0-9]{3,5})\.([0-9]{0,2})\.([A-Z]{2}[0-9,A-Z]{1})'
    )
    regex_seed_id_wildcards = (
        r'^([A-Z,?*]{1,2})\.([A-Z,0-9,?*]{1,5})\.'
        r'([0-9,?*]{0,2})\.([0-9,A-Z,?*]{1,3})'
    )

    def to_seconds(time):
        """
        Convert timedelta64[ns] object to seconds
        """
        if time.dtype == np.dtype('timedelta64[ns]'):
            return time / Helpers.one_second
        else:
            return time

    def to_UTCDateTime(datetime):
        """
        Convert various datetime formats to `obspy.UTCDateTime`
        """
        if isinstance(datetime, UTCDateTime):
            return datetime
        elif isinstance(datetime, str) or isinstance(datetime, pd.datetime):
            return UTCDateTime(datetime)
        elif isinstance(datetime, np.datetime64):
            return UTCDateTime(pd.to_datetime(datetime))

    def verify_receiver(
        receiver: str, allow_wildcards: bool = False, raise_error: bool = False
    ):
        """
        Verify if the receiver string matche the SEED-id regex pattern and
        optionally verify if it contains wildcards (by default allowd).
        """
        if allow_wildcards is False:
            if '*' in receiver or '?' in receiver:
                if raise_error:
                    raise ValueError(
                        'Receiver SEED-id cannot contain wildcards (? or *)! '
                        'Be specific.'
                    )
                return False
            if not re.match(Helpers.regex_seed_id, receiver):
                if raise_error:
                    raise ValueError(
                        'Receiver SEED-id is not of valid format '
                        '"network.station.location.channel".'
                    )
                return False
        else:
            if not re.match(Helpers.regex_seed_id_wildcards, receiver):
                if raise_error:
                    raise ValueError(
                        'Receiver SEED-id is not of valid format '
                        '"network.station.location.channel".'
                    )
                return False
        return True

    def split_pair(pair, separator: str = '-', split_receiver: bool = False):
        """
        Split a receiver SEED-ids `pair` string.
        """
        if isinstance(pair, xr.DataArray):
            pair = str(pair.values)
        elif isinstance(pair, np.ndarray):
            pair = str(pair)
        assert isinstance(pair, str), (
            'Pair should be either a string, numpy.ndarray or '
            'an xarray.DataArray'
        )

        return (
            [Helpers.split_seed_id(p) for p in pair.split(separator)]
            if split_receiver else pair.split(separator)
        )

    def split_receiver(receiver: str):
        """
        Split a receiver SEED-id string to a dictionary.
        """
        return dict(zip(
            ['network', 'station', 'location', 'channel'],
            receiver.split('.')
        ))

    def merge_receiver(receiver: dict):
        """
        Merge a receiver SEED-id dictionary to a string.
        """
        return '{net}.{sta}.{loc}.{cha}'.format(
            net=receiver['network'],
            sta=receiver['station'],
            loc=receiver['location'] if receiver['location'] else '',
            cha=receiver['channel'],
        )

    def get_pair_inventory(pair, inventory: Inventory):
        """
        Return a filtered inventory given receiver SEED-ids `pair` and
        `inventory`.
        """
        if isinstance(pair, xr.DataArray):
            pair = str(pair.values)
        assert isinstance(pair, str), (
            'Pair should be either a string or a xarray.DataArray'
        )

        r = Helpers.split_pair(pair)
        return inventory.select(**r[0]) + inventory.select(**r[1])

    def get_receiver_coordinates(receiver: str, inventory: Inventory):
        """
        Return a dictionary with the extracted coordinates of the receiver from
        the ~obspy.Inventory.
        """
        if receiver[-1] == 'R' or receiver[-1] == 'T':
            receiver = receiver[:-1] + 'Z'
        return inventory.get_coordinates(receiver)

    def get_pair_distance(
        pair, inventory: Inventory, ellipsoid: str = 'WGS84', poi: dict = None,
        km: bool = True
    ):
        """
        Calculate the receiver pair distance. Optionally, specify the ellipsoid
        (default = WGS84), or specify a point-of-interest (a dictionary with
        longitude and latitude in decimal degrees) to obtain a relative
        distance.
        """
        g = Geod(ellps=ellipsoid)

        r = Helpers.split_pair(pair)
        c = [Helpers.get_receiver_coordinates(i, inventory) for i in r]

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
