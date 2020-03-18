# -*- coding: utf-8 -*-
"""
Python module for loading waveform data from a primary local sds archive, and
automatically retrieve missing daily waveforms using fdsn and nms web
request services.

.. module:: clients

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
import pandas as pd
import json

from obspy import UTCDateTime, Stream, Inventory
from ccf.process import Preprocess
from ccf.helpers import Helpers

from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient

from ccf.datafetch import stream2SDS
try:
    from nms_tools.nms_client import Client as nmsClient
except ImportError:
    nmsClient = None  # make it work without nmsClient


# client messages
msg_no_data = 'No waveform data found for {}.'
msg_load_archive = 'Get waveform data for {}.'
msg_loaded_archive = 'Waveform data for {} loaded from archive.'
msg_added_archive = 'Waveform data for {} downloaded and added to archive.'


class Clients:

    sds_root = None
    sds = None
    fdsn = None
    nms = None

    def check(
        raise_error: bool = False
    ):
        """
        Verify if the client globals are set.
        """
        error = (
            Clients.sds_root is None or Clients.sds is None or
            Clients.fdsn is None or Clients.nms is None
        )
        if error and raise_error:
            raise RuntimeError(
                'Clients not yet set! Run Clients.set(..) first!'
            )
        return not error

    def set(
        sds_root: str, fdsn_base_url: str = 'IRIS', tmp_sds_root: str = None,
        **kwargs
    ):
        """
        Set the client globals `sds_root`.
        Optionally change the `fdsn_base_url` (default 'IRIS')
        """
        Clients.sds_root = sds_root
        Clients.sds = sdsClient(sds_root)
        Clients.fdsn = fdsnClient(fdsn_base_url, **kwargs)
        Clients.nms = nmsClient()

    def get_waveforms(
        receiver: str, time: np.datetime64, centered: bool = True,
        duration: float = 86400., buffer: float = 60.,
        allow_wildcards: bool = False, verbose: bool = False
    ):
        """
        Get waveforms given the SEED-id `receiver` and
        `time` (default `centered`)
        for `duration` (default 86400s) and `buffer` (default 60s).
        """
        # check if clients are set
        Clients.check(raise_error=True)

        # check if receiver SEED-id is valid
        Helpers.verify_receiver(
            receiver,
            allow_wildcards=allow_wildcards,
            raise_error=True
        )

        # split receiver SEED-id
        network, station, location, channel = receiver.split('.')

        # center time of 24h window -12h
        t0 = UTCDateTime(pd.to_datetime(time))
        if centered:
            t0 -= duration/2
        t1 = t0 + duration
        if buffer > 0.:
            t0 -= buffer
            t1 += buffer
        if verbose:
            print(
                'Get waveforms for ',
                network,
                station,
                location,
                channel,
                t0,
                t1,
            )

        stream = Stream()
        for day in pd.date_range(
            start=UTCDateTime(t0).date,
            end=(UTCDateTime(t1)+43200).date,
            name='days',
            freq='D'
        ):
            t = UTCDateTime(day)

            if verbose:
                print(msg_load_archive.format(t))

            daystream = Clients.sds.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=t,
                endtime=t + 86400,
            )
            if Clients.daystream_length_passed(daystream, verbose):
                if verbose:
                    print(msg_loaded_archive.format(t))
                stream += daystream
                continue
            if verbose:
                print(msg_no_data.format(t))

            # get attempt via fdsn
            if Clients.fdsn:
                if verbose:
                    print('Try FDSN.')
                try:
                    daystream = Clients.fdsn.get_waveforms(
                        network=network,
                        station=station,
                        location=location,
                        channel=channel,
                        starttime=t,
                        endtime=t + 86400,
                    )
                    if Clients.daystream_length_passed(daystream, verbose):
                        stream2SDS(
                            daystream,
                            sds_path=Clients.sds_root,
                            force_override=True,
                            verbose=False
                        )
                        if verbose:
                            print(msg_added_archive.format(t))
                        stream += daystream
                        continue
                except (KeyboardInterrupt, SystemExit):
                    raise
                except FDSNNoDataException:
                    if verbose:
                        print(msg_no_data.format(t))
                except Exception as e:
                    if verbose:
                        print('an error occurred:')
                        print(e)

            # get attempt via nms
            if Clients.nms:
                if verbose:
                    print('Try NMS_Client')
                try:
                    daystream = Clients.nms.get_waveforms(
                        starttime=t,
                        station=station,
                        channel=channel,
                        verbose=False,
                    )
                    if Clients.daystream_length_passed(daystream, verbose):
                        stream2SDS(
                            daystream,
                            sds_path=Clients.sds_root,
                            force_override=True,
                            verbose=False
                        )
                        if verbose:
                            print(msg_added_archive.format(t))
                        stream += daystream
                        continue
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    if verbose:
                        print('an error occurred:')
                        print(e)

        return stream.trim(starttime=t0, endtime=t1)

    def daystream_length_passed(
        stream: Stream, verbose: bool = False, max_gap: float = 300.
    ):
        """
        Return if a stream (assuming a uniqe SEED-id) contains a day of data
        not exceeding the allowed `gap` (default 300s.)
        """
        if not isinstance(stream, Stream) or len(stream) == 0:
            return False
        npts_day = int(stream[0].stats.sampling_rate * 86400)
        npts_gap = int(stream[0].stats.sampling_rate * max_gap)
        npts_str = sum([trace.stats.npts for trace in stream])
        if verbose:
            print(
                'Samples in day = {}, samples in stream = {}, max gaps = {}.'
                .format(npts_day, npts_str, npts_gap)
            )
        return npts_str >= (npts_day - npts_gap)

    def get_preprocessed_pair_stream(
        pair, time: np.datetime64, operations: dict, **kwargs
    ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-ids receiver
        `pair`, `time` and preprocess `operations`.
        """
        rA, rB = Helpers.split_pair(pair)
        return (
            Clients.get_preprocessed_stream(rA, time, operations, **kwargs) +
            Clients.get_preprocessed_stream(rB, time, operations, **kwargs)
        )

    def get_preprocessed_stream(
        receiver: str, time: np.datetime64, operations: dict,
        duration: float = 86400., inventory: Inventory = None,
        operations_from_json: bool = False, three_components: str = '12Z',
        verbose: bool = False, debug: bool = False, **kwargs
    ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-id `receiver`,
        `time`, preprocess `operations` and `duration` (default 86400s).
        Optionally provide the `obspy.Inventory` and some other options.
        """
        # check if receiver SEED-id is valid
        Helpers.verify_receiver(
            receiver, allow_wildcards=False, raise_error=True
        )

        t0 = Helpers.to_UTCDateTime(time) - duration/2
        t1 = t0 + duration
        ch = receiver.split('.')[-1]

        # radial or transverse component? Request all channels manually.
        if ch[-1] == 'R' or ch[-1] == 'T':
            st = Stream()
            for c in three_components:
                st += Clients.get_waveforms(
                    receiver=receiver[:-1]+c,
                    time=time,
                    duration=duration,
                    centered=True,
                    verbose=debug,
                    **kwargs
                )
        else:
            st = Clients.get_waveforms(
                receiver=receiver,
                time=time,
                duration=duration,
                centered=True,
                verbose=debug,
                **kwargs
            )

        if verbose:
            print(st)
        if not isinstance(st, Stream) or len(st) == 0:
            return Stream()
        try:
            st = Preprocess.preprocess(
                stream=st,
                operations=(
                    json.loads(operations[ch])
                    if operations_from_json else operations[ch]
                ),
                inventory=inventory,
                starttime=t0,
                endtime=t1,
                verbose=verbose,
                debug=debug,
            )
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            if verbose:
                print('an error occurred:')
                print(e)
            return Stream()
        if not isinstance(st, Stream) or len(st) != 1:
            return Stream()
        if st[0].stats.npts * st[0].stats.delta < duration:
            return Stream()
        return st
