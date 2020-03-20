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

# Mandatory imports
import numpy as np
import pandas as pd
import json
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient
try:
    from nms_tools.nms_client import Client as nmsClient
except ImportError:
    nmsClient = None  # make it work without nmsClient


# Relative imports
from ..clients.datafetch import stream2SDS
from ..preprocess import preprocess
from ..utils.receiver import check_receiver, split_pair
from ..utils.time import to_UTCDateTime


__all__ = []


# client messages
_msg_no_data = 'No waveform data found for {}.'
_msg_load_archive = 'Get waveform data for {}.'
_msg_loaded_archive = 'Waveform data for {} loaded from archive.'
_msg_added_archive = 'Waveform data for downloaded and added to archive.'


class Client(object):
    """
    CCF waveform request client.
    For details see the :meth:`~ccf.clients.client.Client.__init__()`
    method.
    """
    def __init__(
        self, sds_root: str = None, sds_root_write: str = None,
        sds_root_read: list = None, fdsn_service='IRIS',
        nms_service: bool = True, max_gap: float = None
    ):
        r"""Initialize a CCF waveform request client.

        >>> client = Client('/tmp')

        Parameters
        ----------
        sds_root : str
            Path to the local SeisComP Data Structure (SDS) archive. All
            downloaded waveforms will be added to this archive if
            `sds_root_write` is not specified.

        sds_root_write: str, optional
            Specify an SDS archive to store automatically downloaded
            waveforms by the FDSN web service and/or NMS Client.

        sds_root_read: list, optional
            Scan multiple SDS archives.

        fdsn_service : str or :class:`obspy.clients.fdsn.Client`, optional
            FDSN web service base URL. If None, 'IRIS' is used (default).

        nms_service : bool, optional
            Enable the NMS Client to access IMS waveform data of the
            Comprehensive Nuclear-Test-Ban Treaty Organization (CTBTO)
            if True (default). If no :class:`nms_tools.nms_client.Client`
            is found `nms_service` will be False.

        max_gap : float, optional
            Specify the maximum time gap in seconds that is allowed in a day.
            If None, `max_gap` is 300s (default).

        """
        if not sds_root and not sds_root_read or not sds_root and not sds_root_write:
            raise AttributeError(
                'At least `sds_root` or `sds_root_read` and '
                '`sds_root_write` are required.'
            )
        self._sds_root_write = sds_root_write if sds_root_write else sds_root
        self._sds_root_read = (sds_root_read if sds_root_read else [sds_root])
        if sds_root_write:
            self._sds_root_read += [sds_root_write]
        else:
            self._sds_root_read += [sds_root]
        self._sds_root_read = list(set(self._sds_root_read))

        self._sds_read = []
        for _sds_root_read in self._sds_root_read:
            self._sds_read.append(sdsClient(_sds_root_read))

        if isinstance(fdsn_service, str):
            self._fdsn = fdsnClient(fdsn_service)
        elif isinstance(fdsn_service, fdsnClient):
            self._fdsn = fdsn_service
        else:
            raise TypeError(
                '`fdsn_service` should be of type str or '
                ':class:`obspy.clients.fdsn.Client`!'
            )
        self._nms = nmsClient() if nms_service and nmsClient else False
        self._max_gap = abs(max_gap or 300.)

    @property
    def sds_root(self):
        raise NotImplementedError('Use either `sds_root_read` or `sds_root_write`!')

    @property
    def sds_root_read(self):
        return self._sds_root_read

    @property
    def sds_root_write(self):
        return self._sds_root_write

    @property
    def sds_read(self):
        return self._sds_read

    @property
    def sds_write(self):
        return self._sds_write

    @property
    def fdsn(self):
        return self._fdsn

    @property
    def nms(self):
        return self._nms

    @property
    def max_gap(self):
        return self._max_gap


    def _sds_write_daystream(stream: Stream, verbose: bool = False):
        """
        Write daystream to `self.sds_root_write` if `stream` passes
        :func:`_check_daystream_length`.
        """
        if not self._check_daystream_length(stream, verbose):
            return False
        stream2SDS(
            stream,
            sds_path=self.sds_root_write,
            force_override=True,
            verbose=False
        )
        if verbose:
            print(_msg_added_archive)
        return True

    def _check_daystream_length(
        self, stream: Stream, verbose: bool = False
    ):
        """
        Return if a stream (assuming a uniqe SEED-id) contains a day of data
        not exceeding the allowed `gap` (default 300s.)
        """
        if not isinstance(stream, Stream) or len(stream) == 0:
            return False
        npts_day = int(stream[0].stats.sampling_rate * 86400)
        npts_gap = int(stream[0].stats.sampling_rate * self._max_gap)
        npts_str = sum([trace.stats.npts for trace in stream])
        if verbose:
            print(
                'Samples in day = {}, samples in stream = {}, max gaps = {}.'
                .format(npts_day, npts_str, npts_gap)
            )
        return npts_str >= (npts_day - npts_gap)

    def get_waveforms(
        self, receiver: str, time: np.datetime64, centered: bool = True,
        duration: float = 86400., buffer: float = 60.,
        allow_wildcards: bool = False, verbose: bool = False
    ):
        """
        Get waveforms given the SEED-id `receiver` and
        `time` (default `centered`)
        for `duration` (default 86400s) and `buffer` (default 60s).
        """
        # check if receiver SEED-id is valid
        check_receiver(
            receiver,
            allow_wildcards=allow_wildcards,
            raise_error=True
        )

        # split receiver SEED-id
        network, station, location, channel = receiver.split('.')

        # center time of 24h window -12h
        t0 = time
        if centered:
            t0 -= pd.offsets.DateOffset(seconds=duration/2)
        t1 = t0 + pd.offsets.DateOffset(seconds=duration) 
        if buffer > 0.:
            t0 -= pd.offsets.DateOffset(seconds=buffer)
            t1 += pd.offsets.DateOffset(seconds=buffer)
        if verbose:
            print(
                'Get waveforms for {} from {} until {}'.format(
                    receiver, t0, t1
                )
            )

        # start with an empty Stream
        stream = Stream()

        # request or download per entire day
        for day in pd.date_range(
            start=t0 + pd.offsets.DateOffset(0, normalize=True),
            end=t1 + pd.offsets.DateOffset(1, normalize=True),
            name='days',
            freq='D'
        ):
            t = UTCDateTime(day)

            if verbose:
                print(_msg_load_archive.format(t))

            for sds in self.sds_read:
                daystream = sds.get_waveforms(
                    network=network,
                    station=station,
                    location=location,
                    channel=channel,
                    starttime=t,
                    endtime=t + 86400,
                )
                if self._check_daystream_length(daystream, verbose):
                    if verbose:
                        print(_msg_loaded_archive.format(t))
                    stream += daystream
                    break 
            else:
                if verbose:
                    print(_msg_no_data.format(t))

                # get attempt via fdsn
                if self.fdsn:
                    if verbose:
                        print('Try FDSN.')
                    try:
                        daystream = self.fdsn.get_waveforms(
                            network=network,
                            station=station,
                            location=location,
                            channel=channel,
                            starttime=t,
                            endtime=t + 86400,
                        )
                        if self._sds_write_daystream(daystream, verbose):
                            stream += daystream
                            continue
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except FDSNNoDataException:
                        if verbose:
                            print(_msg_no_data.format(t))
                    except Exception as e:
                        if verbose:
                            print('an error occurred:')
                            print(e)

                # get attempt via nms
                if self.nms:
                    if verbose:
                        print('Try NMS_Client')
                    try:
                        daystream = self.nms.get_waveforms(
                            starttime=t,
                            station=station,
                            channel=channel,
                            verbose=False,
                        )
                        if self._sds_write_daystream(daystream, verbose):
                            stream += daystream
                            continue
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:
                        if verbose:
                            print('an error occurred:')
                            print(e)

        return stream.trim(starttime=to_UTCDateTime(t0), endtime=to_UTCDateTime(t1))

    def get_preprocessed_waveforms(
        self, receiver: str, time: np.datetime64, operations: dict,
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
        check_receiver(receiver, allow_wildcards=False, raise_error=True)

        t0 = to_UTCDateTime(time) - duration/2
        t1 = t0 + duration
        ch = receiver.split('.')[-1]

        # radial or transverse component? Request all channels manually.
        if ch[-1] == 'R' or ch[-1] == 'T':
            st = Stream()
            for c in three_components:
                st += self.get_waveforms(
                    receiver=receiver[:-1]+c,
                    time=time,
                    duration=duration,
                    centered=True,
                    verbose=debug,
                    **kwargs
                )
        else:
            st = self.get_waveforms(
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
            st = preprocess(
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

    def get_pair_preprocessed_waveforms(
        self, pair, time: np.datetime64, operations: dict, **kwargs
    ):
        """
        Get the preprocessed `obspy.Stream` given the SEED-ids receiver
        `pair`, `time` and preprocess `operations`.
        """
        rA, rB = split_pair(pair)
        return (
            self.get_preprocessed_waveforms(rA, time, operations, **kwargs) +
            self.get_preprocessed_waveforms(rB, time, operations, **kwargs)
        )
