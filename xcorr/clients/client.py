r"""

:mod:`client.client` -- Client
==============================

Load waveform data from a primary local sds archive, and automatically
retrieve missing daily waveforms using fdsn and nms web request services.

"""

# Mandatory imports
import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient
try:
    from nms_tools.nms_client import Client as nmsClient
except ModuleNotFoundError:
    nmsClient = None  # make it work without nmsClient


# Relative imports
from ..clients.datafetch import stream2SDS
from ..preprocess import preprocess as xcorr_preprocess
from ..util.receiver import check_receiver, split_pair
from ..util.time import to_UTCDateTime


__all__ = []


# client messages
_msg_no_data = 'No waveform data found for {}.'
_msg_load_archive = 'Get waveform data for {}.'
_msg_loaded_archive = 'Waveform data for {} loaded from archive.'
_msg_added_archive = 'Waveform data for downloaded and added to archive.'


class Client(object):
    """
    `xcorr` waveform request client.
    For details see the :meth:`~xcorr.clients.client.Client.__init__()`
    method.
    """
    def __init__(
        self, sds_root: str = None, sds_root_write: str = None,
        sds_root_read: list = None, fdsn_service='IRIS',
        nms_service: bool = True, max_gap: float = None
    ):
        r"""Initialize a `xcorr` waveform request client.

        >>> client = Client('/tmp')

        Parameters:
        -----------
        sds_root : `str`
            Path to the local SeisComP Data Structure (SDS) archive. All
            downloaded waveforms will be added to this archive if
            ``sds_root_write`` is not specified.

        sds_root_write: `str`, optional
            Specify an SDS archive to store automatically downloaded
            waveforms by the FDSN web service and/or NMS Client.

        sds_root_read: `list`, optional
            Scan multiple SDS archives.

        fdsn_service : `str` or :class:`obspy.clients.fdsn.Client`, optional
            FDSN web service base URL. If `None` or `False`, FDSN web service
            is disabled. Default web service is 'IRIS'.

        nms_service : `bool`, optional
            Enable the NMS Client to access restricted IMS waveform data of the
            Comprehensive Nuclear-Test-Ban Treaty Organization (CTBTO)
            if `True` (default). If no :class:`nms_tools.nms_client.Client`
            is found or credentials are invalid, ``nms_service`` will be
            disabled.

        max_gap : `float`, optional
            Specify the maximum time gap in seconds that is allowed in a day.
            If `None`, ``max_gap`` is 300s (default).

        """
        if (
            (not sds_root and not sds_root_read) or
            (not sds_root and not sds_root_write)
        ):
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

        if fdsn_service:
            if isinstance(fdsn_service, str):
                self._fdsn = fdsnClient(fdsn_service)
            elif isinstance(fdsn_service, fdsnClient):
                self._fdsn = fdsn_service
            else:
                raise TypeError(
                    '`fdsn_service` should be of type str or '
                    ':class:`obspy.clients.fdsn.Client`!'
                )
        else:
            self._fdsn = None
        self._nms = nmsClient() if nms_service and nmsClient else None
        self._max_gap = abs(max_gap or 300.)

    @property
    def sds_root(self):
        raise NotImplementedError(
            'Use either `sds_root_read` or `sds_root_write`!'
        )

    @property
    def sds_root_read(self):
        r"""sds_root_read.

        Returns:
        --------
        sds_read : `list`
            List of SDS client roots used to initiate ``self.sds_read``.
        """
        return self._sds_root_read

    @property
    def sds_root_write(self):
        r""""sds_root_read

        Returns:
        --------
        sds_root_write : `str`
            SDS client root used to initiate ``self.sds_write``.
        """
        return self._sds_root_write

    @property
    def sds_read(self):
        r"""sds_read

        Returns:
        --------
        sds_read : `list` of :class:`obspy.clients.filesystem.sds.Client`
            List of SDS client objects used to get local waveforms.
        """
        return self._sds_read

    @property
    def sds_write(self):
        r"""sds_write

        Returns:
        --------
        sds_write : :class:`obspy.clients.filesystem.sds.Client`
            SDS client object used to write waveforms.
        """
        return self._sds_write

    @property
    def fdsn(self):
        r"""fdsn

        Returns:
        --------
        fdsn : :class:`obspy.clients.fdsn.Client`
            FDSN web service object used to download missing waveforms.
        """
        return self._fdsn

    @property
    def nms(self):
        r"""fdsn

        Returns:
        --------
        nms : :class:`nms_tools.nms_client.Client`
            NMS Client to access restricted IMS waveform data of the
            Comprehensive Nuclear-Test-Ban Treaty Organization (CTBTO),
            used to download missing waveforms.
        """
        return self._nms

    @property
    def max_gap(self):
        r"""max_gap

        Returns:
        --------
        max_gap : `float`
            Maximum number of seconds that are allowed to be missing in a
            day of data.
        """
        return self._max_gap

    def _sds_write_daystream(
        self, stream: Stream, verbose: bool = False
    ):
        r"""Wrapper to write a day stream of data to the local SDS archive.

        Parameters:
        -----------
        stream : :class:`obspy.Stream`
            Stream with a day of data.

        verbose : `bool`, optional
            Print a message if ``stream`` is successfully added to the SDS
            archive.

        Returns:
        --------
        status : `bool`
            Returns `False` if ``stream`` does not pass the day criteria of
            :meth:'_check_daystream_length', otherwise returns `True`.

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
        r"""Returns `True` if a stream (assuming a uniqe SEED-id) contains a
        day of data not exceeding the allowed ``max_gap``.
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
        r"""Get waveforms from the clients given a SEED-id.

        Parameters:
        -----------
        receiver : `str`
            Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

        time : `np.datetime64`
            Anchor node of the time window, either the start or the center of
            of the waveform time window, depending on ``centered``.

        centered : `bool`, optional
            Controls the anchor type of ``time``. If `True`, ``time``
            corresponds to the center of the waveform time window, otherwise
            ``time`` is the left corner (start) of the time window. Defaults
            to `True`.

        duration : `float`, optional
            Set the duration of the waveform time window, in seconds. Defaults
            to a full day: 86400s.

        buffer : `float`, optional
            Symmetrically extent the time window by a buffer, in seconds.
            Defaults to 60s.

        allow_wildscards : `bool`, optional
            Enable wildcards '*' and '?' in the ``receiver`` SEED-id string.
            Defaults to `False`, not allowing wildcards.

        verbose : `bool`, optional
            Print a message if ``stream`` is successfully added to the SDS
            archive.

        Returns:
        --------
        stream : :class:`obspy.Stream`
            The requested waveforms.

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
        t0 = pd.to_datetime(time)
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
        stream.trim(starttime=to_UTCDateTime(t0), endtime=to_UTCDateTime(t1))
        return stream

    def get_preprocessed_waveforms(
        self, receiver: str, time: np.datetime64, preprocess: dict,
        duration: float = 86400., inventory: Inventory = None,
        three_components: str = '12Z',
        verbose: bool = False, debug: bool = False, **kwargs
    ):
        r"""Get preprocessed waveforms from the clients given a SEED-id and
        an operations dictionary.

        Parameters:
        -----------
        receiver : `str`
            Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

        time : `np.datetime64`
            Center time of the waveform time window.

        preprocess : `dict`
            Preprocessing operations dictionary, containing a list of
            operations per SEED channel as key. Each list item should be a
            tuple ('operation', {parameters}).
            Use :func:`xcorr.preprocess.help` to list all valid operations and
            their documentation.

        duration : `float`, optional
            Set the duration of the waveform time window, in seconds. Defaults
            to a full day: 86400s.

        inventory : :class:`obspy.Inventory`, optional
            Inventory object, including the instrument response.

        three_components: `str` {'12Z', 'NEZ'}, optional
            Set the three-component orientation characters. Defaults to '12Z'.

        verbose : `bool`, optional
            Print standard status messages if `True`. Defaults to `False`.

        debug : `bool`, optional
            Print many status messages if `True`. Defaults to `False`.

        kwargs :
            Parameters passed to :meth:`get_waveforms`.

        Returns:
        --------
        stream : :class:`obspy.Stream`
            The requested waveforms after preprocessing.

        """
        # check if receiver SEED-id is valid
        check_receiver(receiver, allow_wildcards=False, raise_error=True)

        t0 = to_UTCDateTime(time) - duration/2
        t1 = t0 + duration
        ch = receiver.split('.')[-1]

        # radial or transverse component? Request all channels manually.
        if ch[-1] == 'R' or ch[-1] == 'T':
            st = Stream()
            for c in three_components:  # '12NEZ'??
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
            st = xcorr_preprocess(
                stream=st,
                operations=preprocess[ch],
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
        self, pair, **kwargs
    ):
        r"""Get preprocessed waveforms from the clients given a receiver couple
        SEED-id and an operations dictionary.

        Parameters:
        -----------
        pair : str or :mod:`~xarray.DataArray`
            Receiver couple separated by `separator`. Each receiver is
            specified by a SEED-id string:
            '{network}.{station}.{location}.{channel}'.

        kwargs :
            Parameters passed to :meth:`get_preprocessed_waveforms` and
            :meth:`get_waveforms`.

        Returns:
        --------
        stream : :class:`obspy.Stream`
            The requested waveforms after preprocessing.

        """
        rA, rB = split_pair(pair)
        return (
            self.get_preprocessed_waveforms(rA, **kwargs) +
            self.get_preprocessed_waveforms(rB, **kwargs)
        )
