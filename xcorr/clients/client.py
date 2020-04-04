r"""

:mod:`client.client` -- Client
==============================

Load waveform data from a primary local sds archive, and automatically
retrieve missing daily waveforms using fdsn and nms web request services.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient
try:
    from nms_tools.nms_client import Client as nmsClient
except ModuleNotFoundError:
    nmsClient = None
try:
    import dask
except ModuleNotFoundError:
    dask = None


# Relative imports
from ..clients.datafetch import stream2SDS
from ..preprocess import preprocess as xcorr_preprocess
from ..util.receiver import check_receiver, split_pair, receiver_to_dict
from ..util.time import to_UTCDateTime, get_dates


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
        nms_service: bool = True, max_gap: float = None,
        parallel: bool = True,
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

        dask : `bool`, optional
            Parallelize workflows using :class:`dask.delayed`. Defaults to
            `True` if `dask` is found.

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
        self._parallel = dask and parallel

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

    @property
    def parallel(self):
        r"""parallel

        Returns:
        --------
        parallel : `bool`
            `True` if :mod:`dask` is loaded and :class:`Client` is initiated
            ``parallel``=`True`. All routines of :class:`Client` will default
            to this value.
        """
        return self._parallel

    def _sds_write_daystream(
        self, stream: Stream, verb: int = 0
    ):
        r"""Wrapper to write a day stream of data to the local SDS archive.

        Parameters:
        -----------
        stream : :class:`obspy.Stream`
            Stream with a day of data.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        Returns:
        --------
        status : `bool`
            Returns `False` if ``stream`` does not pass the day criteria of
            :meth:'_check_daystream_length', otherwise returns `True`.

        """
        if not self._check_daystream_length(stream, verb > 1):
            return False
        stream2SDS(
            stream,
            sds_path=self.sds_root_write,
            force_override=True,
            verbose=verb == 4,
        )
        if verb > 0:
            print(_msg_added_archive)
        return True

    def _check_daystream_length(
        self, stream: Stream, verb: int = 0
    ):
        r"""Returns `True` if a stream (assuming a uniqe SEED-id) contains a
        day of data not exceeding the allowed ``max_gap``.
        """
        if not isinstance(stream, Stream) or len(stream) == 0:
            return False
        npts_day = int(stream[0].stats.sampling_rate * 86400)
        npts_gap = int(stream[0].stats.sampling_rate * self._max_gap)
        npts_str = sum([trace.stats.npts for trace in stream])
        if verb > 2:
            print(
                'Samples in day = {}, samples in stream = {}, max gaps = {}.'
                .format(npts_day, npts_str, npts_gap)
            )
        return npts_str >= (npts_day - npts_gap)

    def get_waveforms(
        self, receiver: str, time: pd.Timestamp, centered: bool = True,
        duration: float = 86400., buffer: float = 60.,
        allow_wildcards: bool = False, verb: int = 0, **kwargs
    ):
        r"""Get waveforms from the clients given a SEED-id.

        Parameters:
        -----------
        receiver : `str`
            Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

        time : `pd.Timestamp`
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

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
            Parameters passed to :meth:`_get_waveforms_for_date`

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

        # center time of 24h window -12h
        t0 = pd.to_datetime(time)
        if centered:
            t0 -= pd.offsets.DateOffset(seconds=duration/2)
        t1 = t0 + pd.offsets.DateOffset(seconds=duration)
        if buffer > 0.:
            t0 -= pd.offsets.DateOffset(seconds=buffer)
            t1 += pd.offsets.DateOffset(seconds=buffer)
        if verb > 0:
            print(
                'Get waveforms for {} from {} until {}'
                .format(receiver, t0, t1)
            )

        # split receiver SEED-id to a dictionary
        receiver = dict(zip(
            ['network', 'station', 'location', 'channel'],
            receiver.split('.')
        ))

        # list of days
        days = get_dates(t0, t1)

        # get streams per day
        stream = Stream()
        for day in days:
            stream += self._get_waveforms_for_date(
                receiver=receiver,
                date=day
            )

        # trim to asked time range
        stream.trim(starttime=to_UTCDateTime(t0), endtime=to_UTCDateTime(t1))

        return stream

    def _get_waveforms_for_date(self, receiver: dict, date: pd.Timestamp,
                                download: bool = True, verb: int = 0):
        r"""Get the waveforms for a receiver and date.

        Parameters:
        -----------
        receiver : `dict`
            Receiver dictionary with SEED-id keys:
            ['network', 'station', 'location', 'channel'].

        date : :class:`pd.Timestamp`
            The date.

        download : `bool`, optional
            If `True` (default) automatically download waveforms missing in the
            local SDS archives ``self.sds_read`` using ``self.fdsn`` and
            ``self.nms`` services. Data is added to ``self.sds_write``.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        Returns:
        --------
        stream : :class:`obspy.Stream`
            The waveforms for ``receiver`` on ``date``.

        """
        time = pd.to_datetime(date) + pd.offsets.DateOffset(0, normalize=True)
        time = UTCDateTime(time)
        args = dict(**receiver, starttime=time, endtime=time + 86400)

        if verb > 0:
            print(
                "Get waveforms for {network}.{station}.{location}.{channel} "
                "from {starttime} until {endtime}".format(**args)
            )

        if verb > 1:
            print(_msg_load_archive.format(time))

        for sds in self.sds_read:
            daystream = sds.get_waveforms(**args)
            if self._check_daystream_length(daystream, verb > 2):
                if verb > 1:
                    print(_msg_loaded_archive.format(time))
                return daystream
        else:
            if verb > 1:
                print(_msg_no_data.format(time))

            if not download:
                return None

            # get attempt via fdsn
            if self.fdsn:
                if verb > 1:
                    print('Try FDSN.')
                try:
                    daystream = self.fdsn.get_waveforms(**args)
                    if self._sds_write_daystream(daystream, verb > 2):
                        return daystream
                except (KeyboardInterrupt, SystemExit):
                    raise
                except FDSNNoDataException:
                    if verb > 1:
                        print(_msg_no_data.format(time))
                except Exception as e:
                    if verb > 0:
                        print('an error occurred:')
                        print(e)

            # get attempt via nms
            if self.nms:
                if verb > 1:
                    print('Try NMS_Client')
                try:
                    daystream = self.nms.get_waveforms(**args)
                    if self._sds_write_daystream(daystream, verb > 2):
                        return daystream
                    if verb > 1:
                        print(_msg_no_data.format(time))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    if verb > 0:
                        print('an error occurred:')
                        print(e)
        return None

    def get_preprocessed_waveforms(
        self, receiver: str, time: pd.Timestamp, preprocess: dict,
        duration: float = 86400., inventory: Inventory = None,
        three_components: str = '12Z', verb: int = 0, **kwargs
    ):
        r"""Get preprocessed waveforms from the clients given a SEED-id and
        an operations dictionary.

        Parameters:
        -----------
        receiver : `str`
            Receiver SEED-id string '{network}.{station}.{location}.{channel}'.

        time : `pd.Timestamp`
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

        three_components: {'12Z', 'NEZ'}, optional
            Set the three-component orientation characters. Defaults to '12Z'.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
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
        if ch[-1] in 'RT':
            st = Stream()
            for c in three_components:  # '12NEZ'??
                st = st + self.get_waveforms(
                    receiver=receiver[:-1]+c,
                    time=time,
                    duration=duration,
                    centered=True,
                    verb=verb,
                    **kwargs
                )
        else:
            st = self.get_waveforms(
                receiver=receiver,
                time=time,
                duration=duration,
                centered=True,
                verb=verb,
                **kwargs
            )

        if verb > 2:
            print(st)
        if not isinstance(st, Stream) or len(st) == 0:
            return Stream()

        st = xcorr_preprocess(st,
            operations=preprocess[ch],
            inventory=inventory,
            starttime=t0,
            endtime=t1,
            verb=verb-1,
        )
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

        **kwargs :
            Parameters passed to :meth:`get_waveforms` via
            :meth:`get_preprocessed_waveforms`.

        Returns:
        --------
        stream : :class:`obspy.Stream`
            The requested waveforms after preprocessing.

        """
        # split
        rA, rB = split_pair(pair)

        # get streams per receiver
        stream = (
            self.get_preprocessed_waveforms(rA, **kwargs) +
            self.get_preprocessed_waveforms(rB, **kwargs)
        )

        return stream

    def data_availability(
        self, pairs_or_receivers: list, times: pd.DatetimeIndex,
        verb: int = 0, **kwargs
    ):
        r"""Verify the waveform data availability for receivers and times.

        Parameters:
        -----------
        pairs_or_receivers : `list`
            List of receivers or pairs (receiver couple separated by a '-').
            Each receiver should be specified by a SEED-id string of format
            '{network}.{station}.{location}.{channel}'.

        times : `pd.DatetimeIndex`
            Sequence of dates with `freq`="D".

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
            Parameters passed to :meth:`verify_data_availability`.

        Returns:
        --------
        status : :class:`xarray.DataArray`
            Data availability status N-D labelled array with dimensions
            ``time`` and ``receiver``.

        """
        status = self.init_data_availability(pairs_or_receivers, times)

        if verb:
            print('Verify {} (receiver, time) combinations.'
                  .format(status.size))

        verified = self.verify_data_availability(status, **kwargs)

        if verb:
            print('Verified {} out of {}.'
                  .format(verified, status.size))

        return status

    def init_data_availability(
        self, pairs_or_receivers: list, times: pd.DatetimeIndex,
        extend_days: int = 0, **kwargs
    ):
        r"""Create a new N-D labelled array to verify the waveform data
        availability for receivers and times.

        Parameters:
        -----------
        pairs_or_receivers : `list`
            List of receivers or pairs (receiver couple separated by a '-').
            Each receiver should be specified by a SEED-id string of format
            '{network}.{station}.{location}.{channel}'.

        times : `pd.DatetimeIndex`
            Sequence of dates with `freq`="D".

        extend_days : `int`, optional
            Extend ``times`` with n-days at both left and right edges.

        **kwargs :
            Extra parameters are passed on to :func:`xcorr.util.split_pair`

        Returns:
        --------
        status : :class:`xarray.DataArray`
            Data availability status N-D labelled array with dimensions
            ``time`` and ``receiver``.

        """
        assert isinstance(times, pd.DatetimeIndex) and times.freqstr == 'D', (
            '``times`` should be a pandas.DatetimeIndex with freq="D"!'
        )

        # Get all receivers from pairs
        receivers = []
        for p in pairs_or_receivers:
            receivers += split_pair(p, to_dict=False, **kwargs)
        receivers = sorted(list(set(receivers)))

        # Time
        if extend_days > 0:
            times = pd.date_range(
                start=times[0] - pd.offsets.DateOffset(days=extend_days),
                end=times[-1] + pd.offsets.DateOffset(days=extend_days),
                freq=times.freqstr
            )

        # Construct status xarray object
        status = xr.DataArray(
            data=np.zeros((len(receivers), len(times)), dtype=np.byte),
            coords=[np.array(receivers, dtype=object), times],
            dims=['receiver', 'time'],
            name='status',
            attrs={
                'long_name': 'Data availability status',
                'standard_name': 'data_availability_status',
                'units': '-',
                'valid_range': np.byte([-1, 1]),
                'flag_values': np.byte([-1, 0, 1]),
                'flag_meanings': 'missing not_validated available',
            }
        )
        status.receiver.attrs = {
            'long_name': 'Receiver SEED-id',
            'standard_name': 'receiver_seed_id',
            'units': '-',
        }

        return status

    def verify_data_availability(
        self, status: xr.DataArray, count_verified: bool = False,
        parallel: bool = None, compute: bool = True, **kwargs
    ):
        r"""Verify daily waveform availability for receivers and times.

        Parameters:
        -----------
        status : :class:`xarray.DataArray`
            Data availability status N-D labelled array with dimensions
            ``status.time`` and ``status.receiver``. ``status`` is updated in
            place.

        count_verified : `bool`, optional
            If `True`, count the number of verified (receiver, time) couples.
            Defaults to `False`.

        parallel : `bool`, optional
            Enable parallel processing using :method:`dask`. If `None`
            (default) ``self.parallel`` is used.

        compute : `bool`, optional
            Compute the lazy :class:`dask.delayed` result in parallel, if
            `True` (default) and ``dask`` is enabled. Set to `False` to add
            more delayed tasks, or to visualize the process with
            ``stream.visualize()``.

        **kwargs :
            Parameters passed to :meth:`_get_waveforms_for_date`

        Returns:
        --------
        verified : `int`, optional
            The number of verified (receiver, time) couples. If
            ``count_verified`` is `True`, otherwise `None`.

        """
        parallel = parallel or self.parallel
        if parallel and not dask:
            raise RuntimeError('Dask is required but cannot be found!')

        # get daily waveforms per receiver and time
        verified = [] if parallel else 0

        for receiver in status.receiver:
            rec_dict = receiver_to_dict(str(receiver.values))
            for time in status.time:
                if status.loc[{'receiver': receiver, 'time': time}] == 1:
                    continue
                if parallel:
                    stream = dask.delayed(self._get_waveforms_for_date)(
                        rec_dict, time.values, verb=0, **kwargs
                    )
                    flag = dask.delayed(set_availability_by_stream)(
                        status, receiver, time, stream, True
                    )
                    verified.append(flag != 0)
                else:
                    stream = self._get_waveforms_for_date(
                        receiver, time.values, **kwargs
                    )
                    set_availability_by_stream(status, receiver, time, stream)
                    verified += 1

        if parallel:
            verified = dask.delayed(sum)(verified)
            verified = verified.compute() if compute else verified

        if count_verified or (parallel and not compute):
            return verified


# local methods use for dask.delayed

def set_availability_by_stream(
    status: xr.DataArray, receiver: xr.DataArray, time: xr.DataArray,
    stream: Stream, return_flag: bool = False,
):
    r"""Set status availability for receiver and time by stream.

    Parameters:
    -----------
    status : :class:`xarray.DataArray`
        Data availability status N-D labelled array with dimensions
        ``status.time`` and ``status.receiver``, updated in place.

    receiver : :class:`xarray.DataArray`
        ``status`` receiver coordinate to set the status.

    time : :class:`xarray.DataArray`
        ``status`` time coordinate to set the status.

    stream : :class:`obspy.Stream`
        Waveform stream used to define the status code. If `None`,
        ``status``=-1, else 1.

    return_flag : `bool`, optional
       If `True`, return the status flag. Defaults to `False`.

    Returns:
    --------
    flag : `np.int8`
        Returns data availability flag if ``return_flag`` is `True`,
        otherwise `None`.

    """
    flag = 1 if stream else -1
    status.loc[{'receiver': receiver, 'time': time}] = flag
    return flag if return_flag else None


def sum_stream_list(streams: list):
    r"""Sum a list of streams into a single stream object.

    Parameters:
    -----------
    streams : `list`
        A list of :class:`obspy.Stream` objects.

    Returns:
    --------
    stream : :class:`obspy.Stream`
        The merged single stream object.

    """
    stream = Stream()
    for st in streams:
        stream = stream + st
    return stream
