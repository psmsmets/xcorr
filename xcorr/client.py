r"""

:mod:`clients.client` -- Client
===============================

Load waveform data from a primary local sds archive, and automatically
retrieve missing daily waveforms using fdsn and vdms web request services.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.filesystem.sds import Client as sdsClient
from obspy import warnings as obspyWarn
import warnings
import traceback
from tabulate import tabulate
# VDMS client for IMS waveforms?
try:
    from pyvdms import Client as vdmsClient
except ModuleNotFoundError:
    vdmsClient = False
# Dask?
try:
    import dask
except ModuleNotFoundError:
    dask = False
# Dask distributed?
try:
    from dask import distributed
except ModuleNotFoundError:
    distributed = False


# Relative imports
from .preprocess import preprocess as xcorr_preprocess
from .preprocess import operations_to_json
from . import util


__all__ = ['Client']


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
        vdms_service: bool = True, max_gap: float = None,
        force_write: bool = False, parallel: bool = True,
    ):
        """Initialize a `xcorr` waveform request client.

        Parameters
        ----------
        sds_root : `str`
            Path to the local SeisComP Data Structure (SDS) archive. All
            downloaded waveforms will be added to this archive if
            ``sds_root_write`` is not specified.

        sds_root_write: `str`, optional
            Specify an SDS archive to store automatically downloaded
            waveforms by the FDSN web-service and/or NMS Client.

        sds_root_read: `list`, optional
            Scan multiple SDS archives.

        fdsn_service : `str` or :class:`obspy.clients.fdsn.Client`, optional
            FDSN web-service base URL. If `None` or `False`, FDSN web-service
            is disabled. Default web-service is 'IRIS'.

        vdms_service : `bool` or :class:`pyvdms.Client`, optional
           Enable VDMS web-service client to request restricted IMS data (if
           granted) for the verification of the Comprehensive Nuclear-Test-Ban
           Treaty (CTBT). Data access is restricted to principal users only!
           If `None` or `False`, VDMS web-service is disabled. Defaults to
           `True`, initializing a default pyvdms.Client(). If `pyvdms` is not
           found or credentials are invalid, ``vdms_service`` is disabled.

        max_gap : `float`, optional
            Specify the maximum time gap in seconds that is allowed in a day.
            If `None`, ``max_gap`` is 300s (default).

        force_write : `bool`, optional
            Force to write downloaded day streams to the SDS archive even when
            the cummulated gap is larger than max_gap. Defaults to `False`.

        parallel : `bool`, optional
            Parallelize workflows using :class:`dask.delayed`. Defaults to
            `True` if `dask` is found.

        Examples
        --------
        >>> from xcorr import client
        >>> client = Client(sds_root='/tmp')

        """
        if (
            (not sds_root and not sds_root_read) or
            (not sds_root and not sds_root_write)
        ):
            raise AttributeError(
                'At least `sds_root` or `sds_root_read` and '
                '`sds_root_write` are required.'
            )

        # other parameters
        self._max_gap = abs(max_gap or 300.)
        self._force_write = force_write or False
        self._parallel = dask and distributed and parallel

        # set sds roots for reading and writing
        self._sds_root_write = sds_root_write if sds_root_write else sds_root
        self._sds_root_read = (sds_root_read if sds_root_read else [sds_root])

        # append sds write root to read
        if sds_root_write:
            self._sds_root_read += [sds_root_write]
        else:
            self._sds_root_read += [sds_root]

        # keep unique sds roots
        self._sds_root_read = list(set(self._sds_root_read))
        self._sds_read = []
        self._sds_locks = dict()

        # init sds accessors per root
        for root in self._sds_root_read:
            self._sds_read.append(sdsClient(sds_root=root))

        # fdsn web-service
        if fdsn_service:
            if isinstance(fdsn_service, str):
                self._fdsn = fdsnClient(fdsn_service)
            elif isinstance(fdsn_service, fdsnClient):
                self._fdsn = fdsn_service
            else:
                raise TypeError('`fdsn_service` should be of type str or '
                                ':class:`obspy.clients.fdsn.Client`!')
        else:
            self._fdsn = False

        # vdms web-service
        if vdmsClient:
            if isinstance(vdms_service, bool):
                self._vdms = vdmsClient() if vdms_service else False
            elif isinstance(vdms_service, vdmsClient):
                self._vdms = vdms_service
            else:
                raise TypeError('`vdms_service` should be of type bool or '
                                ':class:`pyvdms.Client`!')
            if self._vdms:
                test = self._vdms.get_channels('*H1', 'BDF')
                if not isinstance(test, pd.DataFrame):
                    self._vdms = False
                    warnings.warn(
                        'VDMS Client test failed. Service shall be disabled.'
                    )
        else:
            self._vdms = False

    def __str__(self):
        """Get the formatted xcorr client overview.
        """
        out = []

        out += [['sds read', self.sds_root_read]]
        out += [['sds write', self.sds_root_write]]
        out += [['fdsn', 'Yes' if self.fdsn else 'No']]
        if self.fdsn:
            out += [['fdsn base url', self.fdsn.base_url]]
        out += [['vdms', 'Yes' if self.vdms else 'No']]
        if self.vdms:
            out += [['vdms client', self.vdms._request.clc]]
        out += [['max gap', f'{self.max_gap}s']]
        out += [['force write', 'Yes' if self.force_write else 'No']]
        out += [['parallel', 'Yes' if self.parallel else 'No']]

        return tabulate(out)

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    @property
    def sds_root(self):
        raise NotImplementedError(
            'Use either `sds_root_read` or `sds_root_write`!'
        )

    @property
    def sds_root_read(self):
        """List of SDS client roots used to initiate ``self.sds_read``.
        """
        return self._sds_root_read

    @property
    def sds_root_write(self):
        r""""SDS client root used to initiate ``self.sds_write``.
        """
        return self._sds_root_write

    @property
    def sds_read(self):
        """List of SDS client objects used to get local waveforms.
        """
        return self._sds_read

    @property
    def sds_write(self):
        """SDS client object used to write waveforms.
        """
        return self._sds_write

    @property
    def fdsn(self):
        """FDSN web-service object used to download missing waveforms.
        """
        return self._fdsn

    @property
    def vdms(self):
        """VDMS web-service client to request restricted IMS data (if granted)
        for the verification of the Comprehensive Nuclear-Test-Ban Treaty
        (CTBT). Data access is restricted to principal users only!
        """
        return self._vdms

    @property
    def max_gap(self):
        """Maximum number of seconds that are allowed to be missing in a day
        of data.
        """
        return self._max_gap

    @property
    def force_write(self):
        """Force writing downloaded day streams to the SDS archive even when
        the cummulated gap is larger than max_gap.
        """
        return self._force_write

    @property
    def parallel(self):
        """Returns `True` if :mod:`dask` is loaded and :class:`Client` is
        initiated with ``parallel``=`True`. All routines of :class:`Client`
        will default to this value.
        """
        return self._parallel

    def _sds_write_daystream(
        self, stream: Stream, force_write: bool = None,
        parallel: bool = None, verb: int = 0
    ):
        """
        Wrapper to write a day stream of data to the local SDS archive.

        Parameters
        ----------
        stream : :class:`obspy.Stream`
            Stream with a day of data.

        force_write : `bool`, optional
            Force to write downloaded day streams to the SDS archive even when
            the cummulated gap is larger than max_gap. Defaults to `False`.

        parallel : `bool`, optional
            Enable parallel processing using :method:`dask`. If `None`
            (default) ``self.parallel`` is used.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        Returns
        -------
        status : `bool`
            Returns `False` if ``stream`` does not pass the day criteria of
            :meth:'check_duration', otherwise returns `True`.

        """
        # parallel?
        parallel = self.parallel if parallel is None else parallel

        # lock?
        locked = distributed and parallel

        # always write?
        force_write = force_write or self.force_write

        # check
        passed = self.check_duration(stream, verb=verb)

        # passed?
        if not passed and not force_write:
            return False

        # get sds write lock
        if locked:
            lock = distributed.Lock(stream[0].id)
            lock.acquire()

        # add to archive
        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            # write to sds archive
            util.stream.stream2SDS(
                stream,
                sds_path=self.sds_root_write,
                method='overwrite',
                extra_samples=None,
                verbose=verb == 4,
            )

        # release sds write access
        if locked:
            lock.release()

        if verb > 0:
            print(_msg_added_archive)

        return passed

    def check_duration(
        self, stream: Stream, duration: float = None, receiver: str = None,
        verb: int = 0, **kwargs
    ):
        """
        Wrapper to write a day stream of data to the local SDS archive.

        Parameters
        ----------
        stream : :class:`obspy.Stream`
            Stream waveform data (assuming a unique SEED-id). If multiple
            receivers (SEED-ids) are present specify ``id``.

        duration : `float`, optional
            Set the minimal stream duration. Defaults to one-day (86400s).

        receiver : `str`, optional
            Receiver SEED-id string of format
            '{network}.{station}.{location}.{channel}'. If `None`, the first
            id in the stream is used.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
            Parameters passed to :func:`duration`.

        Returns
        -------
        status : `bool`
            Returns `True` if stream duration is equal or greater than
            ``duration`` - ``self.max_gap`` and `False` otherwise.

        """

        if not isinstance(stream, Stream) or len(stream) == 0:
            return False

        duration = duration or 86400.

        assert isinstance(duration, float), (
            '``duration`` should be float seconds.'
        )

        d = util.stream.duration(stream, receiver, **kwargs)

        if len(d) == 0:
            return False

        time = d['time'] if receiver else d[next(iter(d))]['time']

        passed = time >= duration - self.max_gap

        if verb > 2:
            print(f'Time: {time}s, max gap: {self.max_gap}s, passed: {passed}')

        return passed

    def get_waveforms(
        self, receiver: str, time: pd.Timestamp, centered: bool = True,
        duration: float = None, buffer: float = None,
        allow_wildcards: bool = False, download: bool = True,
        verb: int = 0, **kwargs
    ):
        """
        Get waveforms from the clients given a SEED-id.

        Parameters
        ----------
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

        download : `bool`, optional
            If `True` (default), automatically download waveforms missing in
            all local SDS archives listed in ``self.sds_read`` using
            ``self.fdsn`` and ``self.vdms`` webservices. Data is added to
            ``self.sds_write``.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        Addition arguments are passed to :func:`_get_waveforms_for_date`.

        Returns
        -------
        stream : :class:`obspy.Stream`
            The requested waveforms.

        """

        # check if receiver SEED-id is valid
        util.receiver.check_receiver(
            receiver,
            allow_wildcards=allow_wildcards,
            raise_error=True
        )

        # check duration
        duration = duration or 86400.

        if not (isinstance(duration, float) or isinstance(duration, int)):

            raise TypeError('duration should be in float seconds.')

        if duration <= 0.:

            raise ValueError('duration cannot be zero or negative.')

        # check buffer
        buffer = buffer or 60.

        if not (isinstance(buffer, float) or isinstance(buffer, int)):

            raise TypeError('buffer should be in float seconds.')

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

        # UTCDatetime
        t0_obspy = util.time.to_UTCDateTime(t0)
        t1_obspy = util.time.to_UTCDateTime(t1)

        # split receiver SEED-id to a dictionary
        receiver = util.receiver.receiver_to_dict(receiver)

        # 1. scan local archives for exact period
        stream = self._get_sds_waveforms(
            **dict(**receiver, starttime=t0_obspy, endtime=t1_obspy)
        )

        # 2. download remote per day
        if not stream and download:

            # list of days
            days = util.time.get_dates(t0, t1)

            # get streams per day
            stream = Stream()

            for day in days:

                stream += self._get_waveforms_for_date(
                    receiver=receiver,
                    date=day,
                    scan_sds=False,
                    download=True,
                    verb=verb-1,
                    **kwargs
                )

            # trim to asked time range
            stream.trim(starttime=t0_obspy, endtime=t1_obspy)

        return stream

    def _get_sds_waveforms(
        self, verb: int = 0, parallel: bool = None, **kwargs
    ):
        """
        Get the local waveforms from any of the local sds_read archives.

        Parameters
        ----------
        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        parallel : `bool`, optional
            Enable parallel processing using :method:`dask`. If `None`
            (default) ``self.parallel`` is used.

        **kwargs :
            Parameters passed to
            :meth:`obspy.clients.filesystem.sds.get_waveforms`.
            Required parameters are ``network``, ``station``, ``location``,
            ``channel``, ``starttime``, and ``endtime``.

        Returns
        -------
        stream : :class:`obspy.Stream`
            The requested waveforms.

        """
        # parallel?
        parallel = self.parallel if parallel is None else parallel

        # lock?
        locked = distributed and parallel

        # get seedid, extra arguments are ignored
        receiver = util.receiver.receiver_to_str(kwargs)

        # get sds write lock
        if locked:
            lock = distributed.Lock(receiver)
            lock.acquire()

        # feedback
        if verb > 0:

            print("Get waveforms for {} from {} until {}"
                  .format(receiver, kwargs['starttime'], kwargs['endtime']))

        dt = kwargs['endtime'] - kwargs['starttime']
        st = Stream()

        # Catch massive spill of InternalMSEEDWarning
        with obspyWarn.catch_warnings():

            obspyWarn.filterwarnings('error', 'InternalMSEEDWarning')
            obspyWarn.filterwarnings('error', 'Incompatible traces')

            # Examine multiple sds repositories
            for sds in self.sds_read:

                # get waveforms
                try:

                    st = sds.get_waveforms(**kwargs)

                except Warning as w:

                    if verb > 0:
                        print(f'Intercepted warning @ sds request: {w}')

                    continue

                if not st:

                    continue

                passed = self.check_duration(st, duration=dt,
                                             receiver=receiver, verb=verb-1)

                if passed:

                    if verb > 1:
                        print(_msg_loaded_archive.format(kwargs['starttime']))

                    break

        # release
        if locked:
            lock.release()

        return st

    def _get_waveforms_for_date(
        self, receiver: dict, date: pd.Timestamp, scan_sds: bool = True,
        download: bool = True, force_write: bool = False, verb: int = 0
    ):
        """
        Get the waveforms for a receiver and date.

        Parameters
        ----------
        receiver : `dict`
            Receiver dictionary with SEED-id keys:
            ['network', 'station', 'location', 'channel'].

        date : :class:`pd.Timestamp`
            The date.

        scan_sds : `bool`, optional
            If `True` (default), scan all local SDS archives listed in
            ``self.sds_read``.

        download : `bool`, optional
            If `True` (default), automatically download waveforms missing in
            all local SDS archives listed in ``self.sds_read`` using
            ``self.fdsn`` and ``self.vdms`` webservices. Data is added to
            ``self.sds_write``.

        force_write : `bool`, optional
            Force to write the stream to disk if `True`, even if it contains
            gaps. Defaults to `False`.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        Returns
        -------
        stream : :class:`obspy.Stream`
            The waveforms for ``receiver`` on ``date``.

        """

        # start time
        time = UTCDateTime(pd.to_datetime(date) +
                           pd.offsets.DateOffset(0, normalize=True))

        # get waveform args
        get_args = dict(**receiver, starttime=time, endtime=time + 86400)

        # store new waveform args
        set_args = dict(verb=verb-2, force_write=force_write)

        if verb > 0:
            print(
                "Get waveforms for {network}.{station}.{location}.{channel} "
                "from {starttime} until {endtime}".format(**get_args)
            )

        # 1. check sds
        if scan_sds:

            daystream = self._get_sds_waveforms(verb=verb-1, **get_args)

            if daystream:

                if verb > 1:

                    print(_msg_loaded_archive.format(time))

                return daystream

            else:

                if verb > 1:

                    print(_msg_no_data.format(time))

        # 2. check services
        if download:

            # attempt via fdsn
            if self.fdsn:

                if verb > 1:

                    print('Try FDSN.')

                try:

                    daystream = self.fdsn.get_waveforms(**get_args)

                    if self._sds_write_daystream(daystream, **set_args):

                        return daystream

                except (KeyboardInterrupt, SystemExit):

                    raise

                except FDSNNoDataException:

                    if verb > 1:

                        print(_msg_no_data.format(time))

                except Exception as e:

                    if verb > 0:
                        print(f'Intercepted error @ fdsn request: {e}')

                    if verb > 1:
                        print('-'*79)
                        track = traceback.format_exc()
                        print(track)
                        print('-'*79)

            # attempt via vdms
            if self.vdms:

                if verb > 1:

                    print('Try VDMS')

                try:

                    daystream = self.vdms.get_waveforms(**get_args)

                    if self._sds_write_daystream(daystream, **set_args):

                        return daystream

                    if verb > 1:

                        print(_msg_no_data.format(time))

                except (KeyboardInterrupt, SystemExit):

                    raise

                except Exception as e:

                    if verb > 0:
                        print(f'Intercepted error @ pyvdms request: {e}')

                    if verb > 1:
                        print('-'*79)
                        track = traceback.format_exc()
                        print(track)
                        print('-'*79)

        return Stream()

    def _test_waveforms_for_date(self, **kwargs):
        """
        Test get_waveforms_for_date.

        Parameters
        ----------
        **kwargs :
            Parameters passed to :meth:`_get_waveforms_for_date`.

        Returns
        -------
        flag : `int`
            Flag meaning: -2=failed, -1=missing, 0=not_validated, 1=passed.

        """
        try:

            stream = self._get_waveforms_for_date(**kwargs)

        except (KeyboardInterrupt, SystemExit):

            raise

        except Exception as e:

            if 'verb' in kwargs:

                if kwargs['verb'] > 0:
                    print(f'Intercepted error @ get_waveforms_for_date: {e}')

                if kwargs['verb'] > 1:
                    print('-'*79)
                    track = traceback.format_exc()
                    print(track)
                    print('-'*79)

            return -2

        passed = self.check_duration(
            stream,
            duration=86400.,
            receiver=util.receiver.receiver_to_str(kwargs['receiver'])
        )

        return 1 if passed else -1

    def get_preprocessed_waveforms(
        self, receiver: str, time: pd.Timestamp, preprocess: dict,
        duration: float = 86400., centered: bool = True,
        inventory: Inventory = None, substitute: bool = True,
        three_components: str = '12Z',  duration_check: bool = True,
        strict: bool = True, raise_error: bool = False, verb: int = 0,
        **kwargs
    ):
        """
        Get preprocessed waveforms from the clients given a SEED-id and an
        operations dictionary.

        Parameters
        ----------
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

        centered : `bool`, optional
            Controls the anchor type of ``time``. If `True`, ``time``
            corresponds to the center of the waveform time window, otherwise
            ``time`` is the left corner (start) of the time window. Defaults
            to `True`.

        inventory : :class:`obspy.Inventory`, optional
            Inventory object, including the instrument response.

        three_components: {'12Z', 'NEZ'}, optional
            Set the three-component orientation characters. Defaults to '12Z'.

        duration_check: `bool`, optional
            If `True` (default), verify the preprocessed stream duration using
            the ``strict`` method if enabled.

        strict: `bool`, optional
            If `True` (default), the samples difference between the expected
            and the obtained number of samples is zero. If `False`, a
            two-sample difference is allowed (solving nearest sample related
            differences).

        raise_error : `bool`, optional
            If `True` raise when an error occurs. Otherwise a warning is given.
            Defaults to `False`.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
            Parameters passed to :meth:`get_waveforms`.

        Returns
        -------
        stream : :class:`obspy.Stream`
            The requested waveforms after preprocessing.

        """
        # check if receiver SEED-id is valid
        util.receiver.check_receiver(receiver, allow_wildcards=False,
                                     raise_error=True)

        three_components = three_components or '12Z'
        assert three_components == '12Z' or three_components == 'NEZ', (
            '``three_components`` should be either "12Z" or "NEZ"!'
        )

        ch = receiver.split('.')[-1]

        # radial or transverse component? Request all channels manually.
        if substitute and ch[-1] in 'RT':

            st = Stream()

            for c in three_components:  # '12NEZ'??

                st = st + self.get_waveforms(
                    receiver=receiver[:-1]+c,
                    time=time,
                    duration=duration,
                    centered=centered,
                    verb=verb,
                    **kwargs
                )

        else:

            st = self.get_waveforms(
                receiver=receiver,
                time=time,
                duration=duration,
                centered=centered,
                verb=verb,
                **kwargs
            )

        if verb > 2:

            print(st)

        if not isinstance(st, Stream) or len(st) == 0:

            return Stream()

        if centered:

            t0 = util.time.to_UTCDateTime(time) - duration/2

        else:

            t0 = util.time.to_UTCDateTime(time)

        t1 = t0 + duration

        st = xcorr_preprocess(
            waveforms=st,
            operations=preprocess[ch],
            inventory=inventory,
            starttime=t0,
            endtime=t1,
            raise_error=raise_error,
            verb=verb-1,
        )

        # expects a stream with one trace
        if not isinstance(st, Stream) or len(st) != 1:

            if raise_error:

                raise ValueError('No stream with single trace returned '
                                 'after preprocessing.')

            return Stream()

        # check stream duration
        if duration_check:

            diff = duration / st[0].stats.delta - st[0].stats.npts

            if diff > (0 if strict else 2):

                if raise_error:

                    raise ValueError(
                        ('Preprocessed stream fails {}duration check: '
                         '{} sample difference.')
                        .format('strict ' if strict else '', diff)
                    )

                return Stream()

        return st

    def _test_preprocessed_waveforms(self, sampling_rate: float = None,
                                     **kwargs):
        """
        Test get_preprocessed_waveforms.

        Parameters
        ----------
        sampling_rate : `float`, optional
            The desired final sampling rate of the stream (in Hz).

        **kwargs :
            Parameters passed :meth:`get_preprocessed_waveforms`.

        Returns
        -------
        flag : `int`
            Flag meaning: -2=failed, -1=missing, 0=not_validated, 1=passed.

        """
        if 'duration' not in kwargs:
            kwargs['duration'] = 86400.

        kwargs['centered'] = False
        kwargs['raise_error'] = True
        kwargs['duration_check'] = False

        try:

            stream = self.get_preprocessed_waveforms(**kwargs)

        except (KeyboardInterrupt, SystemExit):

            raise

        except RuntimeError:

            return -2

        passed = self.check_duration(stream, duration=kwargs['duration'],
                                     receiver=kwargs['receiver'],
                                     sampling_rate=sampling_rate)

        return 1 if passed else -1

    def get_pair_preprocessed_waveforms(
        self, pair, **kwargs
    ):
        """
        Get preprocessed waveforms from the clients given a receiver couple
        SEED-id and an operations dictionary.

        Parameters
        ----------
        pair : str or :mod:`~xarray.DataArray`
            Receiver couple separated by `separator`. Each receiver is
            specified by a SEED-id string:
            '{network}.{station}.{location}.{channel}'.

        **kwargs :
            Parameters passed to :meth:`get_waveforms` via
            :meth:`get_preprocessed_waveforms`.

        Returns
        -------
        stream : :class:`obspy.Stream`
            The requested waveforms after preprocessing.

        """
        # split
        rA, rB = util.receiver.split_pair(pair)

        # get streams per receiver
        stream = (
            self.get_preprocessed_waveforms(rA, **kwargs) +
            self.get_preprocessed_waveforms(rB, **kwargs)
        )

        return stream

    def data_availability(
        self, pairs_or_receivers: list, times: pd.DatetimeIndex,
        extend_days: int = None, substitute: bool = False,
        three_components: str = None, parallel: bool = None,
        verb: int = 0, **kwargs
    ):
        """
        Verify the waveform data availability for receivers and times.

        Parameters
        ----------
        pairs_or_receivers : `list`
            List of receivers or pairs (receiver couple separated by a '-').
            Each receiver should be specified by a SEED-id string of format
            '{network}.{station}.{location}.{channel}'.

        times : `pd.DatetimeIndex`
            Sequence of dates with `freq`="D".

        extend_days : `int`, optional
            Extend ``times`` with n-days at both left and right edges.

        substitute : `bool`, optional
            If `True`, convert radial 'R' and transverse 'T' rotated
            orientation codes automatically to ``three_components``.
            Defaults to `False`.

        three_components: {'12Z', 'NEZ'}, optional
            Set the three-component orientation characters for ``substitute``.
            Defaults to '12Z'.

        parallel : `bool`, optional
            Enable parallel processing using :method:`dask`. If `None`
            (default) ``self.parallel`` is used.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 0.

        **kwargs :
            Parameters passed to :meth:`_get_waveforms_for_date`

        Returns
        -------
        status : :class:`xarray.DataArray`
            Data availability flags as an N-D labelled array with dimensions
            ``times`` and ``receivers``.

        """
        assert isinstance(times, pd.DatetimeIndex) and times.freqstr == 'D', (
            '``times`` should be a pandas.DatetimeIndex with freq="D"!'
        )
        extend_days = extend_days or 0

        assert isinstance(extend_days, int), (
            '``extend_days`` should be of type `int`!'
        )

        parallel = self.parallel if parallel is None else parallel

        if parallel and not dask:

            raise RuntimeError('Dask is required but cannot be found!')

        # get all receivers from pairs
        receivers = []

        for p in pairs_or_receivers:

            receivers += util.receiver.split_pair(
                p,
                to_dict=False,
                substitute=substitute,
                three_components=three_components,
            )

        receivers = sorted(list(set(receivers)))

        # time
        if extend_days > 0:

            times = pd.date_range(
                start=times[0] - pd.offsets.DateOffset(days=extend_days),
                end=times[-1] + pd.offsets.DateOffset(days=extend_days),
                freq=times.freqstr
            )

        # construct status xarray object
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
            'substitute': np.byte(substitute),
            'three_components': three_components,
        }

        if verb:
            print('Receivers :')
            for rec in status.receiver:
                print(f'    {rec.values}')
            print(f'Times : {times[0]} to {times[-1]}')
            print(f'Verify {status.size} receiver time combinations.')

        if parallel:
            lazy_flags = sds_locks = []
            for rec in status.receiver:
                sds_locks.append(distributed.Lock(str(rec.values)))

        # evaluate receiver and days
        for receiver in status.receiver:

            rec_dict = util.receiver.receiver_to_dict(str(receiver.values))

            for time in status.time:

                if status.loc[{'receiver': receiver, 'time': time}] == 1:

                    continue

                args = dict(
                    receiver=rec_dict,
                    date=time.values,
                    verb=verb-1,
                    **kwargs
                )

                if parallel:

                    lazy_flags.append(
                        dask.delayed(self._test_waveforms_for_date)(**args)
                    )

                else:

                    status.loc[{'receiver': receiver, 'time': time}] = (
                        self._test_waveforms_for_date(**args)
                    )

        if parallel:
            status.values = np.array(
                dask.compute(lazy_flags)[0]
            ).reshape(status.shape)

        if verb:

            verified = np.sum(status.values != 0)

            pcnt = 100 * verified / status.size

            print('Verified : {} of {} ({:.1f}%)'
                  .format(verified, status.size, pcnt))
            print('Overall availability : {:.2f}%'
                  .format(100 * np.sum(status.values == 1) / status.size))
            print('Receiver availability')

            for rec in status.receiver:

                pcnt = 100*np.sum(
                    status.loc[{'receiver': rec}].values == 1
                ) / status.time.size
                print('    {} : {:.2f}%'.format(rec.values, pcnt))

        return status

    def data_preprocessing(
        self, pairs_or_receivers: list, time: pd.Timedelta,
        preprocess: dict, inventory: Inventory, substitute: bool = False,
        three_components: str = None, parallel: bool = None,
        verb: int = 0, **kwargs
    ):
        """
        Verify the waveform data preprocessing for receivers and time.

        Parameters
        ----------
        pairs_or_receivers : `list`
            List of receivers or pairs (receiver couple separated by a '-').
            Each receiver should be specified by a SEED-id string of format
            '{network}.{station}.{location}.{channel}'.

        time : `pd.Timestamp`
            Date of the waveform to test the ``preprocess`` operations.

        preprocess : `dict`
            Preprocessing operations dictionary, containing a list of
            operations per SEED channel as key. Each list item should be a
            tuple ('operation', {parameters}).
            Use :func:`xcorr.preprocess.help` to list all valid operations and
            their documentation.

        inventory : :class:`obspy.Inventory`, optional
            Inventory object, including the instrument response.

        substitute : `bool`, optional
            If `True`, convert radial 'R' and transverse 'T' rotated
            orientation codes automatically to ``three_components``.
            Defaults to `False`.

        three_components: {'12Z', 'NEZ'}, optional
            Set the three-component orientation characters for ``substitute``.
            Defaults to '12Z'.

        parallel : `bool`, optional
            Enable parallel processing using :method:`dask`. If `None`
            (default) ``self.parallel`` is used.

        verb : {0, 1, 2, 3, 4}, optional
            Level of verbosity. Defaults to 1.

        **kwargs :
            Parameters passed to :meth:`_get_waveforms_for_date`

        Returns
        -------
        status : :class:`xarray.DataArray`
            Data preprocessing status N-D labelled array with dimensions
            ``time`` and ``receiver``.

        """

        assert isinstance(preprocess, dict), (
            '``preprocess`` should be of type `dict`.'
        )
        time = pd.to_datetime(time)

        parallel = self.parallel if parallel is None else parallel

        if parallel and not dask:
            raise RuntimeError('Dask is required but cannot be found!')

        # get all receivers from pairs
        receivers = []

        for p in pairs_or_receivers:
            receivers += util.receiver.split_pair(
                p, to_dict=False, substitute=False
            )

        receivers = sorted(list(set(receivers)))

        # time
        times = pd.date_range(
            start=time - pd.offsets.DateOffset(days=0),
            end=time - pd.offsets.DateOffset(days=0),
            freq='D'
        )

        # construct status xarray object
        status = xr.DataArray(
            data=np.zeros((len(receivers), len(times)), dtype=np.byte),
            coords=[np.array(receivers, dtype=object), times],
            dims=['receiver', 'time'],
            name='status',
            attrs={
                'long_name': 'Data preprocessing status',
                'standard_name': 'data_preprocessing_status',
                'units': '-',
                'valid_range': np.byte([-2, 1]),
                'flag_values': np.byte([-2, -1, 0, 1]),
                'flag_meanings': 'missing failed not_validated passed',
            }
        )
        status.receiver.attrs = {
            'long_name': 'Receiver SEED-id',
            'standard_name': 'receiver_seed_id',
            'units': '-',
            'substitute': np.byte(substitute),
            'three_components': three_components,
            'preprocess': operations_to_json(preprocess),
        }

        if verb:
            print('Receivers :')
            for rec in status.receiver:
                print(f'    {rec.values}')
            print(f'Reference time : {time}')
            print(f'Verify {status.size} receivers.')

        if parallel:
            lazy_flags = sds_locks = []
            for rec in status.receiver:
                sds_locks.append(distributed.Lock(str(rec.values)))

        # evaluate receiver and days
        for receiver in status.receiver:

            for time in status.time:

                if status.loc[{'receiver': receiver, 'time': time}] == 1:

                    continue

                args = dict(
                    receiver=str(receiver.values),
                    time=time.values,
                    preprocess=preprocess,
                    inventory=inventory,
                    substitute=receiver.attrs['substitute'] == 1,
                    three_components=receiver.attrs['three_components'],
                    raise_error=True,
                    centered=False,
                    verb=verb-1,
                    **kwargs,
                )

                if parallel:

                    lazy_flags.append(
                        dask.delayed(self._test_preprocessed_waveforms)(**args)
                    )

                else:

                    status.loc[{'receiver': receiver, 'time': time}] = (
                        self._test_preprocessed_waveforms(**args)
                    )

        if parallel:
            status.values = np.array(
                dask.compute(lazy_flags)[0]
            ).reshape(status.shape)

        if verb:

            verified = np.sum(status.values != 0)

            pcnt = 100 * verified / status.size

            print('Verified : {} of {} ({:.1f}%)'
                  .format(verified, status.size, pcnt))
            print('Overall preprocessing : {:.2f}% passed'
                  .format(100 * np.sum(status.values == 1) / status.size))
            print('Receiver preprocessing')

            for rec in status.receiver:

                passed = np.all(status.loc[{'receiver': rec}].values == 1)

                print('    {} :'.format(rec.values),
                      'passed' if passed else 'failed')

        return status
