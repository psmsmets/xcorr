r"""

:mod:`core.init` -- Init
========================

Initiate an xcorr N-D labelled set of data arrays.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import obspy
import json


# Relative imports
from .accessors import register_xcorr_dataset_accessor
from .. import stream, util


__all__ = ['init']


def init(
    pair: str, starttime: pd.Timestamp, endtime: pd.Timestamp,
    preprocess: dict, attrs: dict, sampling_rate: float,
    window_length: float = 86400., window_overlap: float = 0.875,
    clip_lag=None, unbiased_cc: bool = False, closed: str = 'left',
    dtype: np.dtype = np.float32, inventory: obspy.Inventory = None,
    stationary_poi: dict = None, hash_waveforms: bool = False,
):
    """
    Initiate an xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    pair : `str`
        Receiver couple separated by a '-'. Each receiver is specified by its
        SEED-id string: '{network}.{station}.{location}.{channel}'.

    starttime : `pd.Timestamp`
        Start time for generating cross-correlation time windows.

    endtime : `pd.Timestamp`
        End time for generating cross-correlation time windows.

    preprocess : `dict`
        Stream preprocessing operations dictionary, containing a list of
        operations per SEED channel as key. Each list item should be a tuple
        ('operation', {parameters}).
        Use :func:`xcorr.stream.process.help` to list all valid operations and
        their documentation.

    attrs : `dict`
        Dictionary with global attributes the comply with COARDS and CF
        conventions. Following keys are mandatory:
        ['institution', 'author', 'source']

    sampling_rate : `float`
        Sampling rate for the cross-correlation lag time.

    window_length : `float`, optional
        Cross-correlation window length, in seconds. Defaults to 86400s.

    window_overlap : `float`, optional
        Cross-correlation window overlap, [0,1). Defaults to 0.875.

    clip_lag : `float` or `tuple`, optional
        Clip the cross-correlation lag time. Defaults to `None`.
        If ``clip_lag`` is a single value ``lag`` is clipped symmetrically
        around zero with radius ``clip_lag`` (`float` seconds).
        Provide a `tuple` with two elements, (lag_min, lag_max), both of type
        `float` to clip a specific lag range of interest.

    unbiased_cc : `bool`, optional
        Automatically bias correct the cross-correlation estimate in place if
        `True`. Default is `False`.

    closed : {`None`, 'left', 'right'}, optional
        Make the time interval closed with respect to the given frequency to
        the 'left' (default), 'right', or both sides (`None`).

    dtype : `np.dtype`, optional
        Set the cross-correlation estimate dtype. Defaults to `np.float32`.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    stationary_poi : `dict`, optional
        Specify a point-of-interest `dict` with keys ['longitude','latitude']
        in decimal degrees to obtain a relative distance.
        If `None` (default), the receiver pair geodetic distance is calculated.

    hash_waveforms : `bool`, optional
        Create a sha256 hash of the preprocessed waveforms used for each
        correlation step. Caution: hashing can take some time (~10s per step).

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The initiated `xcorr` N-D labelled set of data arrays.

    """
    # check attributes
    if not isinstance(attrs, dict):
        raise TypeError('attrs should be a dictionary')
    if 'institution' not in attrs:
        raise KeyError("attrs['institution'] = 'Institution, department'!")
    if 'author' not in attrs:
        raise KeyError("attrs['author'] = 'Name - E-mail'!")
    if 'source' not in attrs:
        raise KeyError("attrs['source'] = 'Data source description'!")

    # check pair
    if not isinstance(pair, str):
        raise TypeError("pair should be receiver pair string!")
    # config
    delta = 1/sampling_rate
    npts = int(window_length*sampling_rate)
    encoding = dict(zlib=True, complevel=9, shuffle=True)

    # start dataset
    dataset = xr.Dataset()

    # global attributes
    dataset.attrs = {
        'title': (
            (attrs['title'] if 'title' in attrs else '') +
            ' Cross-correlations - {}{}'
            .format(
                starttime.strftime('%Y.%j'),
                ' to {}'.format(endtime.strftime('%Y.%j'))
                if starttime.strftime('%Y.%j') != endtime.strftime('%Y.%j')
                else ''
            )
        ).strip(),
        'institution': attrs['institution'],
        'author': attrs['author'],
        'source': attrs['source'],
        'history': 'Created @ {}'.format(pd.to_datetime('now')),
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
        'comment': attrs['comment'] if 'comment' in attrs else 'n/a',
        'Conventions': 'CF-1.9',
        'xcorr_version': util.metadata.version,
        'dependencies_version': util.metadata.list_versions(as_str=True),
    }

    # pair
    dataset.coords['pair'] = np.array([pair], dtype=object)
    dataset.pair.attrs = {
        'long_name': 'Cross-correlation receiver pair',
        'standard_name': 'receiver_pair',
        'units': '-',
        'preprocess': stream.process.hash_operations(preprocess),
    }

    # time
    dataset.coords['time'] = pd.date_range(
        start=starttime,
        end=endtime,
        freq='{0:.0f}s'.format(window_length*(1-window_overlap)),
        closed=closed,
    )
    dataset.time.attrs = {
        'window_length': window_length,
        'window_overlap': window_overlap,
        'closed': closed,
    }

    # time lag
    lag = util.cc.lag(npts, delta, pad=True)
    if clip_lag is not None:
        msg = ('``clip_lag`` should be in seconds of type `float` or of type'
               '`tuple` with length 2 specifying start and end.')
        if isinstance(clip_lag, float):
            clip_lag = abs(clip_lag)
            clip_lag = tuple(-clip_lag, clip_lag)
        elif isinstance(clip_lag, tuple) and len(clip_lag) == 2:
            if (
                not(isinstance(clip_lag[0], float)) or
                not(isinstance(clip_lag[0], float))
            ):
                raise TypeError(msg)
        else:
            raise TypeError(msg)
        nmin = np.argmin(abs(lag - clip_lag[0]))
        nmax = np.argmin(abs(lag - clip_lag[1]))
    else:
        nmin = 0
        nmax = 2*npts-1
    dataset.coords['lag'] = lag[nmin:nmax]
    dataset.lag.attrs = {
        'long_name': 'Time lag',
        'standard_name': 'time_lag',
        'units': 's',
        'sampling_rate': float(sampling_rate),
        'delta': float(delta),
        'npts': int(npts),
        'clip_lag': (
            np.array(clip_lag if clip_lag is not None else [])
        ),
        'index_min': int(nmin),
        'index_max': int(nmax),
    }

    # pair distance
    dataset['distance'] = (
        ('pair'),
        np.ones((1), dtype=np.float64) * util.receiver.get_pair_distance(
            pair=pair,
            inventory=inventory,
            poi=stationary_poi,
            ellipsoid='WGS84',
            km=True,
        ),
        {
            'long_name': 'receiver pair distance',
            'standard_name': 'receiver_pair_distance',
            'units': 'km',
            'description': (
                ('relative to poi' if stationary_poi else 'absolute') +
                ' WGS84 geodetic distance'
            ),
            'relative_to_poi': (
                json.dumps(stationary_poi) if stationary_poi else 'n/a'
            ),
        },
        encoding,
    )

    # status
    dataset['status'] = (
        ('pair', 'time'),
        np.zeros((1, dataset.time.size), dtype=np.byte),
        {
            'long_name': 'processing status',
            'standard_name': 'processing_status',
            'units': '-',
            'valid_range': np.byte([-1, 1]),
            'flag_values': np.byte([-1, 0, 1]),
            'flag_meanings': 'missing_data not_processed processed',
        },
        encoding,
    )

    # hash waveforms
    if hash_waveforms:
        dataset['hash'] = (
            ('pair', 'time'),
            np.array([['n/a']*dataset.time.size], dtype=object),
            {
                'long_name': 'pair preprocessed stream hash',
                'standard_name': 'pair_preprocessed_stream_hash',
                'units': '-',
                'description': (
                    "Openssl SHA256 hash of the pair preprocessed waveform "
                    "stream. Be aware that stream/pair order matters! "
                    "The hash is updated per `obspy.Trace` and includes the "
                    "stats with keys=['network', 'station', 'location', "
                    "'channel', 'starttime', 'endtime', 'sampling_rate', "
                    "'delta', 'npts'], sorted and dumped to json with 4 "
                    "character space indentation and separators ',' and ':', "
                    "followed by the hash of each sample byte representation."
                ),
            },
            encoding,
        )

    # pair offset
    dataset['pair_offset'] = (
        ('pair', 'time'),
        np.zeros((1, dataset.time.size), dtype=np.float64),
        {
            'long_name': 'receiver pair start sample offset',
            'standard_name': 'receiver_pair_start_sample_offset',
            'units': 's',
            'description': (
                'offset = receiver[0].starttime - receiver[1].starttime'
            ),
        },
        encoding,
    )

    # time offset
    dataset['time_offset'] = (
        ('pair', 'time'),
        np.zeros((1, dataset.time.size), dtype=np.float64),
        {
            'long_name': 'first receiver start sample offset',
            'standard_name': 'first_receiver_start_sample_offset',
            'units': 's',
            'description': (
                'offset = receiver[0].starttime - time + window_length/2'
            ),
        },
        encoding,
    )

    # cc
    dataset['cc'] = (
        ('pair', 'time', 'lag'),
        np.zeros((1, dataset.time.size, dataset.lag.size), dtype=dtype),
        {
            'long_name': 'Cross-correlation Estimate',
            'standard_name': 'cross-correlation_estimate',
            'units': '-',
            'add_offset': dtype(0.),
            'scale_factor': dtype(1.),
            'valid_range': dtype([-1., 1.]),
            'normalize': np.byte(1),
            'bias_correct': np.byte(unbiased_cc),
            'unbiased': np.byte(0),
        },
        encoding,
    )

    # add metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )

    # register accessor
    register_xcorr_dataset_accessor()

    return dataset
