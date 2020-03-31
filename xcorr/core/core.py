# -*- coding: utf-8 -*-
"""
"""

# Absolute imports
import numpy as np
import xarray as xr
import pandas as pd
import obspy
import scipy
import json
import warnings
import os
from datetime import datetime

# Relative imports
from ..version import version as __version__
from ..clients import Client
from .. import cc
from .. import util
from ..preprocess import (
    hash_operations,
    check_operations_hash,
    operations_to_dict,
    preprocess_operations_to_dict,
    preprocess_operations_to_json
)


__all__ = ['init', 'read', 'write', 'merge', 'process', 'bias_correct']


def init(
    pair: str, starttime: datetime, endtime: datetime, preprocess: dict,
    attrs: dict, sampling_rate: float, window_length: float = 86400.,
    window_overlap: float = 0.875,
    clip_lag=None, unbiased_cc: bool = False,
    closed: str = 'left', dtype: np.dtype = np.float32,
    inventory: obspy.Inventory = None, stationary_poi: dict = None,
):
    r"""Initiate an xcorr N-D labeled data array.

    Parameters:
    -----------
    pair : `str`
        Receiver couple separated by a '-'. Each receiver is specified by its
        SEED-id string: '{network}.{station}.{location}.{channel}'.

    starttime : `datetime.datetime`
        Start time for generating crosscorrelation time windows.

    endtime : `datetime.datetime`
        End time for generating crosscorrelation time windows.

    preprocess : `dict`
        Preprocessing operations dictionary, containing a list of operations
        per SEED channel as key. Each list item should be a tuple
        ('operation', {parameters}).
        Use :func:`xcorr.preprocess.help` to list all valid operations and
        their documentation.

    attrs : `dict`
        Dictionary with global attributes the comply with COARDS and CF
        conventions. Following keys are required:
        ['institution', 'author', 'source']

    sampling_rate : `float`
        Sampling rate for the crosscorrelation lag time.

    window_length : `float`, optional
        Crosscorrelation window length, in seconds. Defaults to 86400s.

    window_overlap : `float`, optional
        Crosscorrelation window overlap, [0,1). Defaults to 0.875.

    clip_lag : :class:`pd.Timedelta` or :class:`pd.TimedeltaIndex`
        Clip the crosscorrelation lag time. Defaults to `None`. If ``clip_lag``
        is a single value of type :class:`pd.Timedelta` ``lag`` is clipped
        symmetrically around zero with radius ``clip_lag``.
        Provide a :class:`pd.TimedeltaIndex` of size 2 to clip a specific lag
        range of interest.

    unbiased_cc : `bool`, optional
        Automatically bias correct the crosscorrelation estimate in place if
        `True`. Default is `False`.

    closed : {`None`, 'left', 'right'}, optional
        Make the time interval closed with respect to the given frequency to
        the 'left' (default), 'right', or both sides (`None`).

    dtype : `np.dtype`, optional
        Set the crosscorrelation estimate dtype. Defaults to `np.float32`.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    stationary_poi : `dict`, optional
        Specify a point-of-interest `dict` with keys ['longitude','latitude']
        in decimal degrees to obtain a relative distance.
        If `None` (default), the receiver pair geodetic distance is calculated.

    Returns:
    --------
    dataset : :class:`xarray.Dataset`
        The initiated `xcorr` N-D labeled data array.

    """
    # check
    assert 'institution' in attrs, (
        "attrs['institution'] = 'Institution, department'!"
    )
    assert 'author' in attrs, (
        "attrs['author'] = 'Name - E-mail'!"
    )
    assert 'source' in attrs, (
        "attrs['source'] = 'Data source description'!"
    )
    assert isinstance(pair, str), (
        "pair should be receiver pair string!"
    )
    # config
    delta = 1/sampling_rate
    npts = int(window_length*sampling_rate)
    encoding = {'zlib': True, 'complevel': 9}

    # start dataset
    dataset = xr.Dataset()

    # global attributes
    dataset.attrs = {
        'title': (
            (attrs['title'] if 'title' in attrs else '') +
            ' Crosscorrelations - {}'
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
        'xcorr_version': __version__,
        'dependencies_version': dependencies_version(as_str=True),
    }

    # pair
    dataset.coords['pair'] = np.array([pair], dtype=object)
    dataset.pair.attrs = {
        'long_name': 'Crosscorrelation receiver pair',
        'standard_name': 'receiver_pair',
        'units': '-',
        'preprocess': hash_operations(preprocess),
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

    # lag
    lag = cc.lag(npts, delta, pad=True)
    if clip_lag is not None:
        if isinstance(clip_lag, pd.Timedelta):
            clip_lag = pd.to_timedelta((-np.abs(clip_lag), np.abs(clip_lag)))
        elif not (
            isinstance(clip_lag, pd.TimedeltaIndex) and len(clip_lag) == 2
        ):
            raise TypeError(
                'clip_lag should be of type ~pandas.Timedelta '
                'or ~pandas.TimedeltaIndex with length 2 '
                'specifying start and end.'
            )
        nmin = np.argmin(abs(lag - clip_lag[0] / util._one_second))
        nmax = np.argmin(abs(lag - clip_lag[1] / util._one_second))
    else:
        nmin = 0
        nmax = 2*npts-1
    dataset.coords['lag'] = pd.to_timedelta(lag[nmin:nmax], unit='s')
    dataset.lag.attrs = {
        'long_name': 'Lag time',
        'standard_name': 'lag_time',
        'sampling_rate': float(sampling_rate),
        'delta': float(delta),
        'npts': int(npts),
        'pad': np.byte(True),
        'clip': int(clip_lag is not None),
        'clip_lag': (
            clip_lag.values / util._one_second
            if clip_lag is not None else None
        ),
        'index_min': int(nmin),
        'index_max': int(nmax),
    }

    # pair distance
    dataset['distance'] = (
        ('pair'),
        np.ones((1), dtype=np.float64) * util.get_pair_distance(
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
    )

    # status
    dataset['status'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.byte),
        {
            'long_name': 'processing status',
            'standard_name': 'processing_status',
            'units': '-',
            'valid_range': np.byte([-1, 1]),
            'flag_values': np.byte([-1, 0, 1]),
            'flag_meanings': 'missing_data not_processed processed',
        },
    )

    # status
    dataset['hash'] = (
        ('pair', 'time'),
        np.array([['n/a']*len(dataset.time)], dtype=object),
        {
            'long_name': 'pair preprocessed stream hash',
            'standard_name': 'pair_preprocessed_stream_hash',
            'units': '-',
            'description': (
                "Openssl SHA256 hash of the pair preprocessed waveform "
                "stream. Be aware that stream/pair order matters! "
                "The hash is updated per `obspy.Trace` and includes the "
                "stats with keys=['network', 'station', 'location', "
                "'channel', 'starttime', 'endtime', 'sampling_rate', 'delta', "
                "'npts'], sorted and dumped to json with 4 character space "
                "indentation and separators ',' and ':', followed by the hash "
                "of each sample byte representation."
            ),
        },
    )

    # pair offset
    dataset['pair_offset'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.timedelta64),
        {
            'long_name': 'receiver pair start sample offset',
            'standard_name': 'receiver_pair_start_sample_offset',
            'description': (
                'offset = receiver[0].starttime - receiver[1].starttime'
            ),
        },
    )

    # time offset
    dataset['time_offset'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.timedelta64),
        {
            'long_name': 'first receiver start sample offset',
            'standard_name': 'first_receiver_start_sample_offset',
            'description': (
                'offset = receiver[0].starttime - time + window_length/2'
            ),
        },
    )

    # cc
    dataset['cc'] = (
        ('pair', 'time', 'lag'),
        np.zeros((1, len(dataset.time), len(dataset.lag)), dtype=dtype),
        {
            'long_name': 'Crosscorrelation Estimate',
            'standard_name': 'crosscorrelation_estimate',
            'units': '-',
            'add_offset': dtype(0.),
            'scale_factor': dtype(1.),
            'valid_range': dtype([-1., 1.]),
            'normalize': np.byte(1),
            'bias_correct': np.byte(unbiased_cc),
            'unbiased': np.byte(0),
        },
        encoding
    )

    if unbiased_cc:
        dataset['w'] = get_weights(dataset.lag, dtype=dtype)

    # add metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset,
        metadata_only=True,
    )

    return dataset


def read(
    path: str, extract: bool = False, load_and_close: bool = False,
    fast: bool = False, quick_and_dirty: bool = False, debug: bool = False
):
    r"""Read an xcorr N-D labeled data array from a netCDF4 file.

    Parameters:
    -----------
    path : `str`
        NetCDF4 filename.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    load_and_close : `bool`, optional
        Load file to memory and close if `True` (default).

    fast : `bool`, optional
        Omit verifying the `sha256_hash` if `True`. Default is `False`.

    quick_and_dirty : `bool`, optional
        Omit verifying both the `sha256_hash` and `sha256_hash_metadata`
        if `True`. Default is `False`.

    debug : `bool`, optional
        If `True` some extra debug information is printed to the screen.
        Default is `False`.

    Returns:
    --------
    dataset : :class:`xarray.Dataset`
        The initiated `xcorr` N-D labeled data array.

    """
    if not os.path.isfile(path):
        return False
    dataset = xr.open_dataset(path)
    if 'xcorr_version' not in dataset.attrs:
        return False

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)

    # calculate some hashes
    if not quick_and_dirty:
        sha256_hash_metadata = util.hasher.hash_Dataset(
            dataset,
            metadata_only=True
        )
        if sha256_hash_metadata != dataset.sha256_hash_metadata:
            warnings.warn(
                'Dataset metadata sha256 hash is inconsistent.',
                UserWarning
            )
            if debug:
                print(
                    'sha256 hash metadata in ncfile :',
                    dataset.sha256_hash_metadata
                )
                print(
                    'sha256 hash metadata computed  :',
                    sha256_hash_metadata
                )
    if not (quick_and_dirty or fast):
        sha256_hash = util.hasher.hash_Dataset(
            dataset,
            metadata_only=False
        )
        if sha256_hash != dataset.sha256_hash:
            warnings.warn(
                'Dataset sha256 hash is inconsistent.',
                UserWarning
            )
            if debug:
                print('sha256 hash in ncfile :', dataset.sha256_hash)
                print('sha256 hash computed  :', sha256_hash)

    if debug:
        print('{p} #(status==1): {n} of {m}'.format(
            p=path,
            n=np.sum(dataset.status.values == 1),
            m=dataset.time.size,
        ))

    if np.sum(dataset.status.values == 1) == 0 and not debug:
        dataset.close()
        return False

    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    if load_and_close:
        dataset.load().close()

    return dataset


def write(
    dataset: xr.Dataset, path: str, close: bool = True,
    force_write: bool = False
):
    r"""Write an xcorr N-D labeled data array to a netCDF4 file using a
    temporary file and replacing the final destination.

    Before writing the dataset metadata and data hash hashes are verified and
    updated if necessary. This changes the dataset attributes in place.

    Parameters:
    -----------
    dataset : :class:`xarray.Dataset`
        The `xcorr` N-D labeled data array.

    path : `str`
        The netCDF4 filename.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    close : `bool`, optional
        Close the data `True` (default).

    force_write : `bool`, optional
        Always write file if `True` even if its empty. Default is `False`.

    """
    if (
        np.sum(dataset.status.values == 1) == 0 or
        (np.sum(dataset.status.values == -1) == 0 and force_write)
    ):
        warnings.warn(
            'Dataset contains no data. No need to save it.',
            UserWarning
        )
        return

    # Verify metadata hash
    sha256_hash_metadata = (
        util.hasher.hash_Dataset(dataset, metadata_only=True)
    )
    if sha256_hash_metadata != dataset.sha256_hash_metadata:
        warnings.warn(
            'Dataset metadata sha256 hash is updated.',
            UserWarning
        )
        dataset.attrs['sha256_hash_metadata'] = sha256_hash_metadata

    print('Write dataset as "{}"'.format(path), end=': ')
    abspath, file = os.path.split(os.path.abspath(path))
    if not os.path.exists(abspath):
        os.makedirs(abspath)

    tmp = os.path.join(
        abspath,
        '{f}.{t}'.format(f=file, t=int(datetime.now().timestamp()*1e3))
    )

    if close:
        print('Close', end='. ')
        dataset.close()

    # calculate dataset hash
    print('Hash', end='. ')
    dataset.attrs['sha256_hash'] = (
        util.hasher.hash_Dataset(dataset, metadata_only=False)
    )

    # convert preprocess operations
    preprocess_operations_to_json(dataset.pair)

    print('To temporary netcdf', end='. ')
    dataset.to_netcdf(path=tmp, mode='w', format='NETCDF4')
    print('Replace', end='. ')
    os.replace(tmp, os.path.join(abspath, file))
    print('Done.')

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)


def merge(
    datasets: list, extract: bool = True, merge_versions: bool = False,
    debug: bool = False, **kwargs
):
    r"""Merge a list of xcorr N-D labeled data arrays.

    Parameters:
    -----------
    datasets : `list`
        A list with either a `str` specifying the netCDF4 path or a
        :class:`xarray.Dataset` containing the `xcorr` N-D labeled data array.

    extract : `bool`, optional
        Mask crosscorrelation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    merge_versions : `bool`, optional
        Ignore data arrays with different `xcorr` versions.

    debug : `bool`, optional
        If `True` some extra debug information is printed to the screen.
        Default is `False`.

    kwargs :
        Additional parameters provided to :func:`read`.

    Returns:
    --------
    datasets : :class:`xarray.Dataset`
        The merged `xcorr` N-D labeled data array.

    """
    dsets = None
    for ds in datasets:
        if isinstance(ds, str):
            if not os.path.isfile(ds):
                warnings.warn(
                    'Datasets item "{}" does not exists. Item skipped.'
                    .format(ds),
                    UserWarning
                )
                continue
            ds = read(ds, extract=False, **kwargs)
        elif isinstance(ds, xr.Dataset):
            preprocess_operations_to_dict(ds.pair)
        else:
            warnings.warn(
                (
                    'Datasets item should be of type `str` '
                    'or :class:`xarray.DataSet`! Item skipped.'
                ),
                UserWarning
            )
            continue
        if debug:
            print("\n# Open and inspect xcorr dataset:\n\n", ds, "\n")
        if ds is False:
            continue
        if not isinstance(dsets, xr.Dataset):
            dsets = ds.copy()
            continue
        if (
            dsets.pair.preprocess['sha256_hash'] !=
            ds.pair.preprocess['sha256_hash']
        ):
            warnings.warn(
                'Dataset preprocess hash does not match. Item skipped.',
                UserWarning
            )
            continue
        if dsets.sha256_hash_metadata != ds.sha256_hash_metadata:
            warnings.warn(
                'Dataset metadata hash does not match. Item skipped.',
                UserWarning
            )
            continue
        if dsets.xcorr_version != ds.xcorr_version:
            if merge_versions:
                warnings.warn(
                    'Dataset xcorr_version does not match.',
                    UserWarning
                )
            else:
                warnings.warn(
                    'Dataset xcorr_version does not match. Item skipped.',
                    UserWarning
                )
                continue
        try:
            dsets = dsets.merge(ds, join='outer')
        except xr.MergeError:
            warnings.warn(
                'Dataset could not be merged. Item skipped.',
                RuntimeWarning
            )
            continue

    # fix status dtype change
    dsets['status'] = dsets.status.astype(np.byte)

    # extract valid data
    if extract:
        dsets['cc'] = dsets.cc.where(dsets.status == 1)

    # update some global attrs
    del dsets.attrs['sha256_hash']

    strt0 = pd.to_datetime(dsets.time.values[0]).strftime('%Y.%j')
    strt1 = pd.to_datetime(dsets.time.values[-1]).strftime('%Y.%j')
    dsets.attrs['title'] = (
        (dsets.attrs['title']).split(' - ')[0] +
        ' - ' +
        strt0 +
        ' to {}'.format(strt1) if strt0 != strt1 else ''
    ).strip()

    dsets.attrs['history'] = 'Merged @ {}'.format(pd.to_datetime('now'))

    return dsets


def process(
    x: xr.Dataset, client: Client, inventory: obspy.Inventory = None,
    retry_missing: bool = False, test_run: bool = False,  **kwargs
):
    r"""Process the xcorr N-D labeled data array.

    Parameters
    ----------
    x: :class:`xarray.Dataset`
        The data array to process.

    client : :class:`xcorr.Client`
        The initiated client to the local and remote data archives.

    inventory : :class:`obspy.Inventory`
        The inventory object with instrument responses.

    retry_missing : `bool`, optional
        If `True`, ``x.status`` with flag "-1" are reprocessed, otherwise only
        flag "0". Defaults is `False`.

    test_run : `bool`, optional
        If `True` the function is aborted after the first iteration.
        Default is `False`.

    kwargs :
        Arguments passed to :meth:``client.get_pair_preprocessed_waveforms``.

    """
    x.attrs['history'] += (
        ', Process started @ {}'.format(pd.to_datetime('now'))
    )
    # extract and validate preprocess operations
    if isinstance(x.pair.preprocess, dict):
        o = x.pair.preprocess
        check_operations_hash(o, raise_error=True)
    else:
        o = operations_to_dict(x.pair.preprocess)

    # process each pair per time step
    for p in x.pair:
        for t in x.time:
            print(str(p.values), str(t.values)[:19], end=': ')
            if x.status.loc[{'pair': p, 'time': t}].values != 0:
                if not (
                    retry_missing and
                    x.status.loc[{'pair': p, 'time': t}].values == -1
                ):
                    print(
                        'Has status "{}". Skip.'
                        .format(
                            x.status.loc[{'pair': p, 'time': t}].values
                        )
                    )
                    continue
            print('Waveforms', end='. ')
            st = client.get_pair_preprocessed_waveforms(
                pair=p.values,
                time=t.values,
                preprocess=o,
                duration=t.window_length,
                buffer=t.window_length/10,
                inventory=inventory,
                **kwargs
            )
            if not isinstance(st, obspy.Stream) or len(st) != 2:
                print('Missing data. Set status "-1" and skip.')
                x.status.loc[{'pair': p, 'time': t}] = -1
                if test_run:
                    break
                continue
            x.pair_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(st[0].stats.starttime.datetime) -
                pd.to_datetime(st[1].stats.starttime.datetime)
            )
            x.time_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(st[0].stats.starttime.datetime) +
                pd.to_timedelta(x.time.window_length / 2, unit='s') -
                x.time.loc[{'time': t}].values
            )
            print('Hash', end='. ')
            x.hash.loc[{'pair': p, 'time': t}] = util.hash_Stream(st)
            print('CC', end='. ')
            x.cc.loc[{'pair': p, 'time': t}] = cc.cc(
                x=st[0].data[:x.lag.npts],
                y=st[1].data[:x.lag.npts],
                normalize=x.cc.normalize == 1,
                pad=x.lag.pad == 1,
                unbiased=False,  # apply correction for full dataset!
            )[x.lag.index_min:x.lag.index_max]
            x.status.loc[{'pair': p, 'time': t}] = 1
            print('Done.')
            if test_run:
                break

    # update history
    x.attrs['history'] += (
        ', Process ended @ {}'.format(pd.to_datetime('now'))
    )

    # bias correct?
    if x.cc.bias_correct == 1:
        x = bias_correct(x)
        x.attrs['history'] += (
            ', Bias corrected @ {}'.format(pd.to_datetime('now'))
        )

    # update metadata hash
    x.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        x,
        metadata_only=True
    )


def bias_correct(
    x: xr.Dataset, biased_var: str = 'cc', unbiased_var: str = None,
    weight_var: str = 'w'
):
    r"""Bias correct the xcorr N-D labeled data array.

    Parameters
    ----------
    x: :class:`xarray.Dataset`
        The data array to process.

    biased_var : `str`, optional
        The name of the biased correlation variable. Defaults to 'cc'.

    unbiased_var : `str`, optional
        The name of the biased correlation variable. If `None`,
        `unbiased_var` = `biased_var`. Defaults to `None`.

    weight_var: `str`, optional
        The name of unbiased correlation weight variable. Defaults to 'w'.
    """
    if x[biased_var].unbiased != 0:
        print('No need to bias correct again.')
        return
    unbiased_var = unbiased_var or biased_var

    if weight_var not in x.data_vars:
        x[weight_var] = get_weights(x.lag, name=weight_var)

    # create unbiased_var in dataset
    if biased_var != unbiased_var:
        x[unbiased_var] = x[biased_var].copy()
    x[unbiased_var].data = (
        x[unbiased_var] *
        x[weight_var].astype(x[unbiased_var].dtype)
    )

    # update attributes
    x[unbiased_var].attrs['unbiased'] = np.byte(True)
    x[unbiased_var].attrs['long_name'] = (
        'Unbiased ' + x[unbiased_var].attrs['long_name']
    )
    x[unbiased_var].attrs['standard_name'] = (
        'unbiased_' + x[unbiased_var].attrs['standard_name']
    )

    # update history
    x.attrs['history'] += (
        ', Bias corrected CC @ {}'.format(pd.to_datetime('now'))
    )

    # update metadata hash
    x.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        x,
        metadata_only=True
    )


def get_weights(
    lag: xr.DataArray, name: str = 'w'
):
    r"""Construct the unbiased crosscorrelation weight vector from the lag vector.

    Parameters
    ----------
    lag: :class:`xarray.DataArray`
        The lag coordinate.

    name : `str`, optional
        Weight variable name. Defaults to 'w'.

    Returns
    -------
       w : :class:`DataArray`
           Unbiased crosscorrelation weight vector.

    """
    return xr.DataArray(
        data=cc.weight(
            lag.npts, pad=True
        )[lag.index_min:lag.index_max],
        dims=('lag'),
        coords={'lag': lag},
        name=name,
        attrs={
            'long_name': 'Unbiased CC estimate scale factor',
            'units': '-',
        }
    )


def dependencies_version(as_str: bool = False):
    r"""Returns a `dict` with core dependencies and its version.
    """
    versions = {
        json.__name__: json.__version__,
        np.__name__: np.__version__,
        obspy.__name__: obspy.__version__,
        pd.__name__: pd.__version__,
        scipy.__name__: scipy.__version__,
        xr.__name__: xr.__version__,
    }
    if as_str:
        return ', '.join(['-'.join(item) for item in versions.items()])
    else:
        return versions
