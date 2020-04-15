r"""

:mod:`core.core` -- Core
========================

Main functions of xcorr to init, process. bias_correct,
read, write and merge N-D labeled arrays of data.

"""

# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import obspy
import scipy
import json
import warnings
import os


# Relative imports
from ..version import version as __version__
from ..clients import Client
from .. import cc as correlate
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
    pair: str, starttime: pd.Timestamp, endtime: pd.Timestamp,
    preprocess: dict, attrs: dict, sampling_rate: float,
    window_length: float = 86400., window_overlap: float = 0.875,
    clip_lag=None, unbiased_cc: bool = False, closed: str = 'left',
    dtype: np.dtype = np.float32, inventory: obspy.Inventory = None,
    stationary_poi: dict = None, hash_waveforms: bool = False,
):
    r"""Initiate an xcorr N-D labeled data array.

    Parameters:
    -----------
    pair : `str`
        Receiver couple separated by a '-'. Each receiver is specified by its
        SEED-id string: '{network}.{station}.{location}.{channel}'.

    starttime : `pd.Timestamp`
        Start time for generating crosscorrelation time windows.

    endtime : `pd.Timestamp`
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

    clip_lag : `float` or `tuple`, optional
        Clip the crosscorrelation lag time. Defaults to `None`. If ``clip_lag``
        is a single value ``lag`` is clipped symmetrically around zero with
        radius ``clip_lag`` (`float` seconds).
        Provide a `tuple` with two elements, (lag_min, lag_max), both of type
        `float` to clip a specific lag range of interest.

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

    hash_waveforms : `bool`, optional
        Create a sha256 hash of the preprocessed waveforms used for each
        correlation step. Caution: hashing can take some time (~10s per step).

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
    lag = correlate.lag(npts, delta, pad=True)
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
        'long_name': 'Lag time',
        'standard_name': 'lag_time',
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

    # hash waveforms
    if hash_waveforms:
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
                    "'channel', 'starttime', 'endtime', 'sampling_rate', "
                    "'delta', 'npts'], sorted and dumped to json with 4 "
                    "character space indentation and separators ',' and ':', "
                    "followed by the hash of each sample byte representation."
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
        dataset, metadata_only=True, debug=False
    )

    return dataset


def read(
    path: str, extract: bool = False, load_and_close: bool = False,
    fast: bool = False, quick_and_dirty: bool = False,
    metadata_hash: str = None, verb: int = 0
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

    metadata_hash : `str`, optional
        Provide a template metadata sha256 hash to filter data arrays.
        A mismatch in ``dataset.sha256_hash_metadata`` will return `None`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns:
    --------
    dataset : :class:`xarray.Dataset`
        The initiated `xcorr` N-D labeled data array.

    """
    # open dataset if it exists
    if not os.path.isfile(path):
        return None
    dataset = xr.open_dataset(path)

    # check existance of main attributes
    if (
        'xcorr_version' not in dataset.attrs or
        'sha256_hash_metadata' not in dataset.attrs or
        'sha256_hash' not in dataset.attrs
    ):
        return None

    # Verify metadata_hash input
    if metadata_hash:
        if not isinstance(metadata_hash, str):
            raise TypeError('``metadata_hash`` should be a string.')
        if not len(metadata_hash) == 64:
            raise ValueError('``metadata_hash`` should be of length 64.')

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)

    # calculate some hashes
    if not quick_and_dirty:
        sha256_hash_metadata = util.hasher.hash_Dataset(
            dataset, metadata_only=True, debug=False
        )
        if sha256_hash_metadata != dataset.sha256_hash_metadata:
            warnings.warn(
                'Dataset metadata sha256 hash is inconsistent.',
                UserWarning
            )
            if verb > 1:
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
            dataset, metadata_only=False, debug=False
        )
        if sha256_hash != dataset.sha256_hash:
            warnings.warn(
                'Dataset sha256 hash is inconsistent.',
                UserWarning
            )
            if verb > 1:
                print('sha256 hash in ncfile :', dataset.sha256_hash)
                print('sha256 hash computed  :', sha256_hash)

    # verbose status
    if verb > 0:
        print('{p} #(status==1): {n} of {m}'.format(
            p=path,
            n=np.sum(dataset.status.data == 1),
            m=dataset.time.size,
        ))

    # compare metadata_hash with template
    if metadata_hash:
        if dataset.sha256_hash_metadata is not metadata_hash:
            dataset.close()
            return None

    # missing/unprocessed to NaN
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    # push to memory and close
    if load_and_close:
        dataset.load().close()

    return dataset


def write(
    dataset: xr.Dataset, path: str, close: bool = True,
    force_write: bool = False, verb: int = 1
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

    close : `bool`, optional
        Close the data `True` (default).

    force_write : `bool`, optional
        Always write file if `True` even if its empty. Default is `False`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    """
    if (
        np.sum(dataset.status.data == 1) == 0 or
        (np.sum(dataset.status.values == -1) == 0 and force_write)
    ):
        warnings.warn(
            'Dataset contains no data. No need to save it.',
            UserWarning
        )
        return

    # verify metadata hash
    sha256_hash_metadata = (
        util.hasher.hash_Dataset(dataset, metadata_only=True, debug=False)
    )
    if sha256_hash_metadata != dataset.sha256_hash_metadata:
        warnings.warn(
            'Dataset metadata sha256 hash is updated.',
            UserWarning
        )
        dataset.attrs['sha256_hash_metadata'] = sha256_hash_metadata

    if verb:
        print('Write dataset as "{}"'.format(path), end=': ')

    # folders and tmp name
    abspath, file = os.path.split(os.path.abspath(path))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    tmp = os.path.join(
        abspath,
        '{f}.{t}'.format(f=file, t=int(pd.to_datetime('now').timestamp()*1e3))
    )

    # close dataset 
    if close:
        if verb:
            print('Close', end='. ')
        dataset.close()

    # calculate dataset hash
    if verb:
        print('Hash', end='. ')
    dataset.attrs['sha256_hash'] = (
        util.hasher.hash_Dataset(dataset, metadata_only=False, debug=False)
    )

    # convert preprocess operations
    if verb > 1:
        print('Operations to json.', end='. ')
    preprocess_operations_to_json(dataset.pair)

    # write to temporary file
    if verb:
        print('To temporary netcdf', end='. ')
    dataset.to_netcdf(path=tmp, mode='w', format='NETCDF4')

    # Replace file
    if verb:
        print('Replace', end='. ')
    os.replace(tmp, os.path.join(abspath, file))

    # convert preprocess operations
    if verb > 1:
        print('Operations to dict.', end='. ')
    preprocess_operations_to_dict(dataset.pair)

    if verb:
        print('Done.')


def merge(
    datasets: list, extract: bool = True, strict: bool = False,
    verb: int = 0, **kwargs
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

    strict : `bool`, optional
        If `True`, do not merge data arrays with different `xcorr` versions.
        Defaults to `False`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    **kwargs :
        Additional parameters provided to :func:`read`.

    Returns:
    --------
    datasets : :class:`xarray.Dataset`
        The merged `xcorr` N-D labeled data array.

    """

    # Todo: make a dask lazy load with open_mfdataset (less memory!)

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
        if verb > 1:
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
                UserWarning
            )
            continue

        if dsets.sha256_hash_metadata != ds.sha256_hash_metadata:
            warnings.warn(
                'Dataset metadata hash does not match. Item skipped.',
                UserWarning
            )
            continue

        if dsets.xcorr_version != ds.xcorr_version and strict:
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
    dataset: xr.Dataset, client: Client, inventory: obspy.Inventory = None,
    retry_missing: bool = False, test_run: bool = False,
    hash_waveforms: bool = True, metadata_hash: str = None,
    verb: int = 1, **kwargs
):
    r"""Process the xcorr N-D labeled data array.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
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

    hash_waveforms : `bool`, optional
        Compute the sha256 hash of the preprocessed waveforms used for each
        correlation step if `True` (default), and ``x`` contains the variable
        ``hash``. Caution: hashing can take some time (~10s per step).

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    **kwargs :
        Arguments passed to :meth:``client.get_pair_preprocessed_waveforms``.

    """
    dataset.attrs['history'] += (
        ', Process started @ {}'.format(pd.to_datetime('now'))
    )
    # extract and validate preprocess operations
    if isinstance(dataset.pair.preprocess, dict):
        o = dataset.pair.preprocess
        check_operations_hash(o, raise_error=True)
    else:
        o = operations_to_dict(dataset.pair.preprocess)

    # hash?
    hash_waveforms = hash_waveforms and 'hash' in dataset.variables

    # process each pair per time step
    for p in dataset.pair:

        for t in dataset.time:

            # set location
            pt = {'pair': p, 'time': t}

            if verb:
                print(str(p.values), str(t.values)[:19], end=': ')

            # skip processed
            if dataset.status.loc[pt].values != 0:
                if not (
                    retry_missing and
                    dataset.status.loc[pt].values == -1
                ):
                    if verb:
                        print('Has status "{}". Skip.'.format(
                              dataset.status.loc[pt].values))
                    continue

            # waveforms
            if verb:
                print('Waveforms', end='. ')
            st = client.get_pair_preprocessed_waveforms(
                pair=p.values,
                time=t.values,
                preprocess=o,
                duration=t.window_length,
                buffer=t.window_length/20,  # 5%
                inventory=inventory,
                verb=verb-1,
                **kwargs
            )
            if not isinstance(st, obspy.Stream) or len(st) != 2:
                print('Missing data. Set status "-1" and skip.')
                dataset.status.loc[pt] = -1
                if test_run:
                    break
                continue

            # track timing offsets
            dataset.pair_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) -
                pd.to_datetime(st[1].stats.starttime.datetime)
            )
            dataset.time_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) +
                pd.to_timedelta(dataset.time.window_length / 2, unit='s') -
                dataset.time.loc[{'time': t}].values
            )

            # hash
            if hash_waveforms:
                if verb:
                    print('Hash', end='. ')
                dataset.hash.loc[pt] = util.hash_Stream(st)

            # cc
            if verb:
                print('CC', end='. ')
            dataset.cc.loc[pt] = correlate.cc(
                x=st[0].data[:dataset.lag.attrs['npts']],
                y=st[1].data[:dataset.lag.attrs['npts']],
                normalize=dataset.cc.attrs['normalize'] == 1,
                pad=True,
                unbiased=False,  # apply correction for full dataset!
                dtype=dataset.cc.dtype,
            )[dataset.lag.attrs['index_min']:dataset.lag.attrs['index_max']]

            # Status
            dataset.status.loc[pt] = 1

            # Finish
            if verb:
                print('Done.')
            if test_run:
                break

    # update history
    dataset.attrs['history'] += (
        ', Process ended @ {}'.format(pd.to_datetime('now'))
    )

    # bias correct?
    if dataset.cc.attrs['bias_correct'] == 1:
        dataset = bias_correct(dataset)

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )


def bias_correct(
    dataset: xr.Dataset, biased_var: str = 'cc', unbiased_var: str = None,
    weight_var: str = 'w'
):
    r"""Bias correct the xcorr N-D labeled data array.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        The data array to process.

    biased_var : `str`, optional
        The name of the biased correlation variable. Defaults to 'cc'.

    unbiased_var : `str`, optional
        The name of the biased correlation variable. If `None`,
        `unbiased_var` = `biased_var`. Defaults to `None`.

    weight_var: `str`, optional
        The name of unbiased correlation weight variable. Defaults to 'w'.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    """
    if dataset[biased_var].unbiased != 0:
        print('No need to bias correct again.')
        return
    unbiased_var = unbiased_var or biased_var

    if weight_var not in dataset.data_vars:
        dataset[weight_var] = get_weights(dataset.lag, name=weight_var)

    # create unbiased_var in dataset
    if biased_var != unbiased_var:
        dataset[unbiased_var] = dataset[biased_var].copy()
    dataset[unbiased_var].data = (
        dataset[unbiased_var] *
        dataset[weight_var].astype(dataset[unbiased_var].dtype)
    )

    # update attributes
    dataset[unbiased_var].attrs['unbiased'] = np.byte(True)
    dataset[unbiased_var].attrs['long_name'] = (
        'Unbiased ' + dataset[unbiased_var].attrs['long_name']
    )
    dataset[unbiased_var].attrs['standard_name'] = (
        'unbiased_' + dataset[unbiased_var].attrs['standard_name']
    )

    # update history
    dataset.attrs['history'] += (
        ', Bias corrected CC @ {}'.format(pd.to_datetime('now'))
    )

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
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
        data=correlate.weight(
            lag.attrs['npts'], pad=True
        )[lag.attrs['index_min']:lag.attrs['index_max']],
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
