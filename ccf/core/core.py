# -*- coding: utf-8 -*-
"""
Core python module of the ccf package to create, open, write and process
an xarray/netCDF4 based ccf file.

.. module:: core

:author:
    Pieter Smets (P.S.M.Smets@tudelft.nl)

:copyright:
    Pieter Smets

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

# Absolute imports
import warnings
from obspy import Stream, Inventory
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import json
import os

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


__all__ = ['write_dataset', 'open_dataset', 'merge_datasets',
           'init_dataset', 'cc_dataset', 'bias_correct_dataset',
           'get_dataset_weights']


def write_dataset(
    dataset: xr.Dataset, path: str, close: bool = True, **kwargs
):
    """
    Write a dataset to netCDF using a tmp file and replacing the destination.
    """
    print("Write dataset as '{}'".format(path), end=': ')
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
        util.hasher.sha256_hash_Dataset(dataset)
    )

    # convert preprocess operations
    preprocess_operations_to_json(dataset.pair)

    print('To temporary netcdf', end='. ')
    dataset.to_netcdf(path=tmp, mode='w', format='NETCDF4', **kwargs)
    print('Replace', end='. ')
    os.replace(tmp, os.path.join(abspath, file))
    print('Done.')

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)


def open_dataset(
    path: str, extract: bool = False, load_and_close: bool = False,
    fast: bool = False, quick_and_dirty: bool = False, debug: bool = False
):
    """
    Open a netCDF dataset with cc while checking the data availability.
    """
    if not os.path.isfile(path):
        return False
    dataset = xr.open_dataset(path)
    if not 'ccf_version' in dataset.attrs:
        return False

    # convert preprocess operations
    preprocess_operations_to_dict(dataset.pair)

    # calculate some hashes
    if not quick_and_dirty:
        sha256_hash_metadata = (
            util.hasher.sha256_hash_Dataset_metadata(dataset)
        )
        if sha256_hash_metadata != dataset.sha256_hash_metadata:
            warnings.warn(
                'Dataset metadata sha256 hash is invalid.',
                UserWarning
            )
    if not (quick_and_dirty or fast):
        sha256_hash = util.hasher.sha256_hash_Dataset(dataset)
        if sha256_hash != dataset.sha256_hash:
            warnings.warn(
                'Dataset sha256 hash is invalid.',
                UserWarning
            )

    if debug:
        print(path, '#(status==1): {} of {}'.format(
            np.sum(dataset.status.values == 1), dataset.time.size
        ))

    if np.sum(dataset.status.values == 1) == 0 and not debug:
        dataset.close()
        return False

    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)

    if load_and_close:
        dataset.load().close()

    return dataset


def merge_datasets(
    datasets: list, extract: bool = True, merge_versions: bool = False,
    debug: bool = False, **kwargs
):
    """
    Merge a list of datasets by specifying either the path as a `str`
    or the :class:`xarray.DataSet` objects.
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
            ds = open_dataset(ds, extract=False, **kwargs)
        elif isinstance(dataset, xr.Dataset):
            preprocess_operations_to_dict(dataset.pair)
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
            print("\n# Open and inspect ccf dataset:\n\n", ds, "\n")
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
        if dsets.ccf_version != ds.ccf_version:
            if merge_versions:
                warnings.warn(
                    'Dataset ccf_version does not match.',
                    UserWarning
                )
            else:
                warnings.warn(
                    'Dataset ccf_version does not match. Item skipped.',
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
            if debug:
                print('Error:', e)
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


def init_dataset(
    pair: str, starttime: datetime, endtime: datetime, preprocess: dict,
    sampling_rate: float, attrs: dict,
    window_length: float = 86400., window_overlap: float = 0.875,
    clip_lag=None, unbiased_cc: bool = False,
    closed: str = 'left', dtype: np.dtype = np.float32,
    inventory: Inventory = None, stationary_poi: dict = None,
):
    """
    Initiate a ccf xarray.Dataset.
    """
    # check metadata
    assert 'institution' in attrs, (
        "attrs['institution'] = 'Institution, department'!"
    )
    assert 'author' in attrs, (
        "attrs['author'] = 'Name - E-mail'!"
    )
    assert 'source' in attrs, (
        "attrs['source'] = 'Data source description'!"
    )
    # config
    delta = 1/sampling_rate
    npts = int(window_length*sampling_rate)
    encoding = {'zlib': True, 'complevel': 9}

    # start dataset
    dataset = xr.Dataset()

    # global attributes
    dataset.attrs = {
        'Conventions': 'CF-1.9',
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
        'ccf_version': __version__,
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
        dataset['w'] = get_dataset_weights(dataset, dtype=dtype)

    # add metadata hash
    dataset.attrs['sha256_hash_metadata'] = (
        util.hasher.sha256_hash_Dataset_metadata(dataset)
    )

    return dataset


def cc_dataset(
    dataset: xr.Dataset, client: Client, inventory: Inventory = None,
    test_run: bool = False, retry_missing: bool = False, **kwargs
):
    """
    Process a dataset.
    """
    dataset.attrs['history'] += (
        ', CC process started @ {}'.format(pd.to_datetime('now'))
    )
    # extract and validate preprocess operations
    if isinstance(dataset.pair.preprocess, dict):
        o = dataset.pair.preprocess
        check_operations_hash(o, raise_error=True)
    else:
        o = operations_to_dict(dataset.pair.preprocess)

    # process each pair per time step 
    for p in dataset.pair:
        for t in dataset.time:
            print(str(p.values), str(t.values)[:19], end=': ')
            if dataset.status.loc[{'pair': p, 'time': t}].values != 0:
                if not (
                    retry_missing and
                    dataset.status.loc[{'pair': p, 'time': t}].values == -1
                ):
                    print(
                        'Has status "{}". Skip.'
                        .format(
                            dataset.status.loc[{'pair': p, 'time': t}].values
                        )
                    )
                    continue
            print('Waveforms', end='. ')
            st = client.get_pair_preprocessed_waveforms(
                pair=p.values,
                time=t.values,
                operations=o,
                duration=t.window_length,
                buffer=t.window_length/10,
                inventory=inventory,
                operations_from_json=False,
                **kwargs
            )
            if not isinstance(st, Stream) or len(st) != 2:
                print('Missing data. Set status "-1" and skip.')
                dataset.status.loc[{'pair': p, 'time': t}] = -1
                if test_run:
                    break
                continue
            dataset.pair_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(st[0].stats.starttime.datetime) -
                pd.to_datetime(st[1].stats.starttime.datetime)
            )
            dataset.time_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(st[0].stats.starttime.datetime) +
                pd.to_timedelta(dataset.time.window_length / 2, unit='s') -
                dataset.time.loc[{'time': t}].values
            )
            print('CC', end='. ')
            dataset.cc.loc[{'pair': p, 'time': t}] = cc.cc(
                x=st[0].data[:dataset.lag.npts],
                y=st[1].data[:dataset.lag.npts],
                normalize=dataset.cc.normalize == 1,
                pad=dataset.lag.pad == 1,
                unbiased=False,  # apply correction for full dataset!
            )[dataset.lag.index_min:dataset.lag.index_max]
            dataset.status.loc[{'pair': p, 'time': t}] = 1
            print('Done.')
            if test_run:
                break

    # update history
    dataset.attrs['history'] += (
        ', CC process ended @ {}'.format(pd.to_datetime('now'))
    )

    # bias correct?
    if dataset.cc.bias_correct == 1:
        dataset = bias_correct_dataset(dataset)
        dataset.attrs['history'] += (
            ', Bias corrected CC @ {}'.format(pd.to_datetime('now'))
        )

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = (
        util.hasher.sha256_hash_Dataset_metadata(dataset)
    )

def bias_correct_dataset(
    dataset: xr.Dataset, biased_var: str = 'cc', unbiased_var: str = None,
    weight_var: str = 'w'
):
    if dataset[biased_var].unbiased != 0:
        print('No need to bias correct again.')
        return
    unbiased_var = unbiased_var or biased_var

    if weight_var not in dataset.data_vars:
        dataset[weight_var] = get_dataset_weights(dataset, name=weight_var)

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
    dataset.attrs['sha256_hash_metadata'] = (
        util.hasher.sha256_hash_Dataset_metadata(dataset)
    )


def get_dataset_weights(
    dataset: xr.Dataset, name: str = 'w'
):
    return xr.DataArray(
        data=cc.weight(
            dataset.lag.npts, pad=True
        )[dataset.lag.index_min:dataset.lag.index_max],
        dims=('lag'),
        coords={'lag': dataset.lag},
        name=name,
        attrs={
            'long_name': 'Unbiased CC estimate scale factor',
            'units': '-',
        }
    )
