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

from obspy import Stream, Inventory
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import json
import os

import ccf


def write_dataset(
    dataset: xr.Dataset, path: str, close: bool = True, **kwargs
):
    """
    Write a dataset to netCDF using a tmp file and replacing the destination.
    """
    print('Write dataset as {}'.format(path), end='. ')
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
    print('To temporary netcdf', end='. ')
    dataset.to_netcdf(path=tmp, mode='w', **kwargs)
    print('Replace', end='. ')
    os.replace(tmp, os.path.join(abspath, file))
    print('Done.')


def open_dataset(
    path: str, extract: bool = True, load_and_close: bool = False,
    debug: bool = False
):
    """
    Open a netCDF dataset with cc while checking the data availability.
    """
    if not os.path.isfile(path):
        return False
    dataset = xr.open_dataset(path)
    if debug:
        print(path, np.sum(dataset.status.values == 1))
    if np.sum(dataset.status.values == 1) == 0:
        dataset.close()
        return False
    if extract:
        dataset['cc'] = dataset.cc.where(dataset.status == 1)
    if load_and_close:
        dataset.load().close()
    return dataset


def init_dataset(
    pair: str, starttime: datetime, endtime: datetime, preprocess: dict,
    sampling_rate: float, attrs: dict,
    window_length: float = 86400., window_overlap: float = 0.875,
    clip_lag=None, unbiased: bool = False, title_prefix: str = '',
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
        'title': (
            title_prefix +
            ' Crosscorrelations - {}'
            .format(starttime.strftime('%B %Y'))
        ).strip(),
        'history': 'Created @ {}'.format(pd.to_datetime('now')),
        'conventions': 'CF-1.7',
        'institution': attrs['institution'],
        'author': attrs['author'],
        'source': attrs['source'],
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
        'comment': attrs['comment'] if 'comment' in attrs else 'n/a',
        'ccf_version': ccf.__version__,
    }

    # pair
    dataset.coords['pair'] = [pair]
    dataset.pair.attrs = {
        'long_name': 'Cross-correlation receiver pair',
        'standard_name': 'receiver_pair',
        'units': '-',
        'preprocess': json.dumps(preprocess)
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
    lag = ccf.cc.lag(npts, delta, pad=True)
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
        nmin = np.argmin(abs(lag - clip_lag[0] / ccf.utils.one_second))
        nmax = np.argmin(abs(lag - clip_lag[1] / ccf.utils.one_second))
    else:
        nmin = 0
        nmax = 2*npts-1
    dataset.coords['lag'] = pd.to_timedelta(lag[nmin:nmax], unit='s')
    dataset.lag.attrs = {
        'long_name': 'Lag time',
        'standard_name': 'lag_time',
        'sampling_rate': sampling_rate,
        'delta': delta,
        'npts': npts,
        'pad': np.int8(1),
        'clip': np.int8(clip_lag is not None),
        'clip_lag': (
            clip_lag.values / ccf.utils.one_second
            if clip_lag is not None else None
        ),
        'index_min': nmin,
        'index_max': nmax,
    }

    # pair distance
    dataset['distance'] = (
        ('pair'),
        np.ones((1), dtype=np.float64) * ccf.utils.get_pair_distance(
            pair=pair,
            inventory=inventory,
            poi=stationary_poi,
            km=True,
        ),
        {
            'long_name': 'receiver pair distance',
            'standard_name': 'receiver_pair_distance',
            'units': 'km',
            'relative_to': json.dumps(stationary_poi),
        },
    )

    # status
    dataset['status'] = (
        ('pair', 'time'),
        np.zeros((1, len(dataset.time)), dtype=np.int8),
        {
            'long_name': 'processing status',
            'standard_name': 'processing_status',
            'units': '-',
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
            'long_name': 'Cross-Correlation Estimate',
            'standard_name': 'cross_correlation_estimate',
            'units': '-',
            'add_offset': dtype(0),
            'scale_factor': dtype(1),
            'valid_range': dtype([-1., 1.]),
            'normalize': np.int8(1),
            'bias_correct': np.int8(unbiased),
            'unbiased': np.int8(0),
        },
        encoding
    )

    if unbiased:
        dataset['w'] = ccf.get_dataset_weights(dataset, dtype=dtype)

    return dataset


def cc_dataset(
    dataset: xr.Dataset, inventory: Inventory = None, test: bool = False,
    retry_missing: bool = False, **kwargs
):
    """
    Process a dataset.
    """
    ccf.clients.check(raise_error=True)
    for p in dataset.pair:
        o = json.loads(p.preprocess)
        for t in dataset.time:
            print(str(p.values), str(t.values)[:19], end='. ')
            if dataset.status.loc[{'pair': p, 'time': t}].values != 0:
                if not (
                    retry_missing and
                    dataset.status.loc[{'pair': p, 'time': t}].values == -1
                ):
                    print(
                        'Has status = {}. Skip.'
                        .format(
                            dataset.status.loc[{'pair': p, 'time': t}].values
                        )
                    )
                    continue
            print('Waveforms', end='. ')
            stream = ccf.clients.get_preprocessed_pair_stream(
                pair=p.values,
                time=t.values,
                operations=o,
                duration=t.window_length,
                buffer=t.window_length / 4,
                inventory=inventory,
                operations_from_json=False,
                **kwargs
            )
            if not isinstance(stream, Stream) or len(stream) != 2:
                print('Missing data. Set status = -1 and skip.')
                dataset.status.loc[{'pair': p, 'time': t}] = -1
                if test:
                    break
                continue
            dataset.pair_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(stream[0].stats.starttime.datetime) -
                pd.to_datetime(stream[1].stats.starttime.datetime)
            )
            dataset.time_offset.loc[{'pair': p, 'time': t}] = (
                pd.to_datetime(stream[0].stats.starttime.datetime) +
                pd.to_timedelta(dataset.time.window_length / 2, unit='s') -
                dataset.time.loc[{'time': t}].values
            )
            print('CC', end='. ')
            # Todo:
            # - store noise window outside of the valid domain when clipping!
            dataset.cc.loc[{'pair': p, 'time': t}] = ccf.cc.cc(
                x=stream[0].data[:dataset.lag.npts],
                y=stream[1].data[:dataset.lag.npts],
                normalize=dataset.cc.normalize == 1,
                pad=dataset.lag.pad == 1,
                unbiased=False,  # apply correction for full dataset!
            )[dataset.lag.index_min:dataset.lag.index_max]
            dataset.status.loc[{'pair': p, 'time': t}] = 1
            print('Done.')
            if test:
                break
    if dataset.cc.bias_correct == 1:
        dataset = ccf.bias_correct_dataset(dataset)


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
    dataset[unbiased_var].attrs['unbiased'] = np.int8(True)
    dataset[unbiased_var].attrs['long_name'] = (
        'Unbiased ' + dataset[unbiased_var].attrs['long_name']
    )
    dataset[unbiased_var].attrs['standard_name'] = (
        'unbiased_' + dataset[unbiased_var].attrs['standard_name']
    )


def get_dataset_weights(
    dataset: xr.Dataset, name: str = 'w'
):
    return xr.DataArray(
        data=ccf.cc.weight(
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
