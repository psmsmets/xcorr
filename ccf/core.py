#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import UTCDateTime, Trace, Stream, Inventory
import numpy as np
import xarray as xr
import pandas as pd
import os

import ccf

one_second = pd.to_timedelta(1,unit='s')

def toUTCDateTime(datetime):
    """
    Convert various datetime formats to `obspy.UTCDateTime`
    """
    if isinstance(datetime,UTCDateTime):
        return datetime
    elif isinstance(datetime,str) or isinstance(datetime,pd.datetime):
        return UTCDateTime(datetime)
    elif isinstance(datetime,np.datetime64):
        return UTCDateTime(pd.to_datetime(datetime))

def write_dataset(dataset:xr.Dataset, path:str, close:bool = True, **kwargs):
    """
    Write a dataset to netCDF using a tmp file and replacing the destination. 
    """
    print('Write dataset as {}'.format(path), end='. ')
    abspath, file = os.path.split(os.path.abspath(path))
    if not os.path.exists(abspath):
        os.makedirs(abspath)    
    tmp = os.path.join(abspath,'{f}.{t}'.format(f=file,t=int(pd.datetime.now().timestamp()*1e3)))
    if close:
        print('Close', end='. ')
        dataset.close()
    print('To temporary netcdf', end='. ')
    dataset.to_netcdf(path=tmp,mode='w',**kwargs)
    print('Replace', end='. ')
    os.replace(tmp,os.path.join(abspath,file))
    print('Done.')
    
def open_dataset(path:str, extract:bool = True, close:bool = False, debug:bool = False):
    """
    Open a netCDF dataset with cc while checking the data availability. 
    """
    if not os.path.isfile(path):
        return False
    ds = xr.open_dataset(path)
    if debug:
        print(np.sum(ds.status.values == 1))
    if np.sum(ds.status.values == 1) == 0:
        ds.close()
        return False
    
    if extract:
        ds['cc'] = ds.cc.where(ds.status == 1, drop=True )
        ds = ds.drop_vars('status')
        
    if close:
        ds.close()
        
    return ds
    
def init_dataset(
    pair:str, starttime:pd.datetime, endtime:pd.datetime, preprocess:dict, sampling_rate = 50., window_length = 86400.,
    window_overlap = 0.875, clip_lag = None, unbiased:bool = False, title_prefix:str = '', closed:str = 'left',
    dtype = np.float32,
):
    """
    Initiate a dataset. 
    """
    # config
    delta = 1/sampling_rate
    npts = int(window_length*sampling_rate)
    encoding = {'zlib': True, 'complevel': 9}
    
    # start dataset
    ds = xr.Dataset()

    ds.attrs = dict(
        title = (title_prefix + ' Cross-correlations - {}'.format(starttime.strftime('%B %Y'))).strip(),
        history = 'Created @ {}'.format(UTCDateTime()),
        conventions = 'CF-1.7',
        institution = 'Delft University of Technology, Department of Geoscience and Engineering',
        author = 'Pieter Smets - P.S.M.Smets@tudelft.nl',
        source = 'CTBTO/IMS hydroacoustic data and IRIS/USGS seismic data', 
        references = 'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). Random data : analysis and measurement procedures. New York (N.Y.): Wiley-Interscience.',
        comment = 'n/a',
    )
    
    ds['stats'] = np.int8(1)
    ds['stats'].attrs = dict(
        window_length = window_length,
        window_overlap = window_overlap,
        sampling_rate = sampling_rate,
        delta = delta,
        npts = npts,
        pad = np.int8(1),
        normalize = np.int8(1),
        unbiased = np.int8(unbiased),
    )
    
    ds['preprocess'] = np.int8(1)
    for channel, operations in preprocess.items():
        ccf.preprocess.add_operations_to_dataset(ds, channel, operations, variable = 'preprocess')
    
    # lag
    lag_attrs = {
        'long_name': 'Lag time',
        'standard_name': 'lag_time',
    }
    
    if clip_lag is not None:
        if isinstance( clip_lag, pd.Timedelta ):
            clip_lag = pd.to_timedelta((-np.abs(clip_lag),np.abs(clip_lag)))
        elif not ( isinstance(clip_lag,pd.TimedeltaIndex) and len(clip_lag) == 2 ):
            raise TypeError(
                'clip_lag should be of type ~pandas.Timedelta or ' +
                '~pandas.TimedeltaIndex with length 2 specifying start and end lag.'
            )
        lag = ccf.cc.lag( npts, delta, pad = True)
        nmin = np.argmin( abs( lag - clip_lag[0] / one_second ) )
        nmax = np.argmin( abs( lag - clip_lag[1] / one_second ) )
        
        ds.coords['lag'] = pd.to_timedelta( lag[nmin:nmax], unit = 's' )
        ds.coords['lag'].attrs = { **lag_attrs,
            'clip': 1, 
            'clip_lag': clip_lag.values / one_second,
            'index_min': nmin,
            'index_max': nmax,
        }
    else:
        ds.coords['lag'] = pd.to_timedelta( ccf.cc.lag( npts, delta, pad = True), unit = 's' )
        ds.coords['lag'].attrs = { **lag_attrs,
            'clip': 0,
            'index_min': 0,
            'index_max': 2*npts-1,
        }
    
    # pair
    ds.coords['pair'] = pair
    ds.coords['pair'].attrs = {'long_name': 'Cross-correlation receiver pair', 'units':'-', 'standard_name': 'receiver_pair'} 

    ds.coords['time'] = pd.date_range(
        start = starttime, 
        end = endtime, 
        freq = '{0:.0f}s'.format(window_length*(1-window_overlap)),
        closed = closed,
    )

    # status
    ds['status'] = (
        ('time'),
        np.zeros((len(ds.time)), dtype = np.int8),
        {
            'units': '-',
            'long_name': 'processing status',
            'standard_name': 'processing_status',
        },
        encoding
    )
    
    ds['pair_offset'] = (
        ('time'),
        np.zeros((len(ds.time)), dtype = np.timedelta64),
        {
            'long_name': 'receiver pair start sample offset',
            'standard_name': 'receiver_pair_start_sample_offset',
            'description': 'offset = receiver[0].starttime - receiver[1].starttime',
        },
        encoding
    )
    
    ds['time_offset'] = (
        ('time'),
        np.zeros((len(ds.time)), dtype = np.timedelta64),
        {
            'long_name': 'first receiver start sample offset',
            'standard_name': 'first_receiver_start_sample_offset',
            'description': 'offset = receiver[0].starttime - time + window_length/2',
        },
        encoding
    )

    ds['cc'] = (
        ('time','lag'), 
        np.zeros((len(ds.time),len(ds.lag)), dtype = dtype ), 
        {
            'units':'-',
            'long_name': 'Cross-Correlation Estimate',
            'standard_name': 'cross_correlation_estimate',
            'add_offset': dtype(0),
            'scale_factor': dtype(1),
            'valid_range': dtype([-1.,1.]),
            'unbiased' : np.int8(0), # flag to track if biased correction is applied.
        },
        encoding
    )
    
    if unbiased:
        ds['w'] = get_cc_weights_dataset(ds, dtype = dtype )
        
    return ds
    
def cc_dataset( ds:xr.Dataset, inventory:Inventory = None, test:bool = False, retry_missing:bool = False, **kwargs ):
    """
    Process a dataset. 
    """
    ccf.clients.check(raise_error = True)
    p = ds.pair
    for t in ds.time:
        print(str(p.values), str(t.values)[:19], end='. ')
        if ds.status.loc[{'time':t}].values != 0:
            if not (retry_missing and ds.status.loc[{'time':t}].values == -1):
                print('Has status = {}. Skip.'.format(ds.status.loc[{'time':t}].values))
                continue
        print('Waveforms', end='. ')
        stream = ccf.clients.get_preprocessed_pair_stream(
            pair = p.values,
            time = t.values,
            operations = ds.preprocess.attrs,
            duration = ds.stats.window_length,
            buffer = ds.stats.window_length / 4,
            inventory = inventory,
            operations_from_json = True,
            **kwargs
        )
        if not isinstance(stream,Stream) or len(stream)!=2:
            print('Missing data. Set status = -1 and skip.')
            ds.status.loc[{'time':t}] = -1
            if test:
                break
            continue
        ds.pair_offset.loc[{'time':t}] = (
            pd.to_datetime( stream[0].stats.starttime.datetime ) - 
            pd.to_datetime( stream[1].stats.starttime.datetime )
        )
        ds.time_offset.loc[{'time':t}] = (
            pd.to_datetime( stream[0].stats.starttime.datetime ) +
            pd.to_timedelta(ds.stats.window_length / 2, unit = 's') -
            ds.time.loc[{'time':t}].values
        )
        print('CC', end='. ')
        # Todo:
        # - store noise window outside of the valid domain when clipping!
        ds.cc.loc[{'time':t}] = ccf.cc.cc(
            x = stream[0].data[:ds.stats.npts],
            y = stream[1].data[:ds.stats.npts],
            normalize = ds.stats.normalize == 1,
            pad = ds.stats.pad == 1,
            unbiased = False, # apply correction for full dataset!
        )[ds.lag.index_min:ds.lag.index_max]
        ds.status.loc[{'time':t}] = 1
        print('Done.')
        if test:
            break
    if ds.stats.unbiased == 1:
        ds = bias_correct_cc_dataset(ds)

def bias_correct_dataset( ds:xr.Dataset, biased_var:str = 'cc', unbiased_var:str = None, weight_var:str='w' ):
    if ds[biased_var].attrs['unbiased'] != 0:
        print('No need to bias correct again.')
        return
    unbiased_var = unbiased_var or biased_var
    
    if not weight_var in ds.data_vars:
        ds[weight_var] = get_dataset_weights(ds, name = weight_var)

    # create unbiased_var in dataset
    if biased_var != unbiased_var:
        ds[unbiased_var] = ds[biased_var].copy()
    ds[unbiased_var].data = ds[unbiased_var] * ds[weight_var].astype(ds[unbiased_var].dtype)

    # update attributes
    ds[unbiased_var].attrs['unbiased'] = np.int8(True)
    ds[unbiased_var].attrs['long_name'] = 'Unbiased ' + ds[unbiased_var].attrs['long_name']
    ds[unbiased_var].attrs['standard_name'] = 'unbiased_' + ds[unbiased_var].attrs['standard_name']
    
def get_dataset_weights( ds:xr.Dataset, name:str = 'w' ):
    return xr.DataArray (
        data = ccf.cc.weight(ds.stats.npts,pad=True)[ds.lag.index_min:ds.lag.index_max],
        dims = ('lag'),
        coords = {'lag': ds.lag},
        name = name,
        attrs = {
            'long_name': 'Unbiased CC estimate scale factor',
            'units': '-',
        }
    )