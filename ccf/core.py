#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy import UTCDateTime, Trace, Stream, Inventory
import numpy as np
import xarray as xr
import pandas as pd
import os

import ccf

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

def write_dataset(dataset:xr.Dataset, path:str, close:bool = True):
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
    dataset.to_netcdf(path=tmp,mode='w')
    print('Replace', end='. ')
    os.replace(tmp,os.path.join(abspath,file))
    print('Done.')
    
def open_dataset(path:str, extract:bool = True, close:bool = False, debug:bool = False):
    """
    Open a netCDF dataset with cc while checking the data availability. 
    """
    if not os.path.isfile(path):
        return None
    ds = xr.open_dataset(path)
    if debug:
        print(np.sum(ds.status.values == 1))
    if np.sum(ds.status.values == 1) == 0:
        tmp.close()
        return None
    
    if extract:
        ds['cc'] = ds.cc.where(ds.status == 1, drop=True )
        ds = ds.drop_vars('status')
        
    if close:
        ds.close()
        
    return ds
    
def init_dataset(
    pair:str, starttime:pd.datetime, endtime:pd.datetime, preprocess:dict, sampling_rate = 50., window_length = 86400.,
    window_overlap = 0.875, clip_max_abs_lag = None, title_prefix = '', closed:str = 'left'
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
        references = 'n/a',
        comment = 'n/a',
    )
    
    ds['stats'] = np.int8(1)
    ds['stats'].attrs = dict(
        window_length = window_length,
        window_overlap = window_overlap,
        sampling_rate = sampling_rate,
        delta = delta,
        npts = npts,
        pad = 1,
        normalize = 1,
    )
    
    ds['preprocess'] = np.int8(1)
    for channel, operations in preprocess.items():
        ccf.preprocess.add_operations_to_dataset(ds, channel, operations, variable = 'preprocess')
    
    if clip_max_abs_lag is not None:
        lag = ccf.cc.lag( npts, delta, pad = True)
        n = np.argmax(lag>clip_max_abs_lag)-1 
        nmin = 2*(npts-1)-n
        nmax = n+1
        
        ds.coords['lag'] = lag[nmin:nmax]
        ds.coords['lag'].attrs = {
            'long_name': 'Lag time', 'units': 's', 'standard_name': 'lag_time', 'clip': 1, 
            'max_abs_lag': clip_max_abs_lag, 'lag_index_min': nmin, 'lag_index_max': nmax,
        }
    else:
        ds.coords['lag'] = ccf.cc.lag( npts, delta, pad = True)
        ds.coords['lag'].attrs = {
            'long_name': 'Lag time', 'units': 's', 'standard_name': 'lag_time', 'clip': 0,
            'lag_index_min': 0, 'lag_index_max': 2*npts-1,
        }
    
    ds.coords['pair'] = pair
    ds.coords['pair'].attrs = {'long_name': 'Cross-correlation receiver pair', 'units':'-', 'standard_name': 'receiver_pair'} 

    ds.coords['time'] = pd.date_range(
        start = starttime, 
        end = endtime, 
        freq = '{0:.0f}s'.format(window_length*(1-window_overlap)),
        closed = closed,
    )

    ds['status'] = (
        ('time'),
        np.zeros((len(ds.time)),dtype=np.int8),
        {
            'units': '-',
            'long_name': 'processing status',
            'standard_name': 'processing_status',
##
# Returns all as float32 instead of int8 !!
##
#             'add_offset': np.int8(0),
#             'scale_factor': np.int8(1),
#             'valid_range': [np.int8(-1), np.int8(1)],
##
        },
        encoding
    )

    ds['cc'] = (
        ('time','lag'), 
        np.empty((len(ds.time),len(ds.lag)),dtype=np.float32), 
        {
            'units':'-',
            'long_name': 'Cross-Correlation Function',
            'standard_name': 'cross_correlation_function',
            'add_offset': np.float32(0),
            'scale_factor': np.float32(1.),
            'valid_range': [np.float32(-1), np.float32(1)],
        },
        encoding
    )
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
            duration = ds.stats.attrs['window_length'],
            buffer = ds.stats.attrs['window_length']/2,
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
        print('CC', end='. ')
        ds.cc.loc[{'time':t}] = ccf.cc.cc(
            x = stream[0].data[:ds.stats.attrs['npts']],
            y = stream[1].data[:ds.stats.attrs['npts']],
            normalize = ds.stats.attrs['normalize'] == 1,
            pad = ds.stats.attrs['pad'] == 1
        )[ds.lag.attrs['lag_index_min']:ds.lag.attrs['lag_index_max']]
        ds.status.loc[{'time':t}] = 1
        print('Done.')
        if test:
            break
