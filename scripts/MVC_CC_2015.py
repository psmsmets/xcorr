#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import xarray as xr
from obspy import read_inventory
import os
import ccf

# filename function
def filename(pair:str,time:pd.datetime):
    return '{pair}.{y:04d}.{d:03d}.nc'.format(pair=pair,y=time.year,d=time.dayofyear)

# local clients
ccf.clients.set(sds_root='/vardim/home/smets/Hydro')

# general parameters
sampling_rate = 50.
window_length = 86400. # 24h
window_overlap = 21./24. # 3h shift
clip_max_abs_lag = 3600. * 3.5
title_prefix = 'Monowai Volcanic Centre'

# set output destination
dest = '/ribarsko/data/smets/MVC.CC.RAW'

# stream preprocess operations (sequential!)
preprocess = {
    'BHZ': [
        ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
        ('filter', {'type':'highpass','freq':.05}),
        ('detrend', { 'type': 'demean' }),
        ('remove_response', {'output': 'VEL'}),
        ('filter', { 'type': 'highpass', 'freq': 4. }),
        ('trim', {}),
        ('detrend', { 'type': 'demean' }),
        ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20}),
        ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
    ],
    'EDH': [
        ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
        ('detrend', { 'type': 'demean' }),
        ('remove_sensitivity', {}),
        ('filter', { 'type': 'bandpass', 'freqmin': 4., 'freqmax': 10. }),
        ('trim', {}),
        ('detrend', { 'type': 'demean' }),
        ('decimate', { 'factor': 5 }),
        ('taper', {'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
    ],
}

inv = read_inventory('/vardim/home/smets/Research/hydro/Monowai/Monowai_EDH_BHZ.xml')
pairs = [
    'IM.H10N1..EDH-IU.RAR.00.BHZ',
    'IM.H10N2..EDH-IU.RAR.00.BHZ',
    'IM.H10N3..EDH-IU.RAR.00.BHZ',
    'IM.H03S1..EDH-IU.RAR.00.BHZ',
    'IM.H03S2..EDH-IU.RAR.00.BHZ',
    'IM.H03S3..EDH-IU.RAR.00.BHZ',
]
times = pd.date_range('2015', '2016', freq='1D')

# cross-correlate all pairs and periods
for pair in pairs:
    print('---------------------------')
    print(pair)
    print('---------------------------')
    for time in times:
        ncfile = os.path.join(dest,pair,filename(pair, time))
        if os.path.isfile(ncfile):
            ds = xr.open_dataset(ncfile)
            if np.all(ds.status.values == 1):
                ds.close()
                continue
        else:
            ds = ccf.init_dataset(
                pair=pair, 
                starttime = time, 
                endtime = time + pd.offsets.DateOffset(1), 
                preprocess = preprocess, 
                sampling_rate = sampling_rate, 
                window_length = window_length, 
                window_overlap = window_overlap, 
                clip_max_abs_lag = clip_max_abs_lag, 
                title_prefix = title_prefix
            )
        try:
            ccf.cc_dataset(ds,inventory=inv,retry_missing=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print('An error occurred. Save and continue next timestep.')
            print('Error:')
            print(e)
        ccf.write_dataset(ds,ncfile)
