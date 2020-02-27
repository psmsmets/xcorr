#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import xarray as xr
from obspy import read_inventory
import os
import sys
import datetime
import ccf
    
def filename(pair:str,time:pd.datetime):
    return '{pair}.{y:04d}.{d:03d}.nc'.format(pair=pair,y=time.year,d=time.dayofyear)

def cc(starttime:datetime.datetime):
    # local clients
    ccf.clients.set(sds_root='/vardim/home/smets/Hydro')

    # general parameters
    sampling_rate = 50.
    window_length = 86400. # 24h
    window_overlap = 21./24. # 3h shift
    clip_lag = pd.to_timedelta((0,9),unit='h')
    title_prefix = 'Monowai Volcanic Centre'
    poi = {'name': 'MVC', 'latitude': -25.887, 'longitude': -177.188, 'elevation': 0., 'local_depth': 132.}

    # set output destination
    dest = '/ribarsko/data/smets/MVC.CC.RAW'

    # stream preprocess operations (sequential!)
    preprocess = {
        'BHZ': [
            ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
            ('filter', {'type':'highpass','freq':.05}),
            ('detrend', { 'type': 'demean' }),
            ('remove_response', {'output': 'VEL'}),
            ('filter', { 'type': 'highpass', 'freq': 3. }),
            ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20 }),
            ('filter', { 'type': 'lowpass', 'freq': 20. }),
            ('trim', {}),
            ('detrend', { 'type': 'demean' }),
            ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
        ],
        'BHR': [
            ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
            ('filter', {'type':'highpass','freq':.05}),
            ('detrend', { 'type': 'demean' }),
            ('remove_response', {'output': 'VEL'}),
            ('rotate', {'method':'->ZNE'}),
            ('rotate', {'method':'NE->RT', 'back_azimuth':250.39 }),
            ('select', {'channel':'BHR'}),
            ('filter', { 'type': 'highpass', 'freq': 3. }),
            ('interpolate', {'sampling_rate': 50, 'method':'lanczos', 'a':20 }),
            ('filter', { 'type': 'lowpass', 'freq': 20. }),
            ('trim', {}),
            ('detrend', { 'type': 'demean' }),
            ('taper', { 'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
        ],
        'EDH': [
            ('merge', { 'method': 1, 'fill_value': 'interpolate', 'interpolation_samples':0 }),
            ('filter', {'type':'highpass','freq':.05}),
            ('detrend', { 'type': 'demean' }),
            ('remove_sensitivity', {}),
            ('filter', { 'type': 'bandpass', 'freqmin': 3., 'freqmax': 20. }),
            ('decimate', { 'factor': 5 }),
            ('trim', {}),
            ('detrend', { 'type': 'demean' }),
            ('taper', {'type': 'cosine', 'max_percentage': 0.05, 'max_length': 30.}),
        ],
    }

    inv = read_inventory('/ribarsko/data/smets/Monowai.xml')
    pairs = [
        'IM.H03S1..EDH-IU.RAR.10.BHZ',
        'IM.H10N1..EDH-IU.RAR.10.BHZ',
        'IM.H03S1..EDH-IU.RAR.10.BHR',
        'IM.H10N1..EDH-IU.RAR.10.BHR',
        'IM.H03S2..EDH-IU.RAR.10.BHZ',
        'IM.H10N2..EDH-IU.RAR.10.BHZ',
        'IM.H03S2..EDH-IU.RAR.10.BHR',
        'IM.H10N2..EDH-IU.RAR.10.BHR',
        'IM.H03S3..EDH-IU.RAR.10.BHZ',
        'IM.H10N3..EDH-IU.RAR.10.BHZ',
        'IM.H03S3..EDH-IU.RAR.10.BHR',
        'IM.H10N3..EDH-IU.RAR.10.BHR',
    ]
    starttime = starttime + pd.offsets.DateOffset(days=0,normalize=True)
    times = pd.date_range(starttime, starttime + pd.offsets.DateOffset(months=1), freq='1D')

    print(times)
    return

    # cross-correlate all pairs and periods
    warnings.filterwarnings('ignore') # no warnings of duplicate inventory items

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
                    endtime = time + pd.offsets.DateOffset(days=1), 
                    preprocess = preprocess, 
                    sampling_rate = sampling_rate, 
                    window_length = window_length, 
                    window_overlap = window_overlap, 
                    title_prefix = title_prefix,
                    clip_lag = clip_lag,
                    unbiased = False,
                    inventory = inv,
                    stationary_poi = poi,
                )
            try:
                ccf.cc_dataset(
                    ds,
                    inventory = inv.select(
                        starttime = UTCDateTime(time),
                        endtime = UTCDateTime(time + pd.offsets.DateOffset(days=1))
                    ),
                    retry_missing = True,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print('An error occurred. Save and continue next timestep.')
                print('Error:')
                print(e)
            ccf.write_dataset(ds,ncfile)

def usage():
    """
    Print the usage.
    """
    print("MVC_CC -t<starttime> [-h]")
    
def main():
    """
    Main caller function.
    """
    year = None
    month = None
    try:
        opts, params['args'] = getopt.getopt(sys.argv[1:],"ht:v",["help","time=","version"])
    except getopt.GetoptError as e:
        print(str(e))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in  ("-h", "--help"):
            usage()
            sys.exit()
        if opt in  ("-v", "--version"):
            print(ccf.__version__)
            sys.exit()
        elif opt in ("-t", "--starttime"):
            time = arg
        else:
            assert False, "unhandled option"

    assert time, "You should specify a valid starttime!" 
    cc(starttime = pd.to_datetime(time) )    

if __name__ === "__main__":
    main()
