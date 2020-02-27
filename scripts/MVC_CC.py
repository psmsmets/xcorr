#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import xarray as xr
from obspy import UTCDateTime, read_inventory
from pathlib import Path
import os
import sys
import getopt
import datetime
import ccf
    
def filename(pair:str,time:pd.datetime):
    return '{pair}.{y:04d}.{d:03d}.nc'.format(pair=pair,y=time.year,d=time.dayofyear)

def cc(start:datetime.datetime, debug:bool = None, hidebug:bool = None, test:bool = None):
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
    dest = '/ribarsko/data/smets/hydro/MVC.CC.RAW'
    p = Path(dest)
    p.touch()

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

    inv = read_inventory('/ribarsko/data/smets/hydro/Monowai.xml')
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
    if debug or test:
        print('pairs = ', pairs)
    start = start + pd.offsets.DateOffset(days=0,normalize=True)
    times = pd.date_range(
        start = start, 
        end = start + pd.offsets.DateOffset(months=1), 
        freq='1D',
        closed = 'left'
    )
    if debug or test:
        print('times = ', times)

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
                    verbose = debug or False,
                    debug = hidebug or False,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print('An error occurred. Save and continue next timestep.')
                print('Error:')
                print(e)
            ccf.write_dataset(ds,ncfile)
            if test:
                sys.exit(0)

def usage():
    """
    Print the usage.
    """
    print("{} -s<year-month> [-h -v]".format(sys.argv[0]))
    print("Arguments:")
    print("-s,--start=   Set the start year and month (fmt \"yyyy-mm\").")
    print("-v,--version  Print the ccf library version.")
    print("-h,--help     Show this help.")
    print("   --debug    General debugging messages.")
    print("   --hidebug  Far more detailed debugging messages.")
    print("   --test     Quit the main loop after one step.")
    sys.exit()
 
def main():
    """
    Main caller function.
    """
    time = None
    debug = False
    hidebug = False
    test = False
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hs:v",
            ["help","start=","version","debug","hidebug","test"]
        )
    except getopt.GetoptError as e:
        print(str(e))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in  ("-h", "--help"):
            usage()
        if opt in  ("-v", "--version"):
            print(ccf.__version__)
            sys.exit()
        elif opt in ("-s", "--start"):
            time = arg
        elif opt in ("--debug"):
            debug = True
        elif opt in ("--hidebug"):
            hidebug = True
        elif opt in ( "--test"):
            test = True
        else:
            print( "unhandled option \"{}\"".format(opt) )
            usage()

    assert time, "You should specify a valid start time -s<start>!" 
    cc( start = pd.to_datetime(time), test = test, debug = debug, hidebug = hidebug )    

if __name__ == "__main__":
    main()
