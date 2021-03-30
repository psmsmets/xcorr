"""
Process
=======

xcorr correlation processing.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read_inventory
import os
import xcorr


###############################################################################
# Parameters
# ----------

# client object.
client = xcorr.Client(sds_root='../../data/WaveformArchive')

# general parameters
sampling_rate = 50.       # Hz
window_length = 86400.    # seconds
window_overlap = 21./24.  # 3h shift, decimal
clip_lag = (0., 9*3600.)  # tuple in seconds
poi = {
    'name': 'MVC',
    'latitude': -25.887,
    'longitude': -177.188,
    'elevation': 0.,
    'local_depth': 132.
}
attrs = {
    'title': 'Monowai Volcanic Centre',
    'institution': ('Delft University of Technology, '
                    'Department of Geoscience and Engineering'),
    'author': 'Pieter Smets - P.S.M.Smets@tudelft.nl',
    'source': 'CTBTO/IMS hydroacoustic data and IRIS/USGS seismic data',
}

# pairs and time range
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H10N1..EDH-IU.RAR.10.BHT',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHT',
]
times = pd.date_range('2015-01-15', '2015-01-17', freq='1D')

# inventory, filtered for pairs and range
inv = xcorr.util.receiver.get_pair_inventory(
    pairs, read_inventory('../../data/Monowai.xml'), times
)

# Stream operations
preprocess = {
    'BHZ': [
        ('merge', {
            'method': 1,
            'fill_value': 'interpolate',
            'interpolation_samples': 0,
        }),
        ('filter', {'type': 'highpass', 'freq': .05}),
        ('detrend', {'type': 'demean'}),
        ('remove_response', {'output': 'VEL'}),
        ('interpolate', {
            'sampling_rate': 50,
            'method': 'lanczos',
            'a': 20,
        }),
        ('filter', {'type': 'lowpass', 'freq': 20.}),
        ('trim', {}),
        ('detrend', {'type': 'demean'}),
        ('taper', {
            'type': 'cosine',
            'max_percentage': 0.05,
            'max_length': 30.,
        }),
    ],
    'BHR': [
        ('merge', {
            'method': 1,
            'fill_value': 'interpolate',
            'interpolation_samples': 0,
        }),
        ('filter', {'type': 'highpass', 'freq': .05}),
        ('detrend', {'type': 'demean'}),
        ('remove_response', {'output': 'VEL'}),
        ('rotate', {'method': '->ZNE'}),
        ('rotate', {'method': 'NE->RT', 'back_azimuth': 250.39}),
        ('select', {'channel': 'BHR'}),
        ('interpolate', {
            'sampling_rate': 50,
            'method': 'lanczos',
            'a': 20,
        }),
        ('filter', {'type': 'lowpass', 'freq': 20.}),
        ('trim', {}),
        ('detrend', {'type': 'demean'}),
        ('taper', {
            'type': 'cosine',
            'max_percentage': 0.05,
            'max_length': 30.,
        }),
    ],
    'BHT': [
        ('merge', {
            'method': 1,
            'fill_value': 'interpolate',
            'interpolation_samples': 0,
        }),
        ('filter', {'type': 'highpass', 'freq': .05}),
        ('detrend', {'type': 'demean'}),
        ('remove_response', {'output': 'VEL'}),
        ('rotate', {'method': '->ZNE'}),
        ('rotate', {'method': 'NE->RT', 'back_azimuth': 250.39}),
        ('select', {'channel': 'BHT'}),
        ('interpolate', {
            'sampling_rate': 50,
            'method': 'lanczos',
            'a': 20,
        }),
        ('filter', {'type': 'lowpass', 'freq': 20.}),
        ('trim', {}),
        ('detrend', {'type': 'demean'}),
        ('taper', {
            'type': 'cosine',
            'max_percentage': 0.05,
            'max_length': 30.,
        }),
    ],
    'EDH': [
        ('merge', {
            'method': 1,
            'fill_value': 'interpolate',
            'interpolation_samples': 0,
        }),
        ('filter', {'type': 'highpass', 'freq': .05}),
        ('detrend', {'type': 'demean'}),
        ('remove_response', {}),
        ('decimate', {'factor': 5}),
        ('trim', {}),
        ('detrend', {'type': 'demean'}),
        ('taper', {
            'type': 'cosine',
            'max_percentage': 0.05,
            'max_length': 30.,
        }),
    ],
}

###############################################################################
# One day of data
# ---------------

# select first pair and day
pair = pairs[0]
time = times[0]

# initialize xcorr dataset
ds = xcorr.init(
    pair=pair,
    starttime=time,
    endtime=time + pd.offsets.DateOffset(1),
    attrs=attrs,
    preprocess=preprocess,
    sampling_rate=sampling_rate,
    window_length=window_length,
    window_overlap=window_overlap,
    clip_lag=clip_lag,
    unbiased_cc=False,
    inventory=inv,
    stationary_poi=poi,
)

# process dataset
xcorr.process(ds, inventory=inv, client=client, test_run=True)

# plot result
ds.cc.loc[{'time': ds.time[0]}].plot.line(x='lag', aspect=2.5, size=4)
plt.tight_layout()
plt.show()

# save as netCDF
xcorr.write(ds, 'test.nc', verb=1)

# read from netCDF
ds1 = xcorr.read('test.nc', verb=1)

###############################################################################
# Whole period
# ------------

root = '../../data/cc'

for pair in pairs:

    print('---------------------------')
    print(pair)
    print('---------------------------')

    for time in times:

        nc = xcorr.io.ncfile(pair, time, root)
        ds = False

        if os.path.isfile(nc):

            ds = xcorr.read(nc, fast=True)

            if ds and np.all(ds.status.values == 1):

                ds.close()
                continue

        if not ds:

            ds = xcorr.init(
                pair=pair,
                starttime=time,
                endtime=time + pd.offsets.DateOffset(1),
                attrs=attrs,
                preprocess=preprocess,
                sampling_rate=sampling_rate,
                window_length=window_length,
                window_overlap=window_overlap,
                clip_lag=clip_lag,
                unbiased_cc=False,
                inventory=inv,
                stationary_poi=poi,
            )

        try:

            xcorr.process(ds, inventory=inv, client=client, retry_missing=True)

        except (KeyboardInterrupt, SystemExit):

            raise

        except Exception as e:

            print('An error occurred. Save and continue next step.')
            print('Error:')
            print(e)

        if ds and np.any(ds.status.values != 0):

            xcorr.write(ds, nc)
