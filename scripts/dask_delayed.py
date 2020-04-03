#!/usr/bin/python

# Mandatory imports
import warnings
import numpy as np
import pandas as pd
from obspy import read_inventory
import os
import dask
from dask.diagnostics import ProgressBar
import xcorr


##############################################################################
#
# Parameters
#
##############################################################################


# general parameters
sampling_rate = 50.
window_length = 86400.      # 24h
window_overlap = 21./24.    # 3h shift
clip_lag = pd.to_timedelta((0, 6), unit='h')
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


# stream preprocess operations (sequential!)
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
    'EDH': [
        ('merge', {
            'method': 1,
            'fill_value': 'interpolate',
            'interpolation_samples': 0,
        }),
        ('filter', {'type': 'highpass', 'freq': .05}),
        ('detrend', {'type': 'demean'}),
        ('remove_sensitivity', {}),
        # ('remove_response', {}),
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

# pairs, times and inv
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
]
times = pd.date_range(start='2015-01-15', end='2015-01-20', freq='1D')

inventory = '../examples/Monowai.xml'

dest = '../data/results'

# start a client
client = xcorr.Client(
    sds_root='../data/WaveformArchive'
)


##############################################################################
#
# Functions
#
##############################################################################


def filename(pair: str, time: pd.datetime, dest: str = None):
    r"""Construct the filename.
    """
    dest = os.path.join(dest, pair) if dest else pair
    ncfile = '{pair}.{y:04d}.{d:03d}.nc'.format(
        pair=pair,
        y=time.year,
        d=time.dayofyear
    )
    return os.path.join(dest, ncfile)


@dask.delayed
def load(pair, time):
    r"""Main xcorr processing sequence.
    """
    # Try to open first
    ds = xcorr.read(filename(pair, time, dest), fast=True)
    # Exists and anything to update?
    if ds and np.all(ds.status.values == 1):
        ds.close()
        return False
    # Create a new dataset
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
            inventory=inventory,
            stationary_poi=poi,
            hash_waveforms=False,
        )
    return ds


@dask.delayed
def process(ds, inventory, **kwargs):
    r"""Main xcorr processing sequence.
    """
    if not ds:
        return ds
    # Process
    xcorr.process(
        ds,
        inventory=inventory,
        client=client,
        retry_missing=True,
        download=False,
        verb=0,
        hash_waveforms=False,
        **kwargs
    )


@dask.delayed
def save(ds, pair, time):
    r"""
    """
    if ds and np.any(ds.status.values != 0):
        xcorr.write(ds, filename(pair, time, dest), verb=0)
        return True
    else:
        return False


def laizy_processes(pairs, times, status, **kwargs):
    r"""
    """
    results = []
    for pair in pairs:
        receivers = xcorr.util.split_pair(pair, substitute=True, to_dict=False)
        for time in times:
            start = time - pd.offsets.DateOffset(
                seconds=window_length/2, normalize=True
            )
            end = time + pd.offsets.DateOffset(
                seconds=window_length*3/2, normalize=True
            )
            availability = status.loc[{
                'receiver': receivers,
                'time': pd.date_range(start, end, freq='D'),
            }] == 1

        # all receivers required for any day
        if any(availability.sum(dim='receiver') == len(receivers)):
            data = load(pair, time)
            data = process(data, inventory, **kwargs)
            result = save(data, pair, time)
            results.append(result)
    return results


##############################################################################
#
# Main processing
#
##############################################################################

# minimize output to stdout in dask!
warnings.filterwarnings("ignore")

# Read and filter inventory
inventory = xcorr.util.get_pair_inventory(
    pairs, read_inventory(inventory), times
)

# Get waveform availability status
status = client.init_data_availability(
    pairs, times, extend_days=1, substitute=True
)

delayed_status = client.verify_data_availability(
    status, download=True, compute=False
)

# Print some dimensions
print('Dimensions')
print('#pairs     : {}'.format(len(pairs)))
print('#receivers : {}'.format(len(status.receiver)))
print('#times     : {} ({})'.format(len(times), len(status.time)))

# Evaluate data availability (parallel), and try to download missing data
print('Verify data availability')
with ProgressBar():
    verified = delayed_status.compute()

# Print data availability status (total and per receiver)
print('Total availability : {:.2f}%'
      .format(100 * np.sum(status.values == 1) / status.size))
for rec in status.receiver:
    availability = 100 * np.sum(
        status.loc[{'receiver': rec}].values == 1
    ) / status.time.size
    print('{} : {:.2f}%'.format(rec.values, availabilty))

# Evaluate process list (parallel)
print('Compute')
with ProgressBar():
    results = dask.compute(laizy_processes(pairs, times, status))
