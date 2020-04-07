#!/usr/bin/python

# Mandatory imports
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from obspy import read_inventory
import dask
from dask.diagnostics import ProgressBar
import xcorr


##############################################################################
#
# Config
#
##############################################################################


# -----------------------------------------------------------------------------
# xcorr_init_args : `dict`
#     A dictionary with input argument to :class:`xcorr.init`
# -----------------------------------------------------------------------------
xcorr_init_args = {
    'sampling_rate': 50.,
    'window_length': 86400.,      # 24h
    'window_overlap': 21./24.,    # 3h shift
    'clip_lag': (0., 9*3600.),
    'unbiased_cc': False,         # correct afterwards at once
    'hash_waveforms': False,      # time consuming!
    'stationary_poi': {
        'name': 'MVC',
        'latitude': -25.887,
        'longitude': -177.188,
        'elevation': 0.,
        'local_depth': 132.
    },
    'attrs': {
        'title': 'Monowai Volcanic Centre',
        'institution': ('Delft University of Technology, '
                        'Department of Geoscience and Engineering'),
        'author': 'Pieter Smets - P.S.M.Smets@tudelft.nl',
        'source': 'CTBTO/IMS hydroacoustic data and IRIS/USGS seismic data',
    },
    'preprocess': {
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
            # ('remove_sensitivity', {}),
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
    },
}

# -----------------------------------------------------------------------------
# xcorr_client_args : `dict`
#     A dictionary with input argument to :class:`xcorr.Client`
# -----------------------------------------------------------------------------
xcorr_client_args = {
    'sds_root': '../data/WaveformArchive',
}

# -----------------------------------------------------------------------------
# pairs : `list`
#     Expects a list of `str`, with receiver couple SEED-id's
#     separated by a '-'.
# -----------------------------------------------------------------------------
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N2..EDH-IU.RAR.10.BHZ',
    'IM.H10N3..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S2..EDH-IU.RAR.10.BHZ',
    'IM.H03S3..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    # 'IM.H10N2..EDH-IU.RAR.10.BHR',
    # 'IM.H10N3..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
    # 'IM.H03S2..EDH-IU.RAR.10.BHR',
    # 'IM.H03S3..EDH-IU.RAR.10.BHR',
]

# -----------------------------------------------------------------------------
# times : `pandas.data_range`
#     Date range from start to end with ``freq``='D'.
# -----------------------------------------------------------------------------
times = pd.date_range(start='2015-01-15', end='2015-01-18', freq='1D')

# -----------------------------------------------------------------------------
# Mandatory parameters
# -----------------------------------------------------------------------------
inventory = '../data/Monowai.xml'  # path to inventory
root = '../data/results'           # path to output dir root
threads = 2                        # set dask number of workers
debug = False                      # run single-threaded
replace = False                    # don't open existing files


##############################################################################
#
# Functions
#
##############################################################################


@dask.delayed
def single_process(pair: str, time: pd.Timestamp, verb: int = 0, **kwargs):
    r"""Main xcorr processing sequence.
    """
    global root, replace, status, inventory, xcorr_init_args

    # File
    ncfile = xcorr.util.ncfile(pair, time, root)

    if not replace:
        # Open
        data = xcorr.read(ncfile, fast=True)

        # Update
        if data and np.all(data.status.values == 1):
            data.close()
            return False
    else:
        data = None

    # Create
    if not data:
        data = xcorr.init(
            pair=pair,
            starttime=time,
            endtime=time + pd.offsets.DateOffset(1),
            inventory=inventory,
            **xcorr_init_args
        )

    # Process
    xcorr.process(
        data,
        inventory=inventory,
        client=client,
        retry_missing=True,
        download=False,
        verb=verb,
        **kwargs
    )

    # Save
    if data and np.any(data.status.values == 1):
        xcorr.write(data, ncfile, verb=verb)
        return True
    else:
        return False


def lazy_processes(pairs: list, times: pd.DatetimeIndex, status: xr.DataArray,
                   verb: int = 0, **kwargs):
    r"""Construct list of lazy single processes for dask.compute
    """
    global xcorr_init_args

    results = []

    for pair in pairs:

        receivers = xcorr.util.split_pair(pair, substitute=True, to_dict=False)

        for time in times:

            if verb:
                print('    Check {} {}'.format(pair, time), end='. ')

            start = time - pd.offsets.DateOffset(
                seconds=xcorr_init_args['window_length']/2, normalize=True
            )
            end = time + pd.offsets.DateOffset(
                seconds=xcorr_init_args['window_length']*3/2, normalize=True
            )
            availability = status.loc[{
                'receiver': receivers,
                'time': pd.date_range(start, end, freq='D'),
            }] == 1

            # all receivers required for any day
            if any(availability.sum(dim='receiver') == len(receivers)):
                if verb:
                    print('Add lazy process.')
                result = single_process(pair, time, verb, **kwargs)
                results.append(result)
            else:
                if verb:
                    print('Skip.')

    return results


##############################################################################
#
# Main processing
#
##############################################################################


# Print some config parameters
print('-'*79)
print('Config')
print('    inventory :', inventory)
print('    root      :', root)
print('    threads   :', 1 if debug else threads)
print('    debug     :', debug)
print('    replace   :', replace)


# init the waveform client
client = xcorr.Client(**xcorr_client_args)

# Read and filter inventory
inventory = xcorr.util.get_pair_inventory(
    pairs, read_inventory(inventory), times
)

# minimize output to stdout in dask!
warnings.filterwarnings("ignore")

# Get waveform availability status
status = client.init_data_availability(
    pairs, times, extend_days=1, substitute=True
)

delayed_status = client.verify_data_availability(
    status, download=True, compute=False
)

# Print main data parameters
print('-'*79)
print('Data')
print('    pairs : {}'.format(len(pairs)))
for p in pairs:
    print('        {}'.format(p))
print('    receivers : {}'.format(len(status.receiver)))
for r in status.receiver:
    print('        {}'.format(str(r.values)))
print('    times : {} ({})'.format(len(times), len(status.time)))
print('        start : {}'.format(str(times[0])))
print('        end   : {}'.format(str(times[-1])))


# Evaluate data availability (parallel), and try to download missing data
print('-'*79)
print('Verify availability')
with ProgressBar():
    verified = delayed_status.compute()

# Print data availability status (total and per receiver)
print('Data availability')
print('    Total : {:.2f}%'
      .format(100 * np.sum(status.values == 1) / status.size))
print('    Receiver :')
for rec in status.receiver:
    pcnt = 100 * np.sum(
        status.loc[{'receiver': rec}].values == 1
    ) / status.time.size
    print('        {} : {:.2f}%'.format(rec.values, pcnt))

# Evaluate lazy process list
print('-'*79)
print('Compute')
if debug:
    results = dask.compute(
        lazy_processes(pairs, times, status, verb=1),
        scheduler='single-threaded'
    )
else:
    with ProgressBar():
        results = dask.compute(
            lazy_processes(pairs, times, status, verb=0),
            scheduler='processes', num_workers=threads
        )
