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
inventory = '../data/Monowai.xml'	# path to inventory
root = '../data/results'		# path to output dir root
threads = 2				# set dask number of workers
debug = False				# run single-threaded
force_overwrite = False			# don't open existing files
progressbar = True			# show dask delayed progress


##############################################################################
#
# Functions
#
##############################################################################


@dask.delayed
def single_process(pair: str, time: pd.Timestamp, verb: int = 0, **kwargs):
    r"""Main xcorr processing sequence.
    """
    global root, force_overwrite, inventory, xcorr_init_args

    # File
    ncfile = xcorr.util.ncfile(pair, time, root)

    if not force_overwrite:
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


def lazy_processes(pairs: list, times: pd.DatetimeIndex,
                   availability: xr.DataArray, preprocessing: xr.DataArray,
                   verb: int = 0, **kwargs):
    r"""Construct list of lazy single processes for dask.compute
    """
    global xcorr_init_args

    results = []

    for pair in pairs:

        # check preprocessing
        pair_preprocessing = preprocessing.loc[{
            'receiver': xcorr.util.split_pair(pair, substitute=False,
                                              to_dict=False),
            'time': preprocessing.time[0],
        }] == 1
        preprocessing_passed = np.all(pair_preprocessing.values == 1)
        preprocessing_status = 'passed' if preprocessing_passed else 'failed'

        # substituted receivers
        receivers = xcorr.util.split_pair(pair, substitute=True, to_dict=False)

        for time in times:

            if verb > 0:
                print('    Check {} {}'.format(pair, time), end='. ')

            # preprocessing status
            if verb > 2:
                print('Preprocessing', preprocessing_status)

            # check availability
            start = time - pd.offsets.DateOffset(
                seconds=xcorr_init_args['window_length']/2, normalize=True
            )
            end = time + pd.offsets.DateOffset(
                seconds=xcorr_init_args['window_length']*3/2, normalize=True
            )
            pair_availability = availability.loc[{
                'receiver': receivers,
                'time': pd.date_range(start, end, freq='D'),
            }] == 1

            availability_passed = np.any(
                pair_availability.sum(dim='receiver') == len(receivers)
            )

            # availability status
            if verb > 2:
                print('Availability',
                      'passed' if availability_passed else 'failed', end='. ')

            # preprocessing and availability passed
            if preprocessing_passed and availability_passed:
                if verb > 0:
                    print('Add lazy process.')
                result = single_process(pair, time, verb, **kwargs)
                results.append(result)
            else:
                if verb > 0:
                    print('Skip.')

    return results


##############################################################################
#
# Main processing
#
##############################################################################


# -----------------------------------------------------------------------------
# Print some config parameters
# -----------------------------------------------------------------------------
print('-'*79)
print('Config')
print('    inventory       :', inventory)
print('    root            :', root)
print('    threads         :', 1 if debug else threads)
print('    debug           :', debug)
print('    force_overwrite :', force_overwrite)

# -----------------------------------------------------------------------------
# various inits
# -----------------------------------------------------------------------------

# init the waveform client
client = xcorr.Client(**xcorr_client_args)

# Read and filter inventory
inventory = xcorr.util.get_pair_inventory(
    pairs, read_inventory(inventory), times
)

# minimize output to stdout in dask!
warnings.filterwarnings("ignore")

# waveform availability status
availability = client.init_data_availability(
    pairs, times, extend_days=1, substitute=True
)
lazy_availability = client.verify_data_availability(
    availability, download=True, compute=False
)

# waveform preprocess status
preprocessing = client.init_data_preprocessing(
    pairs, times[0],
    preprocess=xcorr_init_args['preprocess'], substitute=True
)
lazy_preprocessing = client.verify_data_preprocessing(
    preprocessing, inventory=inventory, download=False, compute=False
)

# progressbar
if progressbar:
    pbar = ProgressBar()
    pbar.register()

# -----------------------------------------------------------------------------
# Print main data parameters
# -----------------------------------------------------------------------------
print('-'*79)
print('Data')
print('    pairs : {}'.format(len(pairs)))
for p in pairs:
    print('        {}'.format(p))
print('    times : {} ({})'.format(len(times), len(availability.time)))
print('        start : {}'.format(str(times[0])))
print('        end   : {}'.format(str(times[-1])))

# -----------------------------------------------------------------------------
# Evaluate data availability (parallel), and try to download missing data
# -----------------------------------------------------------------------------
print('-'*79)
print('Verify availability')
verified = lazy_availability.compute()

# Print data availability status (total and per receiver)
print('    Overall availability : {:.2f}%'
      .format(100 * np.sum(availability.values == 1) / availability.size))
print('    Receiver availability')
for rec in availability.receiver:
    pcnt = 100 * np.sum(
        availability.loc[{'receiver': rec}].values == 1
    ) / availability.time.size
    print('        {} : {:.2f}%'.format(rec.values, pcnt))

# -----------------------------------------------------------------------------
# Evaluate data preprocessing (parallel)
# -----------------------------------------------------------------------------
print('-'*79)
print('Verify preprocessing')
verified = lazy_preprocessing.compute()

# Print data availability status (total and per receiver)
print('    Overall preprocessing : {:.2f}% passed'
      .format(100 * np.sum(preprocessing.values == 1) / preprocessing.size))
print('    Receiver preprocessing')
for rec in preprocessing.receiver:
    passed = np.all(preprocessing.loc[{'receiver': rec}].values == 1)
    print('        {} :'.format(rec.values),
          'passed' if passed else 'failed')

# -----------------------------------------------------------------------------
# Evaluate lazy process list
# -----------------------------------------------------------------------------
print('-'*79)
print('Compute lazy processes')
if debug:
    results = dask.compute(
        lazy_processes(pairs, times, availability, preprocessing, verb=1),
        scheduler='single-threaded'
    )
else:
    results = dask.compute(
        lazy_processes(pairs, times, availability, preprocessing, verb=0),
        scheduler='processes', num_workers=threads
    )
