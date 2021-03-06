"""
Lazy process
============

xcorr lazy client data availability and preprocessing evaluation.

"""

# Mandatory imports
from pandas import date_range
from obspy import read_inventory
from xcorr import lazy_process
from shutil import rmtree
import distributed


##############################################################################
#
# Config
#
##############################################################################


# -----------------------------------------------------------------------------
# xcorr_init_args : `dict`
#     Dictionary with input arguments passed to :class:`xcorr.init`
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
            ('detrend', {'type': 'demean'}),
            ('filter', {'type': 'highpass', 'freq': .05}),
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
            ('detrend', {'type': 'demean'}),
            ('filter', {'type': 'highpass', 'freq': .05}),
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
            ('detrend', {'type': 'demean'}),
            ('filter', {'type': 'highpass', 'freq': .05}),
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
            ('detrend', {'type': 'demean'}),
            ('filter', {'type': 'highpass', 'freq': .05}),
            ('decimate', {'factor': 5}),
            ('remove_response', {}),
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
#     Dictionary with input arguments passed to :class:`xcorr.Client`
# -----------------------------------------------------------------------------
xcorr_client_args = {
    'sds_root': '../../data/WaveformArchive',
}

# -----------------------------------------------------------------------------
# pairs : `list`
#     Expects a list of `str`, with receiver couple SEED-id's separated
#     by a dash ('-').
# -----------------------------------------------------------------------------
pairs = [
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHT',
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H10N1..EDH-IU.RAR.10.BHT',
]

# -----------------------------------------------------------------------------
# times : `pandas.data_range`
#     Date range from start to end with ``freq``='D'.
# -----------------------------------------------------------------------------
times = date_range(start='2015-01-15', end='2015-01-18', freq='1D')

# -----------------------------------------------------------------------------
# inventory : :class:`obspy.Inventory`, optional
#     Inventory object, including the instrument response.
# -----------------------------------------------------------------------------
inventory = read_inventory('../../data/Monowai.xml')

# -----------------------------------------------------------------------------
# root : `str`
#     Path to output base directory. A folder will be created for each pair.
# -----------------------------------------------------------------------------
root = '../../data/cc'


##############################################################################
#
# Lazy process with dask
#
# An internal dask.distributed.LocalCluster and dask.distributed.Client
# are initiated and cleaned.
#
##############################################################################

# start a local cluster
cluster = distributed.LocalCluster(
    processes=False, n_workers=2, threads_per_worker=1,
)

# start a client
client = distributed.Client(cluster)

# lazy process on client
lazy_process(pairs, times, xcorr_init_args, xcorr_client_args, inventory, root,
             dask_client=client)

# clean-up
client.close()
cluster.close()
rmtree('dask-worker-space', ignore_errors=True)
