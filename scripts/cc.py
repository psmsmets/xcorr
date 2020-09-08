"""
CC
===

Crosscorrelation using dask.

"""
from pandas import date_range
from obspy import read_inventory
from xcorr import lazy_process


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
#     A dictionary with input argument to :class:`xcorr.Client`
# -----------------------------------------------------------------------------
xcorr_client_args = {
    'sds_root_read': ['/vardim/home/smets/Data/WaveformArchive',
                      '/vardim/home/smets/Hydro'],
    'sds_root_write': '/vardim/home/smets/Hydro',
    'fdsn_service': 'iris',
    'vdms_service': False,
    'max_gap': .001*86400,  # .1%
    'parallel': True,
}

# -----------------------------------------------------------------------------
# pairs : `list`
#     Expects a list of `str`, with receiver couple SEED-id's
#     separated by a '-'.
# -----------------------------------------------------------------------------
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N2..EDH-IU.RAR.10.BHZ',
    'IM.H10N3..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S2..EDH-IU.RAR.10.BHZ',
    'IM.H03S3..EDH-IU.RAR.10.BHZ',
]

# -----------------------------------------------------------------------------
# times : `pandas.data_range`
#     Date range from start to end with ``freq``='D'.
# -----------------------------------------------------------------------------
times = date_range(start='2014-01-01', end='2015-01-01', freq='1D')

# -----------------------------------------------------------------------------
# inventory : :class:`obspy.Inventory`, optional
#     Inventory object, including the instrument response.
# -----------------------------------------------------------------------------
inventory = read_inventory(
    '/vardim/home/smets/Research/hydro/Monowai/Monowai.xml'
)

# -----------------------------------------------------------------------------
# root : `str`
#     Path to output base directory. A folder will be created for each pair.
# -----------------------------------------------------------------------------
root = '/ribarsko/data/smets/hydro/cc'

##############################################################################
#
# Lazy process with dask
#
# An internal dask.distributed.LocalCluster and dask.distributed.Client
# are initiated and cleaned.
#
##############################################################################

# connect to existing client
client = distributed.client(scheduler-file='~/.scheduler.json')

# lazy processes
lazy_process(pairs, times, xcorr_init_args, xcorr_client_args, inventory, root,
             dask_client=client, download=False)
