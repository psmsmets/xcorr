"""
Client
======

xcorr client.

"""
import pandas as pd
from obspy import read_inventory
from dask import distributed
from shutil import rmtree
import xcorr


###############################################################################
# dask client
# -----------

dcluster = distributed.LocalCluster(
    processes=False, threads_per_worker=1, n_workers=2,
)
dclient = distributed.Client(dcluster)

print('Dask client:', dclient)
print('Dask dashboard:', dclient.dashboard_link)

###############################################################################
# xcorr client
# ------------

# Create a client object.
xclient = xcorr.Client(sds_root='../../data/WaveformArchive')

# Inspect the client summary
print('xcorr client:')
print(xclient)


###############################################################################
# Inventory
# ---------
inv = read_inventory('../../data/Monowai.xml')


###############################################################################
# Preprocess settings
# -------------------
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
# Verify data availability
# ------------------------

# set pairs and times
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
]
times = pd.date_range('2015-01-01', '2015-01-10', freq='1D')

# evaluate data status
status = xclient.data_availability(
    pairs, times, verb=1, download=False, substitute=True
)


###############################################################################
# Verify preprocessing
# --------------------

# on-the-fly evaluate data preprocessing
status = xclient.data_preprocessing(
    pairs, times[0], preprocess, inv, verb=1, substitute=True
)


###############################################################################
# Cleanup
# -------

dclient.close()
dcluster.close()
rmtree('dask-worker-space')
