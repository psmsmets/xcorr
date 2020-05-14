"""
Client
======

xcorr client.

"""
import pandas as pd
from obspy import read_inventory
from dask.diagnostics import ProgressBar
import xcorr

###############################################################################
# Client object
# -------------

# Create a client object.
client = xcorr.Client(sds_root='../../data/WaveformArchive')

# Inspect the client summary
print(client)


###############################################################################
# Get waveforms
# -------------

# Get waveforms for an entire day (default duration is 86400s)
EDH = client.get_waveforms(
    receiver='IM.H10N1..EDH',
    time=pd.to_datetime('2015-01-15T00:00'),
    centered=False,
    verb=3,
)
# View the stream
print(EDH)

# Validate the duration
client.check_duration(EDH, sampling_rate=250.)

# Plot
f = EDH.plot()


###############################################################################
# Inventory
# ---------
inv = read_inventory('../Monowai.xml')
f = inv.plot(color=0., projection='local')


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
# Preprocessed waveforms
# ----------------------

# EDH
H10 = client.get_preprocessed_waveforms(
    receiver='IM.H10N1..EDH',
    time=pd.to_datetime('2015-01-15T12:00'),
    preprocess=preprocess,
    inventory=inv,
    verb=0,
)
f = H10.plot()

# BHZ
BHZ = client.get_preprocessed_waveforms(
    receiver='IU.RAR.10.BHZ',
    time=pd.to_datetime('2015-01-15T12:00'),
    preprocess=preprocess,
    inventory=inv,
    verb=True,
)
f = BHZ.plot()

# BHR
BHR = client.get_preprocessed_waveforms(
    receiver='IU.RAR.10.BHR',
    time=pd.to_datetime('2016-01-15T12:00'),
    preprocess=preprocess,
    inventory=inv,
    verb=2,
)
f = BHR.plot()


###############################################################################
# Preprocessed pair stream
# ------------------------

pair = client.get_pair_preprocessed_waveforms(
    pair='IM.H10N1..EDH-IU.RAR.10.BHZ',
    time=pd.to_datetime('2015-01-15T12:00'),
    preprocess=preprocess,
    inventory=inv,
    verb=2,
)
f = pair[0].plot()
f = pair[1].plot()


###############################################################################
# Hash
# ----

# hash
xcorr.util.hasher.hash_Stream(pair)

# or using the wrapper
xcorr.util.hasher.hash(pair)


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
status = client.data_availability(
    pairs, times, verb=2, download=False, substitute=True
)

# print summary per receiver
print(status.sum(dim='time') / status.time.size)

# or in separate steps to control dask delayed compute showing a progressbar
status = client.init_data_availability(pairs, times, substitute=True)
delayed_status = client.verify_data_availability(
    status, download=False, compute=False
)
with ProgressBar():
    verified = delayed_status.compute()

# number of verified days per receiver
print(verified)


###############################################################################
# Verify preprocessing
# --------------------

# on-the-fly
status = client.data_preprocessing(pairs, times[0], preprocess, inv)

# in steps
status = client.init_data_preprocessing(
    pairs, times[0], preprocess=preprocess, substitute=True
)
delayed_status = client.verify_data_preprocessing(
    status, inventory=inv, download=False, compute=False
)
with ProgressBar():
    verified = delayed_status.compute()
