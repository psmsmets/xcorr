"""
operations
==========

xcorr processed waveforms by SEED channel code.

"""
import pandas as pd
from obspy import read_inventory
import xcorr

###############################################################################
# Client object
# -------------

# Create a client object.
client = xcorr.Client(sds_root='../../data/WaveformArchive')


###############################################################################
# Inventory
# ---------
inv = read_inventory('../../data/Monowai.xml')
f = inv.plot(color=0., projection='local')


###############################################################################
# Stream operations
# -----------------
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
# Processed waveforms
# -------------------

# BHR
BHR = client.get_processed_waveforms(
    receiver='IU.RAR.10.BHR',
    time=pd.to_datetime('2015-01-15T12:00'),
    duration=3600,
    operations=preprocess,
    inventory=inv,
    verb=2,
)
f = BHR.plot()

# BHT
BHT = client.get_processed_waveforms(
    receiver='IU.RAR.10.BHT',
    time=pd.to_datetime('2015-01-15T12:00'),
    duration=3600,
    operations=preprocess,
    inventory=inv,
    verb=5,
)
f = BHT.plot()

# BHZ
BHZ = client.get_processed_waveforms(
    receiver='IU.RAR.10.BHZ',
    time=pd.to_datetime('2015-01-15T12:00'),
    duration=3600,
    operations=preprocess,
    inventory=inv,
    verb=True,
)
f = BHZ.plot()

# EDH
EDH = client.get_processed_waveforms(
    receiver='IM.H10N1..EDH',
    time=pd.to_datetime('2015-01-15T12:00'),
    duration=3600,
    operations=preprocess,
    inventory=inv,
    sampling_rate=500.,
    verb=0,
)
f = EDH.plot()


###############################################################################
# Processed pair stream
# ---------------------

pair = client.get_pair_processed_waveforms(
    pair='IM.H10N1..EDH-IU.RAR.10.BHZ',
    time=pd.to_datetime('2015-01-15T12:00'),
    duration=3600,
    operations=preprocess,
    inventory=inv,
    verb=2,
)
f = pair[0].plot()
f = pair[1].plot()


###############################################################################
# Hash
# ----

# hash
h = xcorr.util.hasher.hash_Stream(pair)
print(h)

# or using the wrapper
h = xcorr.util.hasher.hash(pair)
print(h)
