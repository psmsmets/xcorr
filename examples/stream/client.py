"""
Client
======

xcorr waveform client.

"""
import pandas as pd
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
