"""
Signal
======

xcorr signal.

"""

import matplotlib.pyplot as plt
import os
import xcorr


###############################################################################
# Merged results
# --------------

# results data root
data = '../../data/results'

# filter to select data
filt = ''

# list of files
ncfiles = []
for root, dirs, files in os.walk(data):
    path = root.split(os.sep)
    for f in files:
        if f[1] != '.' and f[-3:] == '.nc' and filt in f if filt else f:
            ncfiles += [os.path.join(root, f)]

# open merged list
ds = xcorr.merge(sorted(ncfiles), extract=False, strict=False, fast=True)
assert ds, 'No data found!'

###############################################################################
# Signal
# ------

# parameters
vel = dict(min=1.46, max=1.50)
filter_params = dict(frequency=3., btype='highpass', order=2, inplace=True)
taper_params = dict(max_length=2/3., inplace=True)

# apply signal processing
xcorr.signal.filter(ds.cc, **filter_params)
xcorr.signal.taper(ds.cc, **taper_params)
xcorr.bias_correct(ds)


###############################################################################
# Mask windows
# ------------

noise_mask = xcorr.signal.mask(
    x=ds.lag,
    lower=6./24.,
    upper=9./.24,
    scalar=ds.time.window_length
)

signal_mask = xcorr.signal.multi_mask(
    x=ds.lag, y=ds.distance,
    lower=vel['min'], upper=vel['max'], invert=True,
)

###############################################################################
# Signal-to-noise ratio
# ---------------------

# calculate snr
ds['snr'] = xcorr.signal.snr(
    ds.cc, signal=signal_mask, noise=noise_mask, dim='lag'
)

# default xarray plot settings
plotset = dict(aspect=2.5, size=4)

# plot snr
ds.plot.scatter(x='time', y='snr', hue='pair', **plotset)
plt.tight_layout()
plt.show()

# plot snr
ds.snr.plot.line(x='time', hue='pair', marker='o', markersize=10, **plotset)
plt.tight_layout()
plt.show()
