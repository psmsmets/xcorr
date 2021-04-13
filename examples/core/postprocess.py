# -*- coding: utf-8 -*-
"""
Postprocess
===========

Postprocess xcorr data.

"""

import pandas as pd
import matplotlib.pyplot as plt
import xcorr

###############################################################################
# Signal parameters
# -----------------

# cc data root
root = '../../data/cc'

# time of correlation data
time = pd.to_datetime('2015-01-15')

# receiver pairs (as a glob string)
pair = 'IM.H03S1..EDH-IU.RAR.10.BH[RZ]'


###############################################################################
# Correlations
# ------------

# Read files given glob strings
nc = xcorr.io.ncfile(pair, time, root, verify_receiver=False)

# open merged files
ds = xcorr.merge(nc, quick_and_dirty=True)
assert ds, 'No data found!'

# apply signal processing
ds['cc'] = (ds.cc
            .signal.filter(frequency=3., btype='highpass', order=2)
            .signal.taper(max_length=5.)
            )
ds['cc_w'] = ds.cc.signal.unbias()


###############################################################################
# Define windows
# --------------

v = (ds.lag <= 9*3600)
s = (ds.lag >= ds.distance/1.50) & (ds.lag <= ds.distance/1.46)
n = (ds.lag >= 6*3600) & v


###############################################################################
# Basic figures
# -------------

# default xarray plot settings
plotset = dict(aspect=2.5, size=4)

# (un)weighted cc.
# Manual figure combining two dataArray variables.
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[10, 4])
line1, = ds.cc_w.isel(time=0, pair=-1).plot.line(x='lag', ax=ax)
line2, = ds.cc.isel(time=0, pair=-1).plot.line(x='lag', ax=ax, _labels=False)
plt.legend((line1, line2), ('unbiased', 'biased'))
ax.set_ylabel('Cross-correlation Estimate [-]')
plt.tight_layout()
plt.show()


# line plot wrapper
def lag_plot(var, prefix=''):
    var.plot.line(x='lag', **plotset)
    ax = plt.gca()
    ax.set_title(f'{prefix} {ax.get_title()}'.strip())
    plt.tight_layout()


# valid, signal and noise
lag_plot(ds.cc_w.isel(time=0).where(v), 'Valid window,')
lag_plot(ds.cc_w.isel(time=0).where(s), 'Signal window,')
lag_plot(ds.cc_w.isel(time=0).where(n), 'Noise window,')
plt.show()


# or all at once for signal
xcorr.plot.plot_ccfs(ds.cc_w.where(s, drop=True), ds.distance)
plt.show()


###############################################################################
# Signal-to-noise ratio
# ---------------------
sn = ds.cc_w.signal.snr(s, n)

# plot of snr values
sn.plot.line(x='time', hue='pair', **plotset)
plt.tight_layout()
plt.show()

# plot each cc colour coded per receiver when snr >= 25.
plt.figure(figsize=[10, 4])
xcorr.plot.plot_ccfs_colored(ds.cc_w.where(s, drop=True), sn, 25.)
plt.tight_layout()
plt.show()

###############################################################################
# Spectrogram
# -----------

# compute spectrogram
psd = (ds.cc_w
       .where(s, drop=True)
       .signal.spectrogram(duration=2., padding_factor=4)
       )

# plot
plt.figure()
psd.isel(time=0, pair=-1).plot.imshow(x='lag')
plt.tight_layout()
plt.show()

# or all at once
xcorr.plot.plot_ccf(ds.cc_w.where(s, drop=True), ds.distance, time=0, pair=-1)
plt.show()
