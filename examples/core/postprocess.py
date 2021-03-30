# -*- coding: utf-8 -*-
"""
Postprocess
===========

Postprocess xcorr data.

"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xcorr

###############################################################################
# Signal parameters
# -----------------

# cc data root
root = '../../data/cc'

# time of correlation data
time = pd.to_datetime('2015-01-15')

# receiver parameters
hydro = {
    'network': 'IM',
    'station': 'H03S1',
    'location': '',
    'channel': 'EDH',
}
seism = {
    'network': 'IU',
    'station': 'RAR',
    'location': '10',
    'channel': 'BHR',
}


###############################################################################
# Correlations
# ------------

# file list with two channels
ds = [
    xcorr.io.ncfile((hydro, seism), time, root),
    xcorr.io.ncfile((hydro, {**seism, 'channel': 'BHT'}), time, root)
]

# open merged list
ds = xcorr.merge(ds, quick_and_dirty=True)
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


###############################################################################
# Signal-to-noise ratio
# ---------------------
snr = ds.cc_w.signal.snr(s, n)

# plot of snr values
snr.plot.line(x='time', hue='pair', **plotset)
plt.tight_layout()
plt.show()


# lag plot cc colour coded per receiver when snr >= snr_min
def snr_lag_plot(cc, sn, sn_min=0., alpha=0.3):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[10, 4])
    lines = []
    for p, c in zip(cc.pair,  mpl.rcParams['axes.prop_cycle']()):
        sn_pass = sn.loc[{'pair': p}] >= sn_min
        if any(sn_pass):
            for t in cc.time[sn_pass]:
                line, = (cc
                         .sel(pair=p, time=t)
                         .plot.line(x='lag', alpha=alpha, ax=ax, **c)
                         )
            lines.append((line, str(p.values)))
    plt.legend(list(zip(*lines))[0], list(zip(*lines))[1])
    ax.set_title(f'{sn.long_name} > {sn_min}')
    plt.tight_layout()
    plt.show()


# plot each cc colour coded per receiver when snr >= 5.
snr_lag_plot(ds.cc_w.where(s), snr, 5.)


###############################################################################
# Signal-to-noise ratio
# ---------------------

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
