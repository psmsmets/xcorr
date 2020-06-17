"""
Signal
======

xcorr signal.

"""

import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import xcorr


###############################################################################
# Signal parameters
# -----------------

# results data root
root = '../../data/results'

# time of correlation data
time = pd.to_datetime('2015-01-15')

# receiver parameters
hydro = {
    'network': 'IM',
    'station': 'H10N1',
    'location': '',
    'channel': 'EDH',
}
seism = {
    'network': 'IU',
    'station': 'RAR',
    'location': '10',
    'channel': 'BHZ',
}

# signal parameters
vel = dict(min=1.46, max=1.50)
filter_params = dict(frequency=3., btype='highpass', order=2)
taper_params = dict(max_length=2/3.)


###############################################################################
# Correlation results
# -------------------

# file list with two channels
ds = [
    xcorr.util.ncfile((hydro, seism), time, root),
    xcorr.util.ncfile((hydro, {**seism, 'channel': 'BHR'}), time, root)
]

# open merged list
ds = xcorr.merge(ds)
assert ds, 'No data found!'

# apply signal processing
ds['cc'] = xcorr.signal.filter(ds.cc, **filter_params)
ds['cc'] = xcorr.signal.taper(ds.cc, **taper_params)
ds['cc_w'] = xcorr.signal.unbias(ds.cc)


###############################################################################
# Define windows
# --------------

noise_win = xcorr.signal.mask(
    x=ds.lag,
    lower=.2,
    upper=.25,
    scalar=ds.time.window_length
)

signal_win = xcorr.signal.mask(
    x=ds.lag,
    lower=1/vel['max'],
    upper=1/vel['min'],
    scalar=ds.distance.values[0]
)

valid_win = xcorr.signal.mask(
    x=ds.lag,
    upper=.25,
    scalar=ds.time.window_length
)

# select first time
t = ds.time[0]

# select last receiver pair
p = ds.pair[-1]

# some short-cuts
pt = dict(pair=p, time=t)
tn = dict(time=t, lag=ds.lag[noise_win])
ts = dict(time=t, lag=ds.lag[signal_win])
tv = dict(time=t, lag=ds.lag[valid_win])


###############################################################################
# Basic figures
# -------------

# default xarray plot settings
plotset = dict(aspect=2.5, size=4)

# (un)weighted cc.
# Manual figure combining two dataArray variables.
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[10, 4])
line1, = xr.plot.line(ds.cc_w.loc[pt], x='lag', ax=ax)
line2, = xr.plot.line(ds.cc.loc[pt], x='lag', ax=ax, _labels=False)
plt.legend((line1, line2), ('unbiased', 'biased'))
ax.set_ylabel('Crosscorrelation Estimate [-]')
plt.tight_layout()
plt.show()


# line plot wrapper
def lag_plot(var, prefix=''):
    var.plot.line(x='lag', **plotset)
    ax = plt.gca()
    ax.set_title(f'{prefix} {ax.get_title()}'.strip())
    plt.tight_layout()
    plt.show()


# valid window, weighted
lag_plot(ds.cc_w.loc[tv], 'Valid window,')

# signal window, weighted
lag_plot(ds.cc_w.loc[ts], 'Signal window,')

# noise window, weighted
lag_plot(ds.cc_w.loc[tn], 'Noise window,')


###############################################################################
# Signal-to-noise ratio
# ---------------------
ds['snr'] = xcorr.signal.snr(x=ds.cc_w, signal=signal_win, noise=noise_win)

# plot of snr values
ds.plot.scatter(x='time', y='snr', hue='pair', **plotset)
plt.tight_layout()
plt.show()


# lag plot cc colour coded per receiver when snr >= snr_min
def snr_lag_plot(ds, snr_min=0., var='cc_w', snr='snr', alpha=0.3):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[10, 4])
    lines = []
    for p, c in zip(ds.pair,  mpl.rcParams['axes.prop_cycle']()):
        snr_pass = ds[snr].loc[{'pair': p}] >= snr_min
        if any(snr_pass):
            for t in ds.time[snr_pass]:
                line, = xr.plot.line(
                    ds[var].loc[{
                        'time': t,
                        'lag': ds.lag[signal_win],
                        'pair': p,
                    }],
                    x='lag', alpha=alpha, ax=ax, **c
                )
            lines.append((line, str(p.values)))
    plt.legend(list(zip(*lines))[0], list(zip(*lines))[1])
    ax.set_title(f'{snr} > {snr_min}')
    plt.tight_layout()
    plt.show()


# plot each cc colour coded per receiver when snr >= 5.
snr_lag_plot(ds, 5.)


###############################################################################
# Signal-to-noise ratio
# ---------------------

# compute spectrogram
psd = xcorr.signal.spectrogram(
    ds.cc_w.loc[{'lag': ds.lag[signal_win]}],
    duration=2.,
    padding_factor=4,
)

# plot
plt.figure()
psd.loc[{'time': t, 'pair': p}].plot.imshow(x='lag')
plt.tight_layout()
plt.show()
