"""
Signal
======

xcorr signal.

"""

import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import xcorr


###############################################################################
# Read and merge
# --------------

# results data root
root = '../../data/results'

# get merged datasets using glob wildcards
# returns radial and vertical components of RAR.10
ds = xcorr.merge([f'{root}/*/*H03S1.*.RAR.10.BH?.2015.015.nc'], fast=True)
assert ds, 'No data found!'

# select a single time and receiver pair
t = ds.time[0]
p = ds.pair[-1]


###############################################################################
# Postprocessing
# --------------

ds['cc'] = xcorr.signal.filter(ds.cc, frequency=3., btype='highpass', order=2)
ds['cc'] = xcorr.signal.taper(ds.cc, max_length=2/3.)
ds['cc_w'] = xcorr.signal.unbias(ds.cc)


###############################################################################
# Mask windows
# ------------

# max valid domain
valid_win = xcorr.signal.mask(
    x=ds.lag,
    upper=9/.24,
    scalar=ds.time.window_length
)

# noise
noise_win = xcorr.signal.mask(
    x=ds.lag,
    lower=6./24.,
    upper=9./24.,
    scalar=ds.time.window_length
)

# signal
vel = dict(min=1.46, max=1.50)
signal_win = xcorr.signal.mask(
    x=ds.lag,
    lower=1/vel['max'],
    upper=1/vel['min'],
    scalar=ds.distance.values[0]
)

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
ds.snr.plot.line(x='time', hue='pair', marker='o', markersize=8, **plotset)
plt.tight_layout()
plt.show()


# lag plot cc colour coded per receiver when snr >= snr_min
def snr_lag_plot(ds, snr_min=0., var='cc_w', snr='snr', alpha=0.3, xlim=[]):
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
    if xlim:
        plt.xlim(xlim)
    plt.show()


# plot each cc colour coded per receiver when snr >= 5.
snr_lag_plot(ds, 5.)

# snr_lag_plot(ds, 5., xlim=[4920, 4950])
# snr_lag_plot(ds, 5., xlim=[9475, 9525])
# snr_lag_plot(ds, 5., xlim=[9490, 9500])

# snr_lag_plot(ds, 5., xlim=[4924.5, 4930.5])
# snr_lag_plot(ds, 5., xlim=[4929.5, 4935.5])
# snr_lag_plot(ds, 5., xlim=[4934.5, 4940.5])
# snr_lag_plot(ds, 5., xlim=[4939.5, 4945.5])
# snr_lag_plot(ds, 5., xlim=[4944.5, 4950.5])


###############################################################################
# Signal-to-noise ratio
# ---------------------

# compute spectrogram for time t and signal window only
psd = xcorr.signal.spectrogram(
    ds.cc_w.loc[{'lag': ds.lag[signal_win], 'time': t}],
    duration=2.,
    padding_factor=4,
)

for pair in psd.pair:
    plt.figure()
    psd.loc[{'pair': psd.pair[0]}].plot.imshow(x='lag')
    # plt.xlim([4920, 4950])
    # plt.xlim([9475, 9525])
    plt.ylim([2, 12])
    plt.tight_layout()
    plt.show()
