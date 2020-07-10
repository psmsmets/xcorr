"""
PSD
===

Spectrograms of triggered datasets by snr using dask.

"""

import matplotlib.pyplot as plt
import dask
import dask.diagnostics
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import os
import xcorr


###############################################################################
# Delayed functions
# -----------------


@dask.delayed
def _load(pair, period, root):
    """Load cc of a single pair for a period.
    """
    t0 = period.start + pd.tseries.offsets.DateOffset(normalize=True)
    t1 = period.end + pd.tseries.offsets.DateOffset(normalize=True)

    files = []
    for t in pd.date_range(t0, t1, freq='1D'):
        files.append(xcorr.util.ncfile(str(pair.values), t, root))

    ds = xcorr.merge(files)

    # extract valid data
    vel = dict(min=1.465, max=1.495)
    t0, t1 = np.datetime64(period.start), np.datetime64(period.end)
    mask = xcorr.signal.multi_mask(
        x=ds.lag,
        y=ds.distance,
        lower=vel['min'],
        upper=vel['max'],
        invert=True,
    ) & (ds.status == 1) & (ds.time >= t0) & (ds.time <= t1)
    d = ds.distance
    ds = ds.where(mask, drop=True)
    ds['distance'] = d  # time_offset and pair_offset need to be fixed as well!
    ds = ds.drop_vars(['status', 'pair_offset', 'time_offset'])

    # phase shift to reset time_offset and pair_offset!

    return ds


@dask.delayed
def _extract_period(ds, period):
    """Extract time period
    """
    t0, t1 = np.datetime64(period.start), np.datetime64(period.end)
    mask = (ds.time >= t0) & (ds.time <= t1)
    ds = ds.where(mask, drop=True)
    return ds


@dask.delayed
def _preprocess(ds):
    """Preprocess cc
    """
    cc = ds.cc
    cc = xcorr.signal.detrend(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=2/3.)
    return cc


@dask.delayed
def _spectrogram(cc):
    """Calculate spectrogram
    """
    psd = xcorr.signal.spectrogram(cc, duration=2., padding_factor=4)
    psd = psd.where((psd.freq >= 1.5) & (psd.freq <= 18.), drop=True)
    return psd


@dask.delayed
def _combine(ds, cc, psd, snr):
    """Combine all into a single dataset
    """
    ds['cc'] = cc
    ds['psd'] = psd
    ds['snr'] = snr.loc[{'pair': ds.pair[0]}]
    return ds


@dask.delayed
def _write(ds, period, root):
    """Merge spectrogram list
    """
    # parameters
    pair = str(ds.pair[0].values)

    # set filename and path
    filename = '{p}.{y:04d}.{d:03d}.{h:03d}.psd.nc'.format(
        p=pair,
        y=period.start.year,
        d=period.start.dayofyear,
        h=int(period.days*24),
    )
    path = os.path.join(root, '..', 'psd', pair, filename)

    # write
    xcorr.write(ds, path, verb=-1, hash_data=False)

    return path


###############################################################################
# Lazy psd for pairs and periods
# ------------------------------

def lazy_spectrogram(snr, trigs, root):
    """Evaluate psds for a pair and a set of periods
    """
    periods = xcorr.signal.trigger.trigger_periods(trigs)
    fnames = []

    for index, period in periods.iterrows():
        snr_period = _extract_period(snr, period)

        for pair in snr.pair:
            ds = _load(pair, period, root)
            cc = _preprocess(ds)
            psd = _spectrogram(cc)
            ds = _combine(ds, cc, psd, snr)
            fname = _write(ds, period, root)
            fnames.append(fname)
    return fnames


###############################################################################
# Parameters
# ----------

# common plot settings
plotset = dict(aspect=2.5, size=4, add_legend=False)

# settings
root = '../../data'
threads = 2


# dask compute arguments
compute_args = dict(num_workers=threads)

# dask progressbar
pbar = dask.diagnostics.ProgressBar()
pbar.register()


###############################################################################
# Active periods
# --------------

# get snr
"""
snr = xr.merge([
    xr.open_dataarray(f) for f in glob(f'{root}/snr/snr_201[4-5].nc')
]).snr
snr = snr.where(
    (snr.time >= np.datetime64('2015-01-10')) &
    (snr.time <= np.datetime64('2015-01-20'))
)
"""
snr = xr.open_dataarray('snr.nc')

# get triggered periods
ct = xcorr.signal.coincidence_trigger(
    snr, thr_on=8., thr_off=5., extend=0, thr_coincidence_sum=None
)

# plot
snr.plot.line(x='time', hue='pair', **plotset)
xcorr.signal.trigger.plot_trigs(snr, ct)
plt.ylim(0, 200)
plt.tight_layout()
plt.show()


# construct datasets with preprocessed cc, snr and psd
files = lazy_spectrogram(snr, ct, os.path.join(root, 'cc'))
files = dask.compute(files, **compute_args)[0]

print(files)
