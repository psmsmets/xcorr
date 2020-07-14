"""
SNR
===

Signal-to-noise ratio of a multi-file dataset using dask.

"""

import matplotlib.pyplot as plt
import dask
import dask.diagnostics
import xarray as xr
import os
import xcorr


###############################################################################
# Delayed functions
# -----------------

@ dask.delayed
def _open(filename):
    """
    """
    ds = xcorr.read(filename, fast=True)
    return ds


@dask.delayed
def _close(ds):
    """
    """
    ds.close()
    return ds


@ dask.delayed
def _mask_valid(ds):
    """
    """
    mask = xcorr.signal.mask(
        x=ds.lag,
        upper=9./24.,
        scalar=ds.time.window_length
    )
    return mask


@ dask.delayed
def _mask_signal(ds):
    """
    """
    vel = dict(min=1.46, max=1.50)
    mask = xcorr.signal.multi_mask(
        x=ds.lag,
        y=ds.distance,
        lower=vel['min'],
        upper=vel['max'],
        invert=True,
    )
    return mask


@ dask.delayed
def _mask_noise(ds):
    """
    """
    mask = xcorr.signal.mask(
        x=ds.lag,
        lower=6./24.,
        upper=9./24.,
        scalar=ds.time.window_length
    )
    return mask


@ dask.delayed
def _select_and_trim(ds, valid):
    """
    """
    cc = ds.cc.where((valid) & (ds.status==1), drop=True)
    return cc


@ dask.delayed
def _unbias(cc):
    """
    """
    cc = xcorr.signal.unbias(cc)
    return cc


@ dask.delayed
def _filter(cc):
    """
    """
    cc = xcorr.signal.filter(cc, frequency=3., btype='highpass', order=2)
    return cc


@ dask.delayed
def _demean(cc):
    """
    """
    cc = xcorr.signal.demean(cc)
    return cc


@ dask.delayed
def _taper(cc):
    cc = xcorr.signal.taper(cc, max_length=2/3.)
    return cc


@ dask.delayed
def _snr(cc, signal, noise):
    """
    """
    snr = xcorr.signal.snr(cc, signal, noise)
    return snr


###############################################################################
# Lazy snr for filelist
# ---------------------

def lazy_snr_list(filenames: list):
    """Evaluate snr for a list of filenames
    """
    snr_list = []
    for filename in filenames:
        ds = _open(filename)
        valid = _mask_valid(ds)
        signal = _mask_signal(ds)
        noise = _mask_noise(ds)
        cc = _select_and_trim(ds, valid)
        cc = _filter(cc)
        cc = _demean(cc)
        cc = _taper(cc)
        snr = _snr(cc, signal, noise)
        ds = _close(ds)
        snr_list.append(snr)
    return snr_list


###############################################################################
# Evaluate
# --------

# common plot settings
plotset = dict(aspect=2.5, size=4)

# settings
root = '../../data/cc'
pair = 'IM.H03S*'
year = 2015
doy = '0*'
threads = 2

# dask compute arguments
compute_args = dict(num_workers=threads)

# dask progressbar
pbar = dask.diagnostics.ProgressBar()
pbar.register()


# list of files using dask
print('Compute validated list')
validated = xcorr.core.validate_list(
    os.path.join(root, f'{year}', f'{pair}', f'{pair}.{year}.{doy}.nc'),
    fast=True,
    paths_only=True,
    keep_opened=False,
    compute_args=compute_args,
)
assert validated, 'No data found!'

# snr
print('Compute signal-to-noise ratio')
sn = dask.compute(lazy_snr_list(validated), **compute_args)
ds = xr.merge(sn[0])
snr = ds.snr

# plot
snr.plot.line(x='time', hue='pair', marker='o', markersize=8, **plotset)
plt.tight_layout()
plt.show()

# to file
xcorr.write(ds, 'snr.nc')
