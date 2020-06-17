"""
Mfread
======

xcorr multi-file read dataset using dask.

"""

import matplotlib.pyplot as plt
import dask
import dask.diagnostics
import xcorr


###############################################################################
# Merged results
# --------------

# open merged list using dask
ds = xcorr.mfread(
    '../../data/results/2015/*/*2015.0*.nc',
    chunks={'time': 4}
)
assert ds, 'No data found!'

# common plot settings
plotset = dict(aspect=2.5, size=4)


###############################################################################
# Masks
# -----

# max valid domain
valid_mask = xcorr.signal.mask(
    x=ds.lag,
    upper=9./24.,
    scalar=ds.time.window_length
)

# noise
noise_mask = xcorr.signal.mask(
    x=ds.lag,
    lower=6./24.,
    upper=9./24.,
    scalar=ds.time.window_length
)

# signal
vel = dict(min=1.46, max=1.50)
signal_mask = xcorr.signal.multi_mask(
    x=ds.lag,
    y=ds.distance,
    lower=vel['min'],
    upper=vel['max'],
    invert=True,
)


###############################################################################
# Postprocess
# -----------

cc = ds.cc.where(valid_mask, drop=True)  # reduce dataset (faster)
cc = xcorr.signal.unbias(cc)
cc = xcorr.signal.filter(cc, frequency=3., btype='highpass', order=2)
cc = xcorr.signal.detrend(cc)
cc = xcorr.signal.taper(cc, max_length=2/3.)


###############################################################################
# Signal-to-noise ratio
# ---------------------

snr = xcorr.signal.snr(cc, signal_mask, noise_mask)

# compute with dask (2 threads)
with dask.diagnostics.ProgressBar():
    dask.compute(snr, num_workers=2)

# plot
snr.plot.line(x='time', hue='pair', marker='o', markersize=8, **plotset)
plt.tight_layout()
plt.show()


raise SystemExit()

###############################################################################
# Spectrogram
# -----------

# should be done per pair to avoid timeshifts due to NaN in the data
pair = ds.pair[1]

# extract and preprocess cc signal
sig = ds.cc.sel(pair=pair).where(valid_mask, drop=True)
sig = xcorr.signal.filter(sig, frequency=3., btype='highpass', order=2)
sig = sig.where(signal_mask.sel(pair=pair), drop=True)
sig = xcorr.signal.unbias(sig)

# spectral density
psd = xcorr.signal.spectrogram(
    sig,
    duration=2., padding_factor=4, scaling='density'
)

# compute
with dask.diagnostics.ProgressBar():
    dask.compute(psd, num_workers=2)

# plot
psd.sel(time=psd.time[0]).plot.imshow(x='lag')
plt.tight_layout()
plt.show()
