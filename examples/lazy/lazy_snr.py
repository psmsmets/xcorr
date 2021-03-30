"""
SNR
===

Signal-to-noise ratio of a multi-file dataset using dask.

"""

import matplotlib.pyplot as plt
import dask
import dask.diagnostics
import os
import xcorr


###############################################################################
# Evaluate
# --------

# common plot settings
plotset = dict(aspect=2.5, size=4)

# settings
root = '../../data/cc'
pair = 'IM.H*'
year = 2015
doy = '*'
threads = 2

# dask compute arguments
compute_args = dict(num_workers=threads)

# dask progressbar
pbar = dask.diagnostics.ProgressBar()
pbar.register()

# open datasets
ds = xcorr.mfread(
    os.path.join(root, f'{year}', f'{pair}', f'{pair}.{year}.{doy}.nc'),
    fast=True
)
assert ds, 'No data found!'

# extract valid cc and postprocess
cc = (ds.cc.where((ds.status == 1), drop=True)
      .signal.unbias()
      .signal.demean()
      .signal.taper(max_length=5.)  # timeshift phase-wrap
      .signal.filter(frequency=3., btype='highpass', order=2)
      .signal.taper(max_length=3/2)  # filter artefacts
      )

# signal-to-noise
s = (cc.lag >= ds.distance/1.50) & (cc.lag <= ds.distance/1.46)
n = (cc.lag >= 6*3600) & (cc.lag <= 9*3600)
sn = cc.signal.snr(s, n, dim='lag', extend=True, envelope=False)

# compute
sn = sn.compute(**compute_args)

# print results
print(sn)

# plot
sn.snr.plot.line(x='time', hue='pair', marker='o', markersize=8, **plotset)
plt.tight_layout()
plt.show()

# to file
xcorr.write(ds, 'snr.nc')
