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
root = "../../data"
year = "2015"
days = "0??"
pair = "*H03*"

ds = xcorr.mfread(
    f"{root}/cc/{year}/{pair}/{pair}.{year}.{days}.nc",
    chunks={'time': 4}, fast=True, naive=True,
)
assert ds, 'No data found!'

# common plot settings
plotset = dict(aspect=2.5, size=4)


###############################################################################
# Masks
# -----

s = (ds.lag >= ds.distance/1.50) & (ds.lag <= ds.distance/1.46)
n = (ds.lag >= 6*3600) & (ds.lag < 9*3600)


###############################################################################
# Postprocess
# -----------

cc = (ds.cc
      .where(ds.status == 1, drop=True)
      .signal.unbias()
      .signal.demean()
      .signal.filter(frequency=3., btype='highpass', order=2)
      .signal.taper(max_length=2/3.)
      )


###############################################################################
# Signal-to-noise ratio
# ---------------------

sn = cc.signal.snr(s, n)

# compute with dask (2 threads)
with dask.diagnostics.ProgressBar():
    sn.compute(num_workers=2)

# plot
sn.plot.line(x='time', hue='pair', marker='o', markersize=8, **plotset)
plt.tight_layout()
plt.show()
