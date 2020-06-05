"""
Lazy client
===========

xcorr lazy client data availability evaluation.

"""
from pandas import date_range
from dask import distributed
from shutil import rmtree
import xcorr


###############################################################################
# dask client
# -----------

dcluster = distributed.LocalCluster(
    processes=False, threads_per_worker=1, n_workers=2,
)
dclient = distributed.Client(dcluster)

print('Dask client:', dclient)
print('Dask dashboard:', dclient.dashboard_link)

###############################################################################
# xcorr client
# ------------

# Create a client object.
xclient = xcorr.Client(sds_root='../../data/WaveformArchive')

# Inspect the client summary
print('xcorr client:')
print(xclient)


###############################################################################
# Verify data availability
# ------------------------

# set pairs and times
pairs = [
    'IM.H10N1..EDH-IU.RAR.10.BHZ',
    'IM.H10N1..EDH-IU.RAR.10.BHR',
    'IM.H03S1..EDH-IU.RAR.10.BHZ',
    'IM.H03S1..EDH-IU.RAR.10.BHR',
]
times = date_range('2015-01-01', '2015-01-10', freq='1D')

# evaluate data status
status = xclient.data_availability(
    pairs, times, verb=1, download=False, substitute=True
)


###############################################################################
# Cleanup
# -------

dclient.close()
dcluster.close()
rmtree('dask-worker-space')
