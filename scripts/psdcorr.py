"""
PSDCORR
===

Crosscorrelation of crosscorrelation spectrograms.

"""
import xarray as xr
import numpy as np
from dask import distributed
import matplotlib.pyplot as plt
import os
from glob import glob
import xcorr


##############################################################################
#
# Config
#
##############################################################################
# root = '/ribarsko/data/smets/hydro/cc'
root = '../data'
pair = 'H03'
freq = np.array(((3., 6.), (6., 12.)))
n_workers = 2


##############################################################################
#
# Extract source activity
#
##############################################################################


# -----------------------------------------------------------------------------
# Filter SNR
# -----------------------------------------------------------------------------
snr = xr.merge([xr.open_dataarray(f) for f in
                glob(os.path.join(root, 'snr', 'snr_20??.nc'))]).snr

snr = snr.where(
    (
        (snr.time >= np.datetime64('2015-01-15')) &
        (snr.time <= np.datetime64('2015-01-18')) &
        (snr.pair.str.contains(pair))
    ),
    drop=True,
)

# -----------------------------------------------------------------------------
# Confindence triggers
# -----------------------------------------------------------------------------
ct = xcorr.signal.coincidence_trigger(
    snr, thr_on=10., extend=0, thr_coincidence_sum=None
)
time = ct.time.where(ct >= 0, drop=True)


##############################################################################
#
# Lazy process with dask
#
##############################################################################

# -----------------------------------------------------------------------------
# Local functions
# -----------------------------------------------------------------------------
def get_spectrogram(pair, time, root: str = None,
                    client: distributed.Client = None):
    """Load spectrogram for a pair and time.
    """
    # construct abs path and filename
    nc = xcorr.util.ncfile(pair, time, root)

    # select pair time location
    sel = {'pair': pair, 'time': time}

    # set lock
    lock = distributed.Lock(nc, client)
    lock.acquire()

    # read file
    ds = xcorr.read(nc, quick_and_dirty=True)
    ds.load().close()

    # release lock
    lock.release()

    # extract cc
    mask = xcorr.signal.multi_mask(
        x=ds.lag,
        y=ds.distance,
        lower=1.465,
        upper=1.495,
        invert=True,
    )
    cc = ds.cc.where(mask, drop=True)
    cc = cc.loc[sel]

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.detrend(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    # solve time_offset and pair_offset
    delay = xcorr.util.time.to_seconds(
        (ds.pair_offset.loc[sel] + ds.time_offset.loc[sel]).values
    )
    if delay != 0.:
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', pad=True)

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2., padding_factor=4)

    return psd


def correlate_spectrograms(obj, **kwargs):
    """Correlate spectrograms.
    """
    # already set?
    if obj.status.any():
        return obj

    # test if object is loaded
    if not (obj.freq.any() and obj.pair.any()):
        return obj

    # load cc and compute psd on-the-fly
    psd1 = get_spectrogram(obj.pair[0], obj.time1[0], **kwargs)
    psd2 = get_spectrogram(obj.pair[0], obj.time2[0], **kwargs)

    # per freq
    for freq in obj.freq:

        # set (min, max) frequency
        fmin = (obj.freq - obj.freq_bw/2).values[0]
        fmax = (obj.freq + obj.freq_bw/2).values[0]

        # extract freq
        in1 = psd1.where((psd1.freq >= fmin) & (psd1.freq <= fmax), drop=True)
        in2 = psd2.where((psd2.freq >= fmin) & (psd2.freq <= fmax), drop=True)

        # correlate psd's
        cc = xcorr.signal.correlate2d(in1, in2)

        # split dims
        dim1, dim2 = cc.dims[-2:]

        # get max index
        amax1, amax2 = np.unravel_index(cc.argmax(), cc.shape)

        # store values in object
        obj['status'].loc[{'freq': freq}] = np.byte(1)
        obj['cc'].loc[{'freq': freq}] = cc.isel({dim1: amax1, dim2: amax2})
        obj[dim1].loc[{'freq': freq}] = cc[dim1][amax1]
        obj[dim2].loc[{'freq': freq}] = cc[dim2][amax2]

    return obj


def mask_upper_triangle(ds):
    """Mark upper triangle (one offset)
    """
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        time1 = ds.time1[ind1[i]]
        time2 = ds.time2[ind2[i]]
        ds.status.loc[{'time1': time1, 'time2': time2}] = np.byte(1)


def fill_upper_triangle(ds):
    """Fill upper triangle (one offset)
    """
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        time1 = ds.time1[ind1[i]]
        time2 = ds.time2[ind2[i]]
        triu = {'time1': time1, 'time2': time2}
        tril = {'time1': time2, 'time2': time1}
        ds.cc.loc[triu] = ds.cc.loc[tril]
        ds.delta_freq.loc[triu] = -ds.delta_freq.loc[tril]
        ds.delta_lag.loc[triu] = -ds.delta_lag.loc[tril]


# -----------------------------------------------------------------------------
# Init cc correlation dataset
# -----------------------------------------------------------------------------

# new dataset
corr = xr.Dataset()
corr.attrs = snr.attrs

# set coordinates
corr['pair'] = snr.pair
corr.pair.attrs = snr.pair.attrs

corr['freq'] = xr.DataArray(
    np.mean(freq, axis=1),
    dims=('freq'),
    coords=(np.mean(freq, axis=1),),
    attrs={
        'long_name': 'Frequency',
        'standard_name': 'frequency',
        'units': 's-1',
    },
)

# set variables
corr['freq_bw'] = xr.DataArray(
    np.diff(freq, axis=1).squeeze(),
    dims=('freq'),
    coords=(corr.freq,),
    attrs={
        'long_name': 'Frequency bandwidth',
        'standard_name': 'frequency_bandwidth',
        'units': 's-1',
    },
)

corr['status'] = xr.DataArray(
    np.zeros((time.size, time.size, len(freq), len(corr.pair)), dtype=np.byte),
    dims=('time1', 'time2', 'freq', 'pair'),
    coords=(time, time, corr.freq, corr.pair),
    attrs={
        'long_name': 'Crosscorrelation status',
        'standard_name': 'crosscorrelation_status',
        'units': '-',
    },
)

corr['cc'] = corr.status.astype(np.float64) * 0
corr['cc'].attrs = {
    'long_name': 'Crosscorrelation Estimate',
    'standard_name': 'crosscorrelation_estimate',
    'units': '-',
    'add_offset': np.float64(0.),
    'scale_factor': np.float64(1.),
    'valid_range': np.float64([-1., 1.]),
    'normalize': np.byte(1),
}

corr['delta_freq'] = corr.status.astype(np.float64) * 0
corr['delta_freq'].attrs = {
    'long_name': 'Delta Frequency',
    'standard_name': 'delta_frequency',
    'units': 's-1',
}

corr['delta_lag'] = corr.status.astype(np.float64) * 0
corr['delta_lag'].attrs = {
    'long_name': 'Delta Lag',
    'standard_name': 'delta_lag',
    'units': 's',
}

# ignore upper diagonal
mask_upper_triangle(corr)

# piecewise chunk dataset
corr = corr.chunk({'pair': 1, 'time1': 1, 'time2': 1})


# -----------------------------------------------------------------------------
# Init Dask
# -----------------------------------------------------------------------------
dcluster = distributed.LocalCluster(
    processes=False, threads_per_worker=1, n_workers=n_workers,
    dashboard_address=':8787'
)
dclient = distributed.Client(dcluster)

# -----------------------------------------------------------------------------
# Process
# -----------------------------------------------------------------------------
mapped = xr.map_blocks(
    correlate_spectrograms, corr, kwargs={'root': os.path.join(root, 'cc')}
)

result = mapped.load()
fill_upper_triangle(result)

# to netcdf
# todo: store snr confidence trigger parameters!!
result.to_netcdf('psdcorr_{}_{}_{}.nc'.format(
    pair,
    result.time[0].dt.strftime('%Y%j'),
    result.time[-1].dt.strftime('%Y%j')
))

# plot cc
plt.figure()
result.cc.isel(pair=1, freq=0).plot()
plt.gca().invert_yaxis()
plt.show()

# plot delta_lag
plt.figure()
result.delta_lag.isel(pair=1, freq=0).plot()
plt.gca().invert_yaxis()
plt.show()

# plot delta_freq
plt.figure()
result.delta_freq.isel(pair=1, freq=0).plot()
plt.gca().invert_yaxis()
plt.show()

# -----------------------------------------------------------------------------
# Close Dask
# -----------------------------------------------------------------------------
dclient.close()
dcluster.close()
