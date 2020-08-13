"""
TIMELAPSE
===

Crosscorrelation of crosscorrelation spectrograms.

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from dask import distributed
from shutil import rmtree
from glob import glob
import os
import sys
import getopt
import xcorr


###############################################################################
# Local functions
# -----------------

def get_spectrogram(pair, time, root: str = None):
    """Load spectrogram for a pair and time.
    """
    # construct abs path and filename
    nc = xcorr.util.ncfile(pair, time, root)

    # select pair time location
    sel = {'pair': pair, 'time': time}

    # set lock
    lock = distributed.Lock(nc)
    lock.acquire()

    # read file
    ds = xcorr.read(nc, quick_and_dirty=True)

    # extract a single cc
    cc = ds.cc.loc[sel]

    # get pair relative distance
    d_km = ds.distance.loc[{'pair': pair}].values

    # extract cc for velocity range (km/s)
    cc = cc.where((cc.lag >= d_km/1.495) & (cc.lag <= d_km/1.465), drop=True)

    # extract lag offset
    delay = xcorr.util.time.to_seconds(
        (ds.pair_offset.loc[sel] + ds.time_offset.loc[sel]).values
    )

    # release lock
    ds.close()
    lock.release()

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.detrend(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    # solve time_offset and pair_offset
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
        in1 = psd1.where((psd1.freq >= fmin) & (psd1.freq < fmax), drop=True)
        in2 = psd2.where((psd2.freq >= fmin) & (psd2.freq < fmax), drop=True)

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
        ds.status.loc[{
            'time1': ds.time1[ind1[i]],
            'time2': ds.time2[ind2[i]],
        }] = np.byte(1)


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


def init_timelapse(snr, ct, pair, starttime, endtime, freq, root):
    """Init a timelapse dataset.
    """
    # extract times with activity
    time = ct.time.where(ct >= 0, drop=True)

    # new dataset
    ds = xr.Dataset()

    # set global attributes
    nc = xcorr.util.ncfile(snr.pair[0], time[0], os.path.join(root, 'cc'))
    cc = xcorr.read(nc, quick_and_dirty=True)
    ds.attrs = cc.attrs
    cc.close()
    ds.attrs['xcorr_version'] = xcorr.__version__
    ds.attrs['dependencies_version'] = xcorr.core.core.dependencies_version()

    # set coordinates
    ds['pair'] = snr.pair
    ds.pair.attrs = snr.pair.attrs

    ds['freq'] = xr.DataArray(
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
    ds['freq_bw'] = xr.DataArray(
        np.diff(freq, axis=1).squeeze(),
        dims=('freq'),
        coords=(ds.freq,),
        attrs={
            'long_name': 'Frequency bandwidth',
            'standard_name': 'frequency_bandwidth',
            'units': 's-1',
        },
    )

    ds['status'] = xr.DataArray(
        np.zeros((len(ds.pair), len(freq), time.size, time.size),
                 dtype=np.byte),
        dims=('pair', 'freq', 'time1', 'time2'),
        coords=(ds.pair, ds.freq, time, time),
        attrs={
            'long_name': 'Crosscorrelation status',
            'standard_name': 'crosscorrelation_status',
            'units': '-',
        },
    )

    ds['snr'] = snr
    ds['ct'] = ct

    ds['cc'] = ds.status.astype(np.float64) * 0
    ds['cc'].attrs = {
        'long_name': 'Crosscorrelation Estimate',
        'standard_name': 'crosscorrelation_estimate',
        'units': '-',
        'add_offset': np.float64(0.),
        'scale_factor': np.float64(1.),
        'valid_range': np.float64([-1., 1.]),
        'normalize': np.byte(1),
    }

    ds['delta_freq'] = ds.status.astype(np.float64) * 0
    ds['delta_freq'].attrs = {
        'long_name': 'Delta Frequency',
        'standard_name': 'delta_frequency',
        'units': 's-1',
    }

    ds['delta_lag'] = ds.status.astype(np.float64) * 0
    ds['delta_lag'].attrs = {
        'long_name': 'Delta Lag',
        'standard_name': 'delta_lag',
        'units': 's',
    }

    # ignore upper diagonal
    mask_upper_triangle(ds)

    # piecewise chunk dataset
    ds = ds.chunk({'pair': 1, 'time1': 1, 'time2': 1})

    return ds


def create_locks(ds, root):
    """Initate distributed cc file access locking.
    """
    locks = []
    for pair in ds.pair:
        for time in ds.time1:
            nc = xcorr.util.ncfile(pair, time, root)
            if nc not in locks:
                locks.append(nc)
    locks = [distributed.Lock(nc) for nc in locks]
    return locks


###############################################################################
# Main functions
# --------------

def help(e=None):
    """
    """
    print('timelapse.py -p <pair> -s <starttime>  -e <endtime> -r <root> '
          '-n <nworkers>')
    raise SystemExit(e)


def main(argv):
    """
    """
    # init args
    root = None
    pair = None
    starttime = None
    endtime = None
    freq = None
    n_workers = None
    plot = False
    debug = False

    try:
        opts, args = getopt.getopt(
            argv,
            'hp:s:e:f:r:n:',
            ['help', 'pair=', 'starttime=', 'endtime=', 'frequency=', 'root=',
             'nworkers=', 'plot', 'debug']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
        elif opt in ('-p', '--pair'):
            pair = arg
        elif opt in ('-s', '--starttime'):
            starttime = arg
        elif opt in ('-e', '--endtime'):
            endtime = arg
        elif opt in ('-f', '--frequency'):
            freq = np.array(eval(arg))
            if len(freq.shape) != 2 or freq.shape[-1] != 2:
                raise ValueError('frequency should be a list of tuple-pairs '
                                 'with start and end frequencies. Example: '
                                 '--frequencies="(3., 6.), (6., 12.)"')
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('-n', '--nworkers'):
            n_workers = int(arg)
        elif opt in ('--plot'):
            plot = True
        elif opt in ('--debug'):
            debug = True

    pair = pair or ''
    freq = np.array(((3., 6.), (6., 12.))) if freq is None else freq
    starttime = pd.to_datetime(starttime or '2015-01-15')
    endtime = pd.to_datetime(endtime or '2015-01-18')
    root = os.path.abspath(root) if root is not None else os.getcwd()
    n_workers = n_workers or 1

    # dask client
    dcluster = distributed.LocalCluster(
        processes=False, threads_per_worker=1, n_workers=n_workers,
    )
    dclient = distributed.Client(dcluster)

    print('Dask client:', dclient)
    print('Dask dashboard:', dclient.dashboard_link)

    # verbose
    print('{:>25} : {}'.format('root', root))
    print('{:>25} : {}'.format('pair', pair))
    print('{:>25} : {}'.format('starttime', starttime))
    print('{:>25} : {}'.format('endtime', endtime))

    # snr
    snr = xr.merge([xr.open_dataarray(f) for f in
                    glob(os.path.join(root, 'snr', 'snr_20??.nc'))]).snr
    snr = snr.where(
        (
            (snr.time >= starttime.to_datetime64()) &
            (snr.time < endtime.to_datetime64()) &
            (snr.pair.str.contains(pair))
        ),
        drop=True,
    )
    if debug:
        print(snr)

    # get confindence triggers
    ct = xcorr.signal.coincidence_trigger(
        snr, thr_on=10., extend=0, thr_coincidence_sum=None,
    )
    if debug:
        print(ct)
    if plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        xcorr.signal.trigger.plot_trigs(snr, ct)  
        plt.tight_layout()
        plt.show()

    # init timelapse
    ds = init_timelapse(snr, ct, pair, starttime, endtime, freq, root)
    if debug:
        print(ds)

    # create all locks
    locks = create_locks(ds, os.path.join(root, 'cc'))

    # map nodes
    mapped = xr.map_blocks(
        correlate_spectrograms, ds, kwargs={'root': os.path.join(root, 'cc')}
    )

    # load results
    result = mapped.load()
    fill_upper_triangle(result)

    if debug:
        print(result)

    # to netcdf (check if all parameters are logged!)
    nc = 'timelapse_{}_{}_{}.nc'.format(
        pair,
        str(result.time1[0].dt.strftime('%Y%j').values),
        str(result.time1[-1].dt.strftime('%Y%j').values),
    )
    xcorr.write(result, nc)

    # Plot?
    if plot:
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

    # Cleanup
    dclient.close()
    dcluster.close()
    rmtree('dask-worker-space', ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
