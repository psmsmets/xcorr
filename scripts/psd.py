"""
PSD
===

Spectrograms of triggered datasets by snr using dask.

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
from dask import distributed
from shutil import rmtree
from glob import glob
import os
import sys
import getopt
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
    src = os.path.join(root, 'cc')

    files = []
    for t in pd.date_range(t0, t1, freq='1D'):
        files.append(xcorr.util.ncfile(str(pair.values), t, src))

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
    path = os.path.join(root, 'psd', pair, filename)

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
            ds = _combine(ds, cc, psd, snr_period)
            fname = _write(ds, period, root)
            fnames.append(fname)
    return fnames


###############################################################################
# Main functions
# --------------

def help(e=None):
    """
    """
    print('psd.py -p <pair> -y <year> -r <root> -n <nthreads>')
    raise SystemExit(e)


def main(argv):
    """
    """
    # init args
    root = None
    starttime = None
    endtime = None
    n_workers = None
    plot = False

    try:
        opts, args = getopt.getopt(
            argv,
            'hp:s:e:r:n:',
            ['help', 'starttime=', 'endtime=', 'root=', 'nworkers=', 'plot']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
        elif opt in ('-s', '--starttime'):
            starttime = arg
        elif opt in ('-e', '--endtime'):
            endtime = arg
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('-n', '--nworkers'):
            n_workers = int(arg)
        elif opt in ('--plot'):
            plot = True

    starttime = pd.to_datetime(starttime or '2015-01-05')
    endtime = pd.to_datetime(endtime or '2015-02-15')
    n_workers = n_workers or 1

    # check root
    root = os.path.abspath(root) if root is not None else os.getcwd()

    # dask client
    dcluster = distributed.LocalCluster(
        processes=False, threads_per_worker=1, n_workers=n_workers,
    )
    dclient = distributed.Client(dcluster)

    print('Dask client:', dclient)
    print('Dask dashboard:', dclient.dashboard_link)

    # verbose
    print('{:>25} : {}'.format('root', root))
    print('{:>25} : {}'.format('starttime', starttime))
    print('{:>25} : {}'.format('endtime', endtime))
    print('{:>25} : {}'.format('nworkers', n_workers))

    # get snr
    snr = xr.merge([
        xr.open_dataarray(f) for f in glob(f'{root}/snr/snr_201[4-9].nc')
    ]).snr
    snr = snr.where(
        (snr.time >= starttime.to_datetime64()) &
        (snr.time <= endtime.to_datetime64())
    )

    # get triggered periods
    ct = xcorr.signal.coincidence_trigger(
        snr, thr_on=8., thr_off=5., extend=0, thr_coincidence_sum=None
    )

    # action?
    if plot:
        # plot
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=4, add_legend=False)
        xcorr.signal.trigger.plot_trigs(snr, ct)
        plt.ylim(0, 200)
        plt.tight_layout()
        plt.show()
    else:
        # construct datasets with preprocessed cc, snr and psd
        files = lazy_spectrogram(snr, ct, root)
        files = dask.compute(files)[0]
        print(files)

    # Cleanup
    dclient.close()
    dcluster.close()
    rmtree('dask-worker-space', ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
