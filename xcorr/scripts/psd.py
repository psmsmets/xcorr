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
from glob import glob
import os
import sys
import getopt
import xcorr


__all__ = []


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

    # copy to avoid dimension blow-up
    distance = ds.distance
    pair_offset = ds.pair_offset
    time_offset = ds.time_offset

    # drop and mask
    ds = ds.drop_vars(
        ['status', 'distance', 'pair_offset', 'time_offset']
    ).where(mask, drop=True)

    # add back
    ds['distance'] = distance
    ds['pair_offset'] = pair_offset
    ds['time_offset'] = time_offset

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

    delay = -xcorr.util.time.to_seconds(ds.pair_offset + ds.time_offset)
    if (delay != 0.).any():
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag')
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
    _help = """
    xcorr-psd [option] ... [arg] ...
    Options and arguments:
        --debug      : Maximize verbosity.
    -e, --endtime=   : Set endtime, e.g., yyyy-mm-dd.
    -h, --help       : Print this help message and exit.
    -n, --nworkers=  : Set number of dask workers for local client. If a
                       a scheduler set the client will wait until the number
                       of workers is available.
    -p, --pair=      : Filter pair that contain the given string. If empty all
                       pairs are used.
    -r, --root=      : Set root. Defaults to current working directory.
    -s, --starttime= : Set starttime, e.g., yyyy-mm-dd
        --scheduler= : Connect to a dask scheduler by a scheduler-file.
        --plot       : Generate some plots during processing (stalls).
    -v, --version    : Print version number and exit.
    """
    print('\n'.join([line[4:] for line in _help.splitlines()]))
    raise SystemExit(e)


def main(argv):
    """
    """
    # init args
    pair, starttime, endtime = None, None, None
    root, n_workers, scheduler = None, None, None
    plot, debug = False, False

    # get args
    try:
        opts, args = getopt.getopt(
            argv,
            'hvp:s:e:f:r:n:c:',
            ['pair=', 'starttime=', 'endtime=', 'root=',
             'nworkers=', 'help', 'plot', 'debug', 'scheduler=']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
        elif opt in ('--version'):
            print(xcorr.__version__)
            raise SystemExit()
        elif opt in ('-p', '--pair'):
            pair = arg
        elif opt in ('-s', '--starttime'):
            starttime = pd.to_datetime(arg)
        elif opt in ('-e', '--endtime'):
            endtime = pd.to_datetime(arg)
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('-n', '--nworkers'):
            n_workers = int(arg)
        elif opt in ('--plot'):
            plot = True
        elif opt in ('--debug'):
            debug = True
        elif opt in ('--scheduler'):
            scheduler = arg

    # optional
    pair = pair or '*'
    root = os.path.abspath(root) if root is not None else os.getcwd()

    # obligatory
    if starttime is None or endtime is None:
        raise RuntimeError('Both --startime and --endtime should be set.')

    # dask client
    if scheduler is not None:
        print('Dask Scheduler:', scheduler)
        client = distributed.Client(scheduler_file=scheduler)
        cluster = None
        if n_workers:
            print(f'.. waiting for {n_workers} workers', end=' ')
            client.wait_for_workers(n_workers=n_workers)
            print('OK.')
    else:
        cluster = distributed.LocalCluster(
            processes=False, threads_per_worker=1, n_workers=n_workers or 4,
        )
        print('Dask LocalCluster:', cluster)
        client = distributed.Client(cluster)
        print('Dask Client:', client)

    # parameters
    print('{:>25} : {}'.format('root', root))
    print('{:>25} : {}'.format('pair', pair))
    print('{:>25} : {}'.format('starttime', starttime))
    print('{:>25} : {}'.format('endtime', endtime))

    # pair wildcard
    pair = '' if pair == '*' else pair

    # snr
    print('.. load signal-to-noise ratio')
    snr = xr.merge([xr.open_dataarray(f) for f in
                    glob(os.path.join(root, 'snr', 'snr_*.nc'))]).snr
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
    print('.. get coincidence triggers', end=', ')
    ct = xcorr.signal.coincidence_trigger(
        snr, thr_on=10., extend=0, thr_coincidence_sum=None,
    )
    print(f'periods = {ct.attrs["nperiods"]}')
    if debug:
        print(ct)
    if plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        xcorr.signal.trigger.plot_trigs(snr, ct)
        plt.tight_layout()
        plt.show()

    # construct datasets with preprocessed cc, snr and psd
    print('.. construct files per active period')
    files = lazy_spectrogram(snr, ct, root)
    files = client.compute(files)[0]
    if debug:
        print(files)

    # close
    print('.. close connections')
    client.close()
    if cluster is not None:
        cluster.close()

    print('.. done')


if __name__ == "__main__":
    main(sys.argv[1:])
