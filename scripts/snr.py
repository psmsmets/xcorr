"""
SNR
===

Signal-to-noise ratio of a multi-file dataset using dask.

"""

# Mandatory imports
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
from dask import distributed
import os
import sys
import getopt

# Relative imports
import xcorr


__all__ = []
__name__ = 'xcorr-snr'


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
    cc = ds.cc.where((valid) & (ds.status == 1), drop=True)
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
# Main functions
# --------------

def help(e=None):
    """
    """
    _help = """
    xcorr-snr [option] ... [arg] ...
    Options and arguments:
        --debug      : Maximize verbosity.
    -e, --endtime=   : Set endtime, e.g., yyyy-mm-dd.
    -h, --help       : Print this help message and exit.
    -n, --nworkers=  : Set number of dask workers for local client. If a
                       a scheduler set the client will wait until the number
                       of workers is available.
    -p, --pair=      : Filter pairs given a glob string. If empty all pairs
                       are used.
    -r, --root=      : Set root. Defaults to current working directory.
    -s, --starttime= : Set starttime, e.g., yyyy-mm-dd
        --scheduler= : Connect to a dask scheduler by a scheduler-file.
        --plot       : Generate some plots during processing (stalls).
    -v, --version    : Print version number and exit.
    """
    print('\n'.join([line[4:] for line in _help.splitlines()]))
    raise SystemExit(e)


def main():
    """
    """
    # init args
    pair, starttime, endtime = None, None, None
    root, n_workers, scheduler = None, None, None
    plot, debug = False, False

    # get args
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
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

    # list of files using dask
    validated = xcorr.core.validate_list(
        [xcorr.util.ncfile(pair, time, root, verify_receiver=False)
         for time in pd.date_range(starttime, endtime)],
        fast=True,
        paths_only=True,
        keep_opened=False,
    )
    assert validated, 'No data found!'

    # snr
    snr = client.map(lazy_snr_list, validated)
    snr = xr.merge(client.gather(snr))

    # to netcdf
    nc = os.path.join(root, 'snr', 'snr_{}_{}_{}.nc'.format(
        'all' if pair == '*' else pair.translate({ord(c): None for c in '*?'}),
        starttime.strftime('%Y%j'),
        endtime.strftime('%Y%j'),
    ))
    print(f'.. write to "{nc}"')
    xcorr.write(
        data=snr,
        path=nc,
        variable_encoding=dict(zlib=True, complevel=9),
        verb=1 if debug else 0,
    )

    # plot
    if plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        plt.tight_layout()
        plt.show()

    # close
    print('.. close connections')
    client.close()
    if cluster is not None:
        cluster.close()

    print('.. done')


if __name__ == "__main__":
    main()
