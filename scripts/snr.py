"""
SNR
===

Signal-to-noise ratio estimation of crosscrorrelations.

"""

# Mandatory imports
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
import distributed
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

@dask.delayed
def _open(filename):
    """
    """
    ds = xcorr.read(filename, fast=True)
    ds.load()
    ds.close()
    return ds


@dask.delayed
def _mask_valid(ds):
    """
    """
    mask = xcorr.signal.mask(
        x=ds.lag,
        upper=9./24.,
        scalar=ds.time.window_length
    )
    return mask


@dask.delayed
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


@dask.delayed
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


@dask.delayed
def _select_and_trim(ds, valid):
    """
    """
    cc = ds.cc.where((valid) & (ds.status == 1), drop=True)
    return cc


@dask.delayed
def _unbias(cc):
    """
    """
    cc = xcorr.signal.unbias(cc)
    return cc


@dask.delayed
def _filter(cc):
    """
    """
    cc = xcorr.signal.filter(cc, frequency=3., btype='highpass', order=2)
    return cc


@dask.delayed
def _demean(cc):
    """
    """
    cc = xcorr.signal.demean(cc)
    return cc


@dask.delayed
def _taper(cc):
    cc = xcorr.signal.taper(cc, max_length=2/3.)
    return cc


@dask.delayed
def _snr(cc, signal, noise):
    """
    """
    snr = xcorr.signal.snr(cc, signal, noise)
    return snr


def snr_of_filenames(filenames: list):
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
        snr_list.append(snr)
    return snr_list


###############################################################################
# Main functions
# --------------

def help(e=None):
    """
    """
    _help = """
    Signal-to-noise ratio estimation of crosscrorrelations.

    Usage: xcorr-snr <start> <end> [option] ... [arg] ...
    <start> <end>    : Start and end datetime given format yyyy-mm-dd.
    Options and arguments:
        --debug      : Maximize verbosity.
    -h, --help       : Print this help message and exit.
    -n, --nworkers=  : Set number of dask workers for local client. If a
                       a scheduler set the client will wait until the number
                       of workers is available.
    -p, --pair=      : Filter pair that contain the given string. If empty all
                       pairs are used.
        --plot       : Generate some plots during processing (stalls).
    -r, --root=      : Set root. Defaults to current working directory.
                       Crosscorrelations should be in "{root}/cc" following
                       xcorr folder structure. SNR results are stored in
                       "{root}/snr".
        --scheduler= : Connect to a dask scheduler by a scheduler-file.
    -v, --version    : Print xcorr version number and exit."""

    print('\n'.join([line[4:] for line in _help.splitlines()]))
    raise SystemExit(e)


def main():
    """
    """

    # help?
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        help()

    # version?
    if '-v' in sys.argv[1:] or '--version' in sys.argv[1:]:
        print(xcorr.__version__)
        raise SystemExit(0)

    # start and end datetime
    if len(sys.argv) < 3:
        print('Both start and end datetime should be set!')
        raise SystemExit(1)
    starttime = pd.to_datetime(sys.argv[1])
    endtime = pd.to_datetime(sys.argv[2])

    # optional args
    pair, root, n_workers, scheduler = None, None, None, None
    plot, debug = False, False

    try:
        opts, args = getopt.getopt(
            sys.argv[3:],
            'c:f:n:p:r:',
            ['debug', 'nworkers=', 'pair=', 'plot', 'root=', 'scheduler=']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('--debug'):
            debug = True
        elif opt in ('-n', '--nworkers'):
            n_workers = int(arg)
        elif opt in ('-p', '--pair'):
            pair = '' if arg == '*' else arg
        elif opt in ('--plot'):
            plot = True
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('--scheduler'):
            scheduler = arg

    pair = pair or '*'
    root = os.path.abspath(root) if root is not None else os.getcwd()

    # print header and core parameters
    print(f'xcorr-timelapse v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', root))
    print('{:>20} : {}'.format('pair', pair))
    print('{:>20} : {}'.format('starttime', starttime))
    print('{:>20} : {}'.format('endtime', endtime))

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

    # list of files using dask
    validated = xcorr.core.validate_list(
        [xcorr.util.ncfile(pair, time, os.path.join(root, 'cc'),
                           verify_receiver=False)
         for time in pd.date_range(starttime, endtime)],
        fast=True,
        paths_only=True,
        keep_opened=False,
    )
    assert validated, 'No data found!'

    # snr
    mapped = client.compute(snr_of_filenames(validated))
    distributed.wait(mapped)
    snr = xr.merge(client.gather(mapped))

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
