"""
SNR
===

Signal-to-noise ratio of a multi-file dataset using dask.

"""
import xarray as xr
import dask
from dask import distributed
from shutil import rmtree
import os
import sys
import getopt
import xcorr


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
    cc = ds.cc.where(valid, drop=True)
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


def help(e=None):
    """
    """
    print('snr.py -p <pair> -y <year> -r <root> -n <nthreads>')
    raise SystemExit(e)


def main(argv):
    """
    """
    # init args
    root = None
    pair = None
    year = None
    nthreads = None

    try:
        opts, args = getopt.getopt(
            argv,
            'hp:y:r:d:n:',
            ['help', 'pair=', 'year=', 'root=', 'nthreads=']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
        elif opt in ('-p', '--pair'):
            pair = arg
        elif opt in ('-y', '--year'):
            year = int(arg)
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('-n', '--nthreads'):
            nthreads = int(arg)

    assert isinstance(year, int) and year >= 1900 and year <= 2100, (
        'Year should be of type integer between 1900 and 2100.'
    )
    pair = pair or '*'
    year = f'{year}'
    doy = '*'

    # check root and dest
    root = os.path.abspath(root) if root is not None else os.getcwd()

    # dask client
    dcluster = distributed.LocalCluster(
        processes=False, threads_per_worker=1, n_workers=nthreads or 2,
    )
    dclient = distributed.Client(dcluster)

    print('Dask client:', dclient)
    print('Dask dashboard:', dclient.dashboard_link)

    # verbose
    print('{:>25} : {}'.format('root', root))
    print('{:>25} : {}'.format('pair', pair))
    print('{:>25} : {}'.format('year', year))
    print('{:>25} : {}'.format('doy', doy))
    print('{:>25} : {}'.format('nthreads', nthreads))

    # list of files using dask
    validated = xcorr.core.validate_list(
        os.path.join(root, 'cc', year, pair, f'{pair}.{year}.{doy}.nc'),
        fast=True,
        paths_only=True,
        keep_opened=False,
    )
    assert validated, 'No data found!'

    # snr
    sn = dask.compute(lazy_snr_list(validated))
    ds = xr.merge(sn[0])

    # save
    xcorr.write(ds.snr, os.path.join(root, 'snr', f'snr_{year}.nc'), verb=0)

    # cleanup
    dclient.close()
    dcluster.close()
    rmtree('dask-worker-space', ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
