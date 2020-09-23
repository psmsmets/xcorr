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
import argparse

# Relative imports
import xcorr
from .helpers import init_dask, close_dask, ncfile

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(filename):
    """
    """
    ds = xcorr.read(filename, fast=True)
    ds.load()
    ds.close()
    return ds


@dask.delayed
def process(ds):
    """
    """
    valid = xcorr.signal.mask(
        x=ds.lag,
        upper=9./24.,
        scalar=ds.time.window_length
    )

    signal = xcorr.signal.multi_mask(
        x=ds.lag,
        y=ds.distance,
        lower=1.45,
        upper=1.50,
        invert=True,
    )

    noise = xcorr.signal.mask(
        x=ds.lag,
        lower=6./24.,
        upper=9./24.,
        scalar=ds.time.window_length
    )

    cc = ds.cc.where((valid) & (ds.status == 1), drop=True)
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.filter(cc, frequency=3., btype='highpass', order=2)
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    snr = xcorr.signal.snr(cc, signal, noise)

    return snr


def snr_per_file(filenames: list):
    """Evaluate snr for a list of filenames
    """
    results = []
    for f in filenames:
        ds = load(f)
        snr = process(ds)
        results.append(snr)
    return results


###############################################################################
# Main functions
# --------------
def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-snr',
        description='Signal-to-noise ratio estimation of crosscrorrelations.',
        epilog='See also xcorr-ct xcorr-timelapse xcorr-psd xcorr-beamform',
    )
    parser.add_argument(
        'start', metavar='start', type=str,
        help='Start date'
    )
    parser.add_argument(
        'end', metavar='end', type=str,
        help='End date'
    )
    parser.add_argument(
        '--format', metavar='..', type=str, default=None,
        help=('The strftime to parse start and end (default: "%%Y-%%m-%%d"). '
              'See strftime documentation for more information on choices: '
              'https://docs.python.org/3/library/datetime.html#strftime-and-'
              'strptime-behavior.')
    )
    parser.add_argument(
        '-p', '--pair', metavar='..', type=str, default='*',
        help='Filter pairs that contain the given string'
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set crosscorrelation root directory (default: current '
              'working directory)')
    )
    parser.add_argument(
        '-n', '--nworkers', metavar='..', type=int, default=None,
        help=('Set number of dask workers for local client. If a scheduler '
              'is set the client will wait until the number of workers is '
              'available.')
    )
    parser.add_argument(
        '--scheduler', metavar='..', type=str, default=None,
        help='Connect to a dask scheduler by a scheduler-file'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots during processing (stalls)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Maximize verbosity'
    )
    parser.add_argument(
        '--version', action='version', version=xcorr.__version__,
        help='Print xcorr version and exit'
    )
    args = parser.parse_args()
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start, format=args.format)
    args.end = pd.to_datetime(args.end, format=args.format)

    # print header and core parameters
    print(f'xcorr-snr v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', args.pair))
    print('{:>20} : {}'.format('start', args.start.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('end', args.end.strftime('%Y-%m-%d')))

    # init dask client
    client, cluster = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # list of files using dask
    print('.. validate cc filelist')
    validated = xcorr.core.validate_list(
        [xcorr.util.ncfile(args.pair, t, args.root, verify_receiver=False)
         for t in pd.date_range(args.start, args.end)],
        fast=True,
        paths_only=True,
        keep_opened=False,
    )
    if len(validated) == 0:
        raise RuntimeError('No data found!')

    # snr
    print('.. compute snr for filename list')
    mapped = client.compute(snr_per_file(validated))
    distributed.wait(mapped)

    print('.. merge snr list')
    snr = xr.merge(client.gather(mapped))

    # to netcdf
    nc = ncfile('snr', args.pair, args.start, args.end)
    print(f'.. write to "{nc}"')
    xcorr.write(snr, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot
    if args.plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        plt.tight_layout()
        plt.show()

    # close dask client and cluster
    close_dask(client, cluster)

    print('.. done')


if __name__ == "__main__":
    main()
