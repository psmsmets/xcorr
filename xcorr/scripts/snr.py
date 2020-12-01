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
import argparse

# Relative imports
import xcorr
from .helpers import init_dask, ncfile

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(pair, time, root):
    """
    """
    try:
        ds = xcorr.mfread(xcorr.util.ncfile(pair, time, root), fast=True)
    except Exception:
        ds = None
    return ds


@dask.delayed
def process(ds):
    """
    """
    if ds is None:
        return

    try:
        cc = ds.cc.where((ds.status == 1), drop=True)
        if xr.ufuncs.isnan(cc).any():
            return
        delay = -(ds.pair_offset + ds.time_offset) / pd.Timedelta('1s')
        cc = xcorr.signal.unbias(cc)
        cc = xcorr.signal.demean(cc)
        cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase-wrapping
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
        cc = xcorr.signal.filter(cc, frequency=3., btype='highpass', order=2)
        cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts

        cc = cc.compute()

    except Exception:
        cc = None

    return cc


@dask.delayed
def estimate_snr(ds, cc, **kwargs):
    """
    """
    if ds is None or cc is None:
        return
    try:
        signal = (ds.lag >= ds.distance/1.50) & (ds.lag <= ds.distance/1.46)
        noise = (ds.lag >= 6*3600) & (ds.lag <= 9*3600)

        snr = xcorr.signal.snr(cc, signal, noise, dim='lag',
                               extend=True, envelope=True, **kwargs)
    except Exception:
        snr = None

    return snr


def delayed_snr_estimate(pair, start, end, root, **kwargs):
    """Estimate snr for a time period
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        ds = load(pair, day, root)
        cc = process(ds)
        sn = estimate_snr(ds, cc, **kwargs)
        results.append(sn)
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
        help=('Set cross-correlation root directory (default: current '
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

    # init dask cluster and client
    cluster, client = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # estimate snr
    print('.. estimate signal-to-noise per day for period')
    mapped = client.compute(
        delayed_snr_estimate(args.pair, args.start, args.end, args.root)
    )
    distributed.wait(mapped)

    print('.. gather signal-to-noise results')
    snr = client.gather(mapped)

    print('.. merge signal-to-noise results')
    snr = xr.merge(list(filter(None, snr)))
    if args.debug:
        print(snr)

    # to netcdf
    nc = ncfile('snr', args.pair, args.start, args.end)
    print(f'.. write to "{nc}"')
    xcorr.write(snr, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot
    if args.plot:
        print('.. plot')
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        plt.tight_layout()
        plt.show()

    # close dask client and cluster
    print('.. close dask')
    client.close()
    if cluster is not None:
        cluster.close()

    print('.. done')
    sys.exit(0)


if __name__ == "__main__":
    main()
