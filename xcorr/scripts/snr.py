"""
SNR
===

Signal-to-noise ratio estimation of cross-crorrelations.

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
from .helpers import (init_dask, ncfile, add_common_arguments,
                      add_attrs_group, parse_attrs_group)

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(pair, time, root):
    """
    """
    try:
        ds = xcorr.mfread(
            xcorr.util.ncfile(pair, time, root, verify_receiver=False),
            fast=True
        )
    except Exception as e:
        print(f'Error @ load {pair} {time}:', e)
        return
    else:
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
    except Exception as e:
        print('Error @ process:', e)
    else:
        return cc


@dask.delayed
def estimate_snr(ds, cc, attrs):
    """
    """
    if ds is None or cc is None:
        return
    try:
        # s = xcorr.signal.multi_mask(
        #     x=ds.lag,
        #     y=ds.distance,
        #     lower=1.46,
        #     upper=1.50,
        #     invert=True,
        # )
        # n = xcorr.signal.mask(
        #     x=ds.lag,
        #     lower=3.,
        #     upper=9.,
        #     scalar=3600.,
        # )
        s = (ds.lag >= ds.distance/1.50) & (ds.lag <= ds.distance/1.46)
        n = (ds.lag >= 6*3600) & (ds.lag <= 9*3600)
        sn = xcorr.signal.snr(cc, signal=s, noise=n, dim='lag',
                              extend=True, envelope=True, **attrs)
    except Exception as e:
        print('Error @ estimate_snr:', e)
        return
    else:
        return sn


def delayed_snr_estimate(pair, start, end, root, attrs):
    """Estimate snr for a time period
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        ds = load(pair, day, root)
        cc = process(ds)
        sn = estimate_snr(ds, cc, attrs)
        results.append(sn)
    return results


###############################################################################
# Main functions
# --------------
def main():
    """Main script function.
    """
    # arguments
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

    add_common_arguments(parser)
    add_attrs_group(parser)

    args = parser.parse_args()

    # update arguments
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start, format=args.format)
    args.end = pd.to_datetime(args.end, format=args.format)
    args.out = ncfile('snr', args.pair, args.start, args.end)
    args.attrs = parse_attrs_group(args)

    # print header and core parameters
    print(f'xcorr-snr v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', args.pair))
    print('{:>20} : {}'.format('start', args.start.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('end', args.end.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('outfile', args.out))
    print('{:>20} : {}'.format('overwrite', args.overwrite))

    # check if output file exists
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f'Output file "{args.out}" already exists'
                              ' and overwrite is False.')

    # init dask cluster and client
    cluster, client = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # estimate snr
    print('.. estimate signal-to-noise per day for period')
    mapped = client.compute(
        delayed_snr_estimate(args.pair, args.start, args.end, args.root,
                             args.attrs)
    )
    distributed.wait(mapped)

    print('.. gather signal-to-noise results')
    snr = client.gather(mapped)

    print('.. merge signal-to-noise results')
    print(list(filter(None, snr)))
    snr = xr.merge(
        objects=list(filter(None, snr)),
    )
    if args.debug:
        print(snr)

    # to netcdf
    print(f'.. write to "{args.out}"')
    xcorr.write(snr, args.out, variable_encoding=dict(zlib=True, complevel=9),
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
