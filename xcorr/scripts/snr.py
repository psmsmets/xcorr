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
def estimate_snr_for_day(pair_str, day, root, attrs, verbose=False):
    """
    """
    if verbose:
        print('.'*4, f'Load {pair_str} {day}')
    try:
        ds = xcorr.mfread(
            xcorr.util.ncfile(pair_str, day, root, verify_receiver=False),
            fast=True
        )
    except Exception as e:
        print(f'Error @ load {pair_str} {day}:', e)
        return

    snr = []
    for pair in ds.pair:
        if verbose:
            print('.'*6, str(pair.values))
        try:
            cc = ds.cc.sel(pair=pair, drop=False).where(
                (ds.status.sel(pair=pair) == 1),
                drop=True
            ).load()
            if xr.ufuncs.isnan(cc).any():
                continue
            pair_offset = ds.pair_offset.sel(pair=pair)
            time_offset = ds.time_offset.sel(pair=pair)
            delay = -(pair_offset + time_offset) / pd.Timedelta('1s')
            cc = xcorr.signal.unbias(cc)
            cc = xcorr.signal.demean(cc)
            cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase-wrap
            cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
            cc = xcorr.signal.filter(cc, frequency=3., btype='highpass',
                                     order=2)
            cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts
        except Exception as e:
            print('Error @ process cc:', e)
            continue
        try:
            d = ds.distance.sel(pair=pair, drop=False).load()
            s = (cc.lag >= d/1.50) & (cc.lag <= d/1.46)
            n = (cc.lag >= 6*3600) & (cc.lag <= 9*3600)
            sn = xcorr.signal.snr(cc, s, n, dim='lag',
                                  extend=True, envelope=True, **attrs)
        except Exception as e:
            print('Error @ estimate snr:', e)
            continue
        else:
            if sn is not None:
                snr.append(sn)
    snr =  xr.concat(
        snr, dim='pair', combine_attrs='override'
    ) if snr else None
    return snr


def delayed_snr_estimate(pair, start, end, root, attrs, verbose=False):
    """Estimate snr for a time period
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        results.append(estimate_snr_for_day(pair, day, root, attrs, verbose))
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
                             args.attrs, verbose=args.debug)
    )
    distributed.wait(mapped)

    print('.. gather signal-to-noise results')
    snr = client.gather(mapped)

    print('.. merge signal-to-noise results')
    snr = xr.merge(list(filter(None, snr)), combine_attrs='override')
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
