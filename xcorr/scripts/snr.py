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
from glob import glob
import sys
import argparse

# Relative imports
import xcorr
from . import utils

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def estimate_snr_for_day(
    pairs, day, root, envelope, attrs, debug=False
):
    """
    """
    if debug:
        print('.'*4, f'Load pairs for {day}')
    snr = []
    for pair in pairs:
        if debug:
            print('.'*6, str(pair.values))
        try:
            ds = xcorr.read(xcorr.io.ncfile(pair, day, root), fast=True)
            if ds is None:
                continue
            cc = ds.cc.where((ds.status == 1), drop=True)
            if xr.ufuncs.isnan(cc).any():
                continue
            delay = -(ds.pair_offset + ds.time_offset)
            cc = (cc
                  .signal.unbias()
                  .signal.demean()
                  .signal.taper(max_length=5.)  # timeshift phase-wrap
                  .signal.timeshift(delay=delay, dim='lag', fast=True)
                  .signal.filter(frequency=3., btype='highpass', order=2)
                  .signal.taper(max_length=3/2)  # filter artefacts
                  )
        except Exception as e:
            print(f'Error @ process cc {str(pair.values)} {day}:', e)
            continue
        try:
            s = (cc.lag >= ds.distance/1.50) & (cc.lag <= ds.distance/1.46)
            n = (cc.lag >= 6*3600) & (cc.lag <= 9*3600)
            sn = cc.signal.snr(
                s, n,
                dim='lag',
                extend=True,
                envelope=envelope,
                **attrs
            )
        except Exception as e:
            print(f'Error @ estimate snr {day}:', e)
            continue
        else:
            if sn is not None:
                snr.append(sn)
    if ds is not None:
        ds.close()
    snr = xr.concat(
        snr, dim='pair', combine_attrs='override',
    ) if snr else None
    return snr


def delayed_snr_estimate(
    pairs, start, end, root, envelope, attrs, debug=False
):
    """Estimate snr for a time period
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        results.append(
            estimate_snr_for_day(pairs, day, root, envelope, attrs, debug)
        )
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
        description='Signal-to-noise ratio estimation of cross-crorrelations.',
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
        '-e', '--envelope', action="store_true", default=False,
        help=('Calculate the amplitude envelope of the signal part before '
              'locating the peak amplitude (default: `False`)')
    )
    parser.add_argument(
        '-p', '--pair', metavar='..', type=str, default='*',
        help='Filter pairs that contain the given glob string'
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set cross-correlation root directory (default: current '
              'working directory)')
    )
    utils.add_common_arguments(parser)
    utils.add_attrs_group(parser)

    # parse arguments
    args = parser.parse_args()
    args.attrs = utils.parse_attrs_group(args)

    # get pair list
    args.pairs = list(set([
        os.path.split(p)[-1]
        for p in glob(os.path.join(args.root, '*', args.pair))
    ]))

    # update some arguments
    args.start = pd.to_datetime(args.start, format=args.format)
    args.end = pd.to_datetime(args.end, format=args.format)
    args.out = utils.ncfile('snr_envelope' if args.envelope else 'snr',
                            args.pair, args.start, args.end,
                            args.prefix, args.suffix)

    # print header and core parameters
    print(f'xcorr-snr v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', args.pair + f" (#{len(args.pairs)})"))
    print('{:>20} : {}'.format('start', args.start.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('end', args.end.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('envelope', args.envelope))
    print('{:>20} : {}'.format('outfile', args.out))
    print('{:>20} : {}'.format('overwrite', args.overwrite))

    if args.debug:
        for i, p in enumerate(args.pairs):
            print('{:>20} : {}'.format('pairs', p) if i == 0 else ' '*22 + p)

    # check if output file exists
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f'Output file "{args.out}" already exists'
                              ' and overwrite is False.')

    # init dask cluster and client
    cluster, client = utils.init_dask(n_workers=args.nworkers,
                                      scheduler_file=args.scheduler)

    # estimate snr
    print('.. estimate signal-to-noise per day for period')
    mapped = client.compute(
        delayed_snr_estimate(args.pairs, args.start, args.end, args.root,
                             args.envelope, args.attrs, debug=args.debug)
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
