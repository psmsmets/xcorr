"""
Beamform
========

Least-squares beamforming of a plane wave traversing the array.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
import distributed
import argparse
import os
import sys
from obspy import read_inventory
from glob import glob

# Relative imports
import xcorr
from . import utils

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(pairs, time, root, qd=False):
    """
    """
    try:
        ds = xcorr.merge(
            *[xcorr.io.ncfile(pair, time, root) for pair in pairs],
            fast=True, quick_and_dirty=qd,
        )
    except Exception as e:
        print(f'Error @ loading pairs for {time}:', e)
        return
    return ds if ds is not None and ds.pair.size == len(pairs) else None


@dask.delayed
def process(ds):
    """
    """
    if ds is None:
        return
    try:
        cc = ds.cc.where(
            (ds.lag >= ds.distance.mean()/1.50) &
            (ds.lag <= ds.distance.mean()/1.46),
            drop=True,
        )
        if  np.isnan(cc).any():
            return
        delay = -(ds.pair_offset + ds.time_offset)
        cc = (cc
              .signal.unbias()
              .signal.demean()
              .signal.taper(max_length=5.)  # timeshift phase-wrapping
              .signal.timeshift(delay=delay, dim='lag', fast=True)
              .signal.filter(frequency=3., btype='highpass', order=2)
              .signal.taper(max_length=3/2)  # filter artefacts
              )
    except Exception as e:
        print('Error @ process:', e)
        return
    else:
        return cc


@dask.delayed
def lse_fit(cc, xy, envelope, attrs):
    """
    """
    if cc is None:
        return
    try:
        fit = cc.signal.plane_wave_estimate(
            x=xy.x, y=xy.y,
            dim='lag',
            dtype='float64',
            envelope=envelope,
            **attrs
        )
    except Exception as e:
        print('Error @ lse_fit:', e)
        return
    else:
        return fit


def delayed_plane_wave_fit(xy, start, end, root, qd, envelope, attrs):
    """Plane wave fit for a time period
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        ds = load(xy.pair, day, root, qd)
        cc = process(ds)
        fit = lse_fit(cc, xy, envelope, attrs)
        results.append(fit)
    return results


###############################################################################
# Main functions
# --------------
def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-beamform',
        description='Plane wave beamforming of cross-correlation functions.',
        epilog='See also xcorr-snr xcorr-ct xcorr-timelapse xcorr-psd',
    )
    parser.add_argument(
        'name', metavar='name', type=str,
        help='Array name to select pairs and set the output file'
    )
    parser.add_argument(
        'inventory', metavar='inventory', type=str,
        help='Path to stationxml inventory with receiver coordinates'
    )
    parser.add_argument(
        'start', metavar='start', type=str,
        help='Start date'
    )
    parser.add_argument(
        'end', metavar='end', type=str, default=None, nargs='?',
        help='End date (default: start)'
    )
    parser.add_argument(
        '-c', '--channel', metavar='..', type=str, default='',
        help='Set channel code to select specific pairs'
    )
    parser.add_argument(
        '-e', '--envelope', action="store_true", default=False,
        help=('Calculate the amplitude envelope of the co-array cross-'
              'correlated signal before extracting the time lag at the peak '
              'correlation coefficient (default: `False`)')
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--first', action="store_true", default=True,
        help=('Select first receiver from each pair')
    )
    group.add_argument(
        '--last', action="store_true", default=False,
        help=('Select last receiver from each pair')
    )
    parser.add_argument(
        '--format', metavar='..', type=str, default=None,
        help=('The strftime to parse start and end (default: "%%Y-%%m-%%d"). '
              'See strftime documentation for more information on choices: '
              'https://docs.python.org/3/library/datetime.html#strftime-and-'
              'strptime-behavior.')
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

    # update some arguments
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start, format=args.format)
    args.end = (pd.to_datetime(args.end, format=args.format)
                if args.end else args.start)
    args.out = utils.ncfile('beamform', args.name, args.start, args.end,
                            args.prefix, args.suffix)

    args.pairs = sorted(list(set([
        p.split(os.path.sep)[-1]
        for p in glob(os.path.join(args.root, '*', f'*{args.name}*'))
        if args.channel in p
    ])))

    if len(args.pairs) < 3:
        raise ValueError(
            f'At least three pairs/receivers should be found in "{args.root}"'
        )

    # print header and core parameters
    print(f'xcorr-snr v{xcorr.__version__}')
    print('{:>20} : {}'.format('pairs', args.pairs[0]))
    for pair in args.pairs[1:]:
        print(' '*22, pair)
    print('{:>20} : {}'.format('receiver', 'first' if args.first else 'last'))
    print('{:>20} : {}'.format('inventory', args.inventory))
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('start', args.start.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('end', args.end.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('outfile', args.out))
    print('{:>20} : {}'.format('overwrite', args.overwrite))

    # check if output file exists
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f'Output file "{args.out}" already exists'
                              ' and overwrite is False.')

    # get pair and coordinates
    pair = xr.DataArray(
        data=args.pairs,
        dims=('pair'),
        coords={'pair': args.pairs},
        attrs={
            'long_name': 'Cross-correlation receiver pair',
            'units': '-',
        },
        name='pair',
    )
    if args.debug:
        print(pair)

    inv = read_inventory(args.inventory)
    if args.debug:
        print(inv)
    xy = xcorr.util.receiver.get_pair_xy_coordinates(
        pair, inv, first=args.first
    )
    if args.debug:
        print(xy)

    # init dask cluster and client
    cluster, client = utils.init_dask(n_workers=args.nworkers,
                                      scheduler_file=args.scheduler)

    # fit plane wave
    print('.. compute plane wave per day for period')
    mapped = client.compute(
        delayed_plane_wave_fit(xy, args.start, args.end, args.root,
                               args.quick_and_dirty, args.envelope, args.attrs)
    )
    distributed.wait(mapped)

    print('.. gather plane wave results')
    fit = client.gather(mapped)

    print('.. merge plane wave results')
    fit = xr.merge(list(filter(None, fit)), combine_attrs="override")
    if args.debug:
        print(fit)

    # to netcdf
    print(f'.. write to "{args.out}"')
    xcorr.write(fit, args.out, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot
    if args.plot:
        print('.. plot')

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

        fit.plot.scatter(x='time', y='doa', hue='err', ax=ax1, )
        fit.plot.scatter(x='time', y='vel', hue='err', ax=ax2)

        ax1.set_ylim(0, 360)
        ax1.set_xlabel(None)

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
