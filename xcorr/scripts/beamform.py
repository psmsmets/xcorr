"""
Beamform
========

Least-squares beamforming of plane wave traversing the array.

"""

# Mandatory imports
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
import distributed
import argparse
import os
from obspy import read_inventory
from glob import glob

# Relative imports
import xcorr
from .helpers import init_dask, close_dask, ncfile

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(time, pairs, root):
    """
    """
    ds = xcorr.merge(
        [xcorr.util.ncfile(pair, time, root) for pair in pairs],
        fast=True
    )
    return None if ds.pair.size != len(pairs) else ds


@dask.delayed
def process(ds):
    """
    """
    if ds is None:
        return

    cc = ds.cc.where(
        (ds.lag >= ds.distance.mean()/1.50) &
        (ds.lag <= ds.distance.mean()/1.45),
        drop=True,
    )

    if xr.ufuncs.isnan(cc).any():
        return

    # extract time_offset and pair_offset
    delay = -(ds.pair_offset + ds.time_offset) / pd.Timedelta('1s')

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase-wrapping
    cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts

    return cc


@dask.delayed
def lse_fit(cc, xy):
    """
    """
    if cc is None:
        return
    fit = xcorr.signal.beamform.plane_wave(cc, x=xy.x, y=xy.y, dim='lag')
    return fit


def plane_waves_list(xy, start, end, root):
    """Evaluate snr for a list of filenames
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        ds = load(day, xy.pair, root)
        cc = process(ds)
        fit = lse_fit(cc, xy)
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
        description='Plane wave beamforming of crosscorrelation functions.',
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
    args.end = (pd.to_datetime(args.end, format=args.format)
                if args.end else args.start)

    args.pairs = sorted(list(set([
        p.split(os.path.sep)[-1]
        for p in glob(os.path.join(args.root, '*', f'*{args.name}*'))
        if args.channel in p
    ])))

    if len(args.pairs) < 3:
        raise ValueError(
            f'At least three pairs/receivers should be found in "{root}"'
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

    # get pair and coordinates
    pair = xr.DataArray(
        data=args.pairs,
        dims=('pair'),
        coords={'pair': args.pairs},
        attrs={
            'long_name': 'Crosscorrelation receiver pair',
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

    # init dask client
    client, cluster = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # fit plane wae
    print('.. compute plane wave per day for period')
    mapped = client.compute(
        plane_waves_list(xy, args.start, args.end, args.root)
    )
    distributed.wait(mapped)

    print('.. merge plane wave list')
    fit = xr.merge(list(filter(None, client.gather(mapped))))

    if args.debug:
        print(fit)

    # to netcdf
    nc = ncfile('beamform', args.name, args.start, args.end)
    print(f'.. write to "{nc}"')
    xcorr.write(fit, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

        fit.plot.scatter(x='time', y='doa', hue='err', ax=ax1, )
        fit.plot.scatter(x='time', y='vel', hue='err', ax=ax2)

        ax1.set_ylim(0, 360)
        ax1.set_xlabel(None)

        plt.tight_layout()
        plt.show()

    # close dask client and cluster
    close_dask(client, cluster)

    print('.. done')


if __name__ == "__main__":
    main()
