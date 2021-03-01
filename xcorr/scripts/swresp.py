"""
Surface-wave response
=====================

Response between the vertical and radial component of a single station.

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
from glob import glob

# Relative imports
import xcorr
from .helpers import (init_dask, ncfile, add_common_arguments,
                      add_attrs_group, parse_attrs_group)

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(pairs, time, root):
    """
    """
    try:
        ds = xcorr.merge(
            [xcorr.util.ncfile(pair, time, root) for pair in pairs],
            fast=True
        )
    except Exception:
        ds = None
    if ds is not None:
        return ds if ds.pair.size == len(pairs) else None
    else:
        return


@dask.delayed
def process(ds):
    """
    """
    if ds is None:
        return
    try:
        cc = ds.cc.where(
            (ds.lag >= ds.distance.mean()/1.50) &
            (ds.lag <= ds.distance.mean()/1.46) &
            (ds.status == 1),
            drop=True,
        )
        if xr.ufuncs.isnan(cc).any():
            return
        delay = -(ds.pair_offset + ds.time_offset)
        cc = xcorr.signal.unbias(cc)
        cc = xcorr.signal.demean(cc)
        cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase-wrapping
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
        cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
        cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts
    except Exception:
        cc = None

    return cc


@dask.delayed
def surface_wave_response(cc, normalize: bool = True, **kwargs):
    """
    """
    if cc is None:
        return
    try:
        if normalize:
            cc = xcorr.signal.norm1d(cc, dim='lag')

        Y = xcorr.signal.rfft(cc)
        F = Y.isel(pair=1) * xr.ufuncs.conj(Y.isel(pair=0))  # Vertical first

        resp = xr.Dataset()
        resp.attrs = xcorr.util.metadata.global_attrs({
            'title': (
                kwargs.pop('title', '') +
                'Surface wave response - {} to {}'
                .format(
                    cc.time[0].dt.strftime('%Y.%j').item(),
                    cc.time[-1].dt.strftime('%Y.%j').item(),
                )
            ).strip(),
            **kwargs,
            'references': (
                 'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
                 'Random data : analysis and measurement procedures. '
                 'New York (N.Y.): Wiley-Interscience.'
            ),
        })

        resp['magnitude'] = xcorr.signal.abs(F*xr.ufuncs.conj(F))
        resp['magnitude'].attrs = {
            'long_name': 'Magnitude',
            'units': '-',
            'normalize': normalize,
        }
        resp['phase'] = xr.ufuncs.arctan2(xr.ufuncs.real(F),
                                          xr.ufuncs.imag(F)) / np.pi
        resp['phase'].attrs = {'long_name': 'Phase', 'units': 'pi'}
    except Exception:
        resp = None
    return resp


def surface_wave_response_list(pair, start, end, root, **kwargs):
    """Evaluate surface wave response for a list of filenames
    """
    results = []
    for day in pd.date_range(start, end, freq='1D'):
        ds = load(pair, day, root)
        cc = process(ds)
        resp = surface_wave_response(cc, **kwargs)
        results.append(resp)
    return results


###############################################################################
# Main functions
# --------------
def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-swresp',
        description=('Phase difference between vertical and radial component '
                     'of cross-correlation functions.'),
        epilog=('See also xcorr-snr xcorr-ct xcorr-timelapse xcorr-psd '
                'xcorr-beamform'),
    )
    parser.add_argument(
        'station', metavar='station', type=str,
        help='Station code to select channels and set the output file'
    )
    parser.add_argument(
        'start', metavar='start', type=str,
        help='Start date'
    )
    parser.add_argument(
        'end', metavar='end', type=str, default=None, nargs='?',
        help='End date (default: start)'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--radial', action="store_true", default=True,
        help=('Select vertical and radial receiver channel')
    )
    group.add_argument(
        '--transverse', action="store_true", default=False,
        help=('Select vertical and transverse receiver channel')
    )
    parser.add_argument(
        '--format', metavar='..', type=str, default=None,
        help=('The strftime to parse start and end (default: "%%Y-%%m-%%d"). '
              'See strftime documentation for more information on choices: '
              'https://docs.python.org/3/library/datetime.html#strftime-and-'
              'strptime-behavior.')
    )
    parser.add_argument(
        '--disable-norm', action='store_true', default=False,
        help='Disable normalization (default: normalization)'
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set cross-correlation root directory (default: current '
              'working directory)')
    )

    add_common_arguments(parser)
    add_attrs_group(parser)

    args = parser.parse_args()

    # update args
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start, format=args.format)
    args.end = (pd.to_datetime(args.end, format=args.format)
                if args.end else args.start)
    args.channels = 'ZT' if args.transverse else 'ZR'
    args.normalize = not args.disable_norm
    args.out = ncfile('swresp', f'{args.station}_{args.channels}',
                      args.start, args.end, args.prefix, args.suffix)
    args.attrs = parse_attrs_group(args)
    args.pairs = sorted(list(set([
        p.split(os.path.sep)[-1]
        for p in glob(os.path.join(args.root, '*', f'*{args.station}*'))
        if args.channels[0] in p or args.channels[1] in p
    ])))

    if len(args.pairs) != 2:
        raise ValueError(
            f'Two receivers should be found in "{args.root}"'
        )

    # print header and core parameters
    print(f'xcorr-snr v{xcorr.__version__}')
    print('{:>20} : {}'.format('pairs', args.pairs[0]))
    for pair in args.pairs[1:]:
        print(' '*22, pair)
    print('{:>20} : {}'.format('channels', args.channels))
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('start', args.start.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('end', args.end.strftime('%Y-%m-%d')))
    print('{:>20} : {}'.format('normalize', args.normalize))
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
            'long_name': 'Crosscorrelation receiver pair',
            'units': '-',
        },
        name='pair',
    )
    if args.debug:
        print(pair)

    # init dask cluster and client
    cluster, client = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # surface wave response
    print('.. compute surface wave response per day for period')
    mapped = client.compute(
        surface_wave_response_list(
            pair, args.start, args.end, args.root,
            normalize=args.normalize,
        )
    )
    distributed.wait(mapped)

    print('.. gather surface wave response list')
    resp = client.gather(mapped)

    print('.. merge surface wave response list')
    resp = xr.merge(
        objects=list(filter(None, resp)),
        join="outer",
        combine_attrs="override",
    )
    if args.debug:
        print(resp)

    # to netcdf
    print(f'.. write to "{args.out}"')
    xcorr.write(resp, args.out, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot
    if args.plot:
        print('.. plot')
        resp.mean('time', keep_attrs=True).sortby('magnitude').plot.scatter(
            x='freq', y='phase', hue='magnitude', cmap='gray_r',
            size=3, aspect=3, robust=True
        )
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
