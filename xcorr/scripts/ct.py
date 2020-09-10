"""
SNR
===

Coincidence triggers of crosscrorrelation signal-to-noise ratios.

"""

# Mandatory imports
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import argparse

# Relative imports
import xcorr
from .helpers import ncfile

__all__ = []


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-ct',
        description=('Coincidence triggers of crosscrorrelation '
                     'signal-to-noise ratios.'),
        epilog='See also xcorr-snr xcorr-timelapse xcorr-psd',
    )
    parser.add_argument(
        'paths', metavar='paths', type=str, nargs='+',
        help='Paths to netcdf files with signal-to-noise ratios'
    )
    parser.add_argument(
        '-s', '--start', metavar='..', type=str, default=None,
        help='Set start datetime (format: yyyy-mm-dd)'
    )
    parser.add_argument(
        '-e', '--end', metavar='..', type=str, default=None,
        help='Set end datetime (format: yyyy-mm-dd)'
    )
    parser.add_argument(
        '-p', '--pair', metavar='..', type=str, default='',
        help='Filter pairs that contain the given string'
    )
    parser.add_argument(
        '-t', '--threshold', metavar='..', type=float, default=10.,
        help='Coincidence trigger threshold (default: 10.)'
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

    # print header and core parameters
    print(f'xcorr-ct v{xcorr.__version__}')
    print('{:>20} : {}'.format('files', len(args.paths)))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    if args.start:
        print('{:>20} : {}'.format('start', args.start))
    if args.end:
        print('{:>20} : {}'.format('end', args.end))
    print('{:>20} : {}'.format('threshold', args.threshold))
    print('{:>20} : {}'.format('plot', args.plot))

    if args.debug:
        print(args.paths)

    # snr
    print('.. load signal-to-noise ratio')
    ds = xr.merge([xr.open_dataarray(path) for path in args.paths])
    if args.debug:
        print(ds)

    # filter snr
    print('.. filter snr for pair and time range')
    t0 = pd.to_datetime(args.start or ds.time[0].values)
    t1 = pd.to_datetime(args.end or ds.time[-1].values)
    ds = ds.where(
        (
            (ds.time >= t0.to_datetime64()) &
            (ds.time < t1.to_datetime64()) &
            (ds.pair.str.contains(args.pair))
        ),
        drop=True,
    )
    if args.debug:
        print(ds)

    # get confindence triggers
    print('.. get coincidence triggers', end=', ')
    ds['ct'] = xcorr.signal.coincidence_trigger(
        ds.snr, thr_on=args.threshold, extend=0, thr_coincidence_sum=None,
    )
    print(f'periods = {ds.ct.attrs["nperiods"]}')
    if args.debug:
        print(ds.ct)

    # timelapse filename
    nc = ncfile('snr_ct', args.pair, t0, t1)

    # to netcdf
    print(f'.. write to "{nc}"')
    xcorr.write(ds, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    if args.plot:
        ds.snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                         add_legend=False)
        xcorr.signal.trigger.plot_trigs(ds.snr, ds.ct)
        plt.tight_layout()
        plt.show()

    print('.. done')


if __name__ == "__main__":
    main()