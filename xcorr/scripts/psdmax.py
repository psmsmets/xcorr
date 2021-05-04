"""
PSDMAX
======

Spectrogram peak local maxima of triggered datasets by snr using dask.

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
from . import utils

__all__ = []


###############################################################################
# Delayed functions
# -----------------


@dask.delayed
def load_and_postprocess(pair, period, root, clim=(1460, 1500)):
    """Load cc of a single pair for a period.
    """
    t0 = period.start + pd.tseries.offsets.DateOffset(normalize=True)
    t1 = period.end + pd.tseries.offsets.DateOffset(normalize=True)
    src = os.path.join(root, 'cc')

    files = []
    for t in pd.date_range(t0, t1, freq='1D'):
        file = xcorr.io.ncfile(str(pair.values), t, src)
        if os.path.isfile(file):
            files.append(file)

    # nothing found?
    if not files:
        return

    ds = xcorr.mfread(files, naive=True, parallel=False)
    if ds is None:
        return
    try:
        ds = ds.xcorr.postprocess(
            clim=clim,
            time_lim=(period.start.to_datetime64(), period.end.to_datetime64()),
            filter_kwargs=dict(frequency=3., order=2),
        )
    except ValueError:
        ds = None

    return ds


@dask.delayed
def spectrogram(ds):
    """Calculate spectrogram
    """
    if ds is None:
        return
    return ds.cc.signal.spectrogram(duration=2.5, padding_factor=4)


@dask.delayed
def peak_local_max(da, attrs):
    """Determine spectrogram peak local max
    """
    if da is None:
        return
    df = da.signal.peak_local_max(
        min_distance=25,
        threshold_rel=.01,
        extend=True,
        as_dataframe=True,
        attrs=attrs
    )
    return df.reset_index()


###############################################################################
# Lazy psd for pairs and periods
# ------------------------------

def period_spectrograms_max(snr, ct, root, clim, attrs):
    """Evaluate psds for a pair and a set of periods
    """
    periods = xcorr.signal.trigger.trigger_periods(ct)
    df = []

    for index, period in periods.iterrows():
        for pair in snr.pair:
            d = load_and_postprocess(pair, period, root, clim)
            d = spectrogram(d)
            d = peak_local_max(d, attrs)
            df.append(d)

    return df


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """
    # arguments
    parser = argparse.ArgumentParser(
        prog='xcorr-psdmax',
        description=('Spectrogram peak local maxima estimation of '
                     ' signal-to-noise ratio triggered periods.'),
        epilog=('See also xcorr-psd xcorr-snr xcorr-ct xcorr-timelapse '
                'xcorr-beamform'),
    )
    parser.add_argument(
        'snr_ct', metavar='snr_ct', type=str,
        help=('Path to netcdf dataset with signal-to-noise ratio (snr) '
              'and coincidence triggers (ct) indicating the active periods '
              'of interest')
    )
    parser.add_argument(
        '-s', '--start', metavar='..', type=str,
        help='Start date (format: yyyy-mm-dd)'
    )
    parser.add_argument(
        '-e', '--end', metavar='', type=str,
        help='End date (format: yyyy-mm-dd)'
    )
    parser.add_argument(
        '--format', metavar='..', type=str, default=None,
        help=('The strftime to parse start and end (default: "%%Y-%%m-%%d"). '
              'See strftime documentation for more information on choices: '
              'https://docs.python.org/3/library/datetime.html#strftime-and-'
              'strptime-behavior.')
    )
    parser.add_argument(
        '-p', '--pair', metavar='..', type=str, default='',
        help='Filter pairs that contain the given string'
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set cross-correlation root directory (default: current '
              'working directory)')
    )
    parser.add_argument(
        '-c', '--clim', metavar='..', type=str, default="1460, 1500",
        help='Celerity range (min, max) in meters per second'
    )

    utils.add_common_arguments(parser)
    utils.add_attrs_group(parser)

    args = parser.parse_args()
    args.attrs = utils.parse_attrs_group(args)

    # snr and ct file
    snr_ct = xr.open_dataset(args.snr_ct)

    # update arguments
    args.root = os.path.abspath(args.root)
    if args.start:
        args.start = pd.to_datetime(args.start, format=args.format)
    if args.end:
        args.end = pd.to_datetime(args.end, format=args.format)
    args.clim = tuple(eval(args.clim))
    if len(args.clim) != 2:
        raise ValueError("Celerity range should be a tuple of length 2: "
                         "(min, max)")

    # print header and core parameters
    print(f'xcorr-psdmax v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))
    print('{:>20} : {}'.format('clim', args.clim))
    print('{:>20} : {}'.format('overwrite', args.overwrite))

    args.start = args.start or pd.to_datetime(snr_ct.time[0].item())
    args.end = args.end or pd.to_datetime(snr_ct.time[-1].item())

    # init dask client
    cluster, client = utils.init_dask(n_workers=args.nworkers,
                                      scheduler_file=args.scheduler)

    # extract snr and ct
    print('..Extract snr and ct')
    if args.debug:
        print(snr_ct)
    snr = snr_ct.snr.where(
        (
            (snr_ct.time >= args.start.to_datetime64()) &
            (snr_ct.time < args.end.to_datetime64()) &
            (snr_ct.pair.str.contains(args.pair))
        ),
        drop=True,
    )
    if args.debug:
        print(snr)
    ct = snr_ct.ct.where(
        (
            (snr_ct.time >= args.start.to_datetime64()) &
            (snr_ct.time < args.end.to_datetime64())
        ),
        drop=True,
    )
    if args.debug:
        print(ct)
    if args.plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        xcorr.signal.trigger.plot_trigs(snr, ct)
        plt.tight_layout()
        plt.show()

    # output filename
    h5 = utils.h5file(
        'psdmax', args.pair, snr.time[0].item(), snr.time[-1].item(),
        args.prefix, args.suffix
    )

    # check if output file exists
    if os.path.exists(h5) and not args.overwrite:
        raise FileExistsError(f'Output file "{h5}" already exists'
                              ' and overwrite is False.')

    # create output file
    container = pd.HDFStore(h5)

    # construct datasets with preprocessed cc, snr and psd
    print("..Find spectrogram local maxima for all active periods")

    mapped = client.compute(
        period_spectrograms_max(snr, ct, args.root, args.clim, args.attrs)
    )
    distributed.wait(mapped)

    # gather dataframe list from client
    df = client.gather(mapped)

    # merge dataframes
    df = pd.concat(df, ignore_index=True)
    if args.debug:
        print(df)

    # write to hdf5
    print(f'..Write to "{h5}"')
    container['df'] = df
    container.close()

    # close dask client and cluster
    print('..Close dask')
    client.close()
    if cluster is not None:
        cluster.close()

    print('..Done')
    sys.exit(0)


if __name__ == "__main__":
    main()
