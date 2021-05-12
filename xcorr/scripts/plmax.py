"""
PLMAX
======

Spectrogram/scaleogram peak local maxima of snr triggered periods.

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
def get_spectrogram(
    pair, time, root, period=None, clim=(1460, 1500), cwt=False
):
    """Load spectrogram for a pair and time.
    """
    # construct filename
    nc = xcorr.io.ncfile(pair, time, root)
    if not os.path.isfile(nc):
        return

    # read
    ds = xcorr.read(nc, fast=True, engine='h5netcdf')
    if ds is None:
        return

    # process
    try:
        ds = ds.xcorr.postprocess(
            clim=clim,
            time_lim=period,
            filter_kwargs=dict(frequency=3., order=2),
        )
    except ValueError:
        ds = None

    # obtain spectrogram
    s = (ds.cc.signal.scaleogram(wavelet="cmor1.0-3.0", scales=500) if cwt else
         ds.cc.signal.spectrogram(duration=2.5, padding_factor=4))

    # clean
    ds.close()
    del ds

    return s


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
# Lazy plmax for pairs and periods
# --------------------------------

def period_plmax(pairs, ct, root, clim, cwt, attrs):
    """Evaluate psds for a pair and a set of periods
    """
    df = []
    for (pid, period) in ct.time.groupby(ct):
        t0 = period[0].values
        t1 = period[-1].values
        for day in pd.date_range(t0, t1, freq='1D', normalize=True):
            for pair in pairs:
                p = get_spectrogram(pair, day, root, (t0, t1), clim, cwt)
                p = peak_local_max(p, attrs)
                df.append(p)

    return df


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """
    # arguments
    parser = argparse.ArgumentParser(
        prog='xcorr-plmax',
        description=('Spectrogram/Scaleogram peak local maxima of '
                     'signal-to-noise ratio triggered periods.'),
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
    parser.add_argument(
        '-w', '--wavelet', action="store_true", default=False,
        help=('Compute the scaleogram by the Continuous Wavelet Transform '
              'using a cmor1.0-3.0 wavelet instead of the default '
              'short-time Fourier-based spectogram.')
    )

    utils.add_common_arguments(parser)
    utils.add_attrs_group(parser)

    args = parser.parse_args()
    args.attrs = utils.parse_attrs_group(args)

    # snr and ct file
    snr_ct = xr.open_dataset(args.snr_ct)

    # update arguments
    if args.start:
        args.start = pd.to_datetime(args.start, format=args.format)
    if args.end:
        args.end = pd.to_datetime(args.end, format=args.format)
    args.clim = tuple(eval(args.clim))
    if len(args.clim) != 2:
        raise ValueError("Celerity range should be a tuple of length 2: "
                         "(min, max)")

    # print header and core parameters
    print(f'xcorr-plmax v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))
    print('{:>20} : {}'.format('clim', args.clim))
    print('{:>20} : {}'.format('method', ('scaleogram (CWT)' if args.wavelet
                                          else 'spectrogram (FFT)')))
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
        "plmax_{}".format('scaleogram' if args.wavelet else 'spectogram'),
        args.pair, snr.time[0].item(), snr.time[-1].item(),
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
        period_plmax(snr.pair, ct, args.root, args.clim,
                     args.wavelet, args.attrs)
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
