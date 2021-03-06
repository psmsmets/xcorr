"""
PSD
===

Spectrograms of triggered datasets by snr using dask.

"""

# Mandatory imports
import numpy as np
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
def extract_period(ds, period):
    """Extract time period
    """
    t0, t1 = np.datetime64(period.start), np.datetime64(period.end)
    mask = (ds.time >= t0) & (ds.time <= t1)
    ds = ds.where(mask, drop=True)
    return ds


@dask.delayed
def load_and_postprocess(pair, period, root):
    """Load cc of a single pair for a period.
    """
    t0 = period.start + pd.tseries.offsets.DateOffset(normalize=True)
    t1 = period.end + pd.tseries.offsets.DateOffset(normalize=True)
    src = os.path.join(root, 'cc')

    files = [xcorr.io.ncfile(str(pair.values), t, src)
             for t in pd.date_range(t0, t1, freq='1D')]

    ds = xcorr.mfread(files).postprocess(
        clim=(1460, 1500),
        time_lim=(np.datetime64(period.start), np.datetime64(period.end)),
        filter_kwargs=dict(frequency=1.5, order=4),
    )

    return ds


@dask.delayed
def spectrogram(cc):
    """Calculate spectrogram
    """
    psd = cc.signal.spectrogram(duration=2.5, padding_factor=4)
    psd = psd.where((psd.freq <= 20.), drop=True)  # Nyquist
    return psd


@dask.delayed
def combine(ds, psd, snr, attrs):
    """Combine all into a single dataset
    """
    ds['psd'] = psd
    ds['snr'] = snr.loc[{'pair': ds.pair[0]}]
    ds.attrs = {**ds.attrs, **attrs}
    return ds


@dask.delayed
def write(ds, period, root):
    """Merge spectrogram list
    """
    # parameters
    pair = str(ds.pair[0].values)

    # set filename and path
    filename = '{p}.{y:04d}.{d:03d}.{h:03d}.psd.nc'.format(
        p=pair,
        y=period.start.year,
        d=period.start.dayofyear,
        h=int(period.days*24),
    )
    path = os.path.join(root, 'psd', pair, filename)

    # write
    xcorr.write(ds, path, verb=-1, hash_data=False)

    return path


###############################################################################
# Lazy psd for pairs and periods
# ------------------------------

def period_spectrograms(snr, ct, root, attrs, overwrite: bool = True):
    """Evaluate psds for a pair and a set of periods
    """
    periods = xcorr.signal.trigger.trigger_periods(ct)
    fnames = []

    for index, period in periods.iterrows():
        snr_period = extract_period(snr, period)

        for pair in snr.pair:
            ds = load_and_postprocess(pair, period, root)
            psd = spectrogram(ds.cc)
            ds = combine(ds, psd, snr_period, attrs)
            fname = write(ds, period, root)
            fnames.append(fname)

    return fnames


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """
    # arguments
    parser = argparse.ArgumentParser(
        prog='xcorr-psd',
        description=('Spectrogram estimation of signal-to-noise ratio '
                     'triggered periods.'),
        epilog='See also xcorr-snr xcorr-ct xcorr-timelapse xcorr-beamform',
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
        '-p', '--pair', metavar='..', type=str, default='*',
        help='Filter pairs that contain the given string'
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set cross-correlation root directory (default: current '
              'working directory)')
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

    # print header and core parameters
    print(f'xcorr-psd v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))
    print('{:>20} : {}'.format('overwrite', args.overwrite))

    args.start = args.start or pd.to_datetime(snr_ct.time[0].item())
    args.end = args.end or pd.to_datetime(snr_ct.time[-1].item())

    # init dask client
    cluster, client = utils.init_dask(n_workers=args.nworkers,
                                      scheduler_file=args.scheduler)

    # filter snr and ct
    print('.. filter snr and ct')
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

    # construct datasets with preprocessed cc, snr and psd
    print('.. construct files per active period')

    mapped = client.compute(
        period_spectrograms(snr, ct, args.root, args.attrs, args.overwrite)
    )
    distributed.wait(mapped)

    files = client.gather(mapped)
    if args.debug:
        print(files)

    # close dask client and cluster
    print('.. close dask')
    client.close()
    if cluster is not None:
        cluster.close()

    print('.. done')
    sys.exit(0)


if __name__ == "__main__":
    main()
