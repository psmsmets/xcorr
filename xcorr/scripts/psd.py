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
import argparse

# Relative imports
import xcorr
from .helpers import init_dask, close_dask

__all__ = []


###############################################################################
# Delayed functions
# -----------------

@dask.delayed
def load(pair, period, root):
    """Load cc of a single pair for a period.
    """
    t0 = period.start + pd.tseries.offsets.DateOffset(normalize=True)
    t1 = period.end + pd.tseries.offsets.DateOffset(normalize=True)
    src = os.path.join(root, 'cc')

    files = []
    for t in pd.date_range(t0, t1, freq='1D'):
        files.append(xcorr.util.ncfile(str(pair.values), t, src))

    ds = xcorr.merge(files)

    # extract valid data
    vel = dict(min=1.465, max=1.495)
    t0, t1 = np.datetime64(period.start), np.datetime64(period.end)
    mask = xcorr.signal.multi_mask(
        x=ds.lag,
        y=ds.distance,
        lower=vel['min'],
        upper=vel['max'],
        invert=True,
    ) & (ds.status == 1) & (ds.time >= t0) & (ds.time <= t1)

    # copy to avoid dimension blow-up
    distance = ds.distance
    pair_offset = ds.pair_offset
    time_offset = ds.time_offset

    # drop and mask
    ds = ds.drop_vars(
        ['status', 'distance', 'pair_offset', 'time_offset']
    ).where(mask, drop=True)

    # add back
    ds['distance'] = distance
    ds['pair_offset'] = pair_offset
    ds['time_offset'] = time_offset

    return ds


@dask.delayed
def extract_period(ds, period):
    """Extract time period
    """
    t0, t1 = np.datetime64(period.start), np.datetime64(period.end)
    mask = (ds.time >= t0) & (ds.time <= t1)
    ds = ds.where(mask, drop=True)
    return ds


@dask.delayed
def preprocess(ds):
    """Preprocess cc
    """
    cc = ds.cc
    cc = xcorr.signal.detrend(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    delay = -xcorr.util.time.to_seconds(ds.pair_offset + ds.time_offset)
    if (delay != 0.).any():
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag')
        cc = xcorr.signal.taper(cc, max_length=2/3.)

    return cc


@dask.delayed
def spectrogram(cc):
    """Calculate spectrogram
    """
    psd = xcorr.signal.spectrogram(cc, duration=2., padding_factor=4)
    psd = psd.where((psd.freq >= 1.5) & (psd.freq <= 18.), drop=True)
    return psd


@dask.delayed
def combine(ds, cc, psd, snr):
    """Combine all into a single dataset
    """
    ds['cc'] = cc
    ds['psd'] = psd
    ds['snr'] = snr.loc[{'pair': ds.pair[0]}]
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

def period_spectrograms(snr, ct, root):
    """Evaluate psds for a pair and a set of periods
    """
    periods = xcorr.signal.trigger.trigger_periods(ct)
    fnames = []

    for index, period in periods.iterrows():
        snr_period = extract_period(snr, period)

        for pair in snr.pair:
            ds = load(pair, period, root)
            cc = preprocess(ds)
            psd = spectrogram(cc)
            ds = combine(ds, cc, psd, snr_period)
            fname = write(ds, period, root)
            fnames.append(fname)
    return fnames


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-psd',
        description=('Spectrogram estimation of signal-to-noise ratio '
                     'triggered periods.'),
        epilog='See also xcorr-snr xcorr-ct xcorr-timelapse',
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
        '-p', '--pair', metavar='..', type=str, default='*',
        help='Filter pairs that contain the given string'
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
        '--scheduler', metavar='path', type=str, default=None,
        help='Connect to a dask scheduler by a scheduler-file.'
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

    # snr and ct file
    snr_ct = xr.open_dataset(args.snr_ct)

    # update arguments
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start or snr_ct.time[0].values)
    args.end = pd.to_datetime(args.end or snr_ct.time[-1].values)

    # print header and core parameters
    print(f'xcorr-psd v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))

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

    # init dask client
    client, cluster = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # construct datasets with preprocessed cc, snr and psd
    print('.. construct files per active period')

    mapped = client.compute(period_spectrograms(snr, ct, args.root))
    distributed.wait(mapped)

    files = client.gather(mapped)
    if args.debug:
        print(files)

    # close dask client and cluster
    close_dask(client, cluster)

    print('.. done')


if __name__ == "__main__":
    main()
