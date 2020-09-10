"""
TIMELAPSE
===

Two-dimensional crosscorrelation of crosscorrelation spectrograms.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import distributed
from time import sleep
import os
import argparse

# relative imports
import xcorr
from .helpers import init_dask, close_dask, ncfile

__all__ = []


###############################################################################
# Local functions
# ---------------

def get_spectrogram(pair, time, root):
    """Load spectrogram for a pair and time.
    """
    # construct abs path and filename
    nc = xcorr.util.ncfile(pair, time, root)

    # set lock
    lock = distributed.Lock(nc)
    lock.acquire(timeout='15s')

    # get data from disk
    ds, ok = False, False
    try:
        ds = xcorr.read(nc, fast=True, engine='h5netcdf')
        ds = ds.loc[{'pair': pair, 'time': time}]
        ok = ds.status.all()
        if ok:
            ds.load()
        ds.close()
    except Exception:
        ds = None

    # release lock
    lock.release()

    # no data?
    if ds is None or not ok:
        return

    # extract cc
    cc = ds.cc.where(
        (ds.lag >= ds.distance/1.495) & (ds.lag <= ds.distance/1.465),
        drop=True,
    )

    # no valid data?
    if xr.ufuncs.isnan(cc).any():
        return

    # solve time_offset and pair_offset
    delay = -xcorr.util.time.to_seconds(ds.pair_offset + ds.time_offset)

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag')
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2., padding_factor=8)

    # clear
    cc, ds = None, None

    return psd


def correlate_spectrograms(obj, root):
    """Correlate spectrograms.
    """
    # already set?
    if obj.status.all():
        sleep(.5)  # give scheduler and worker some time
        return obj

    # test if object is loaded
    if not (obj.freq.any() and obj.pair.any()):
        sleep(.5)  # give scheduler and worker some time
        return obj

    # process per item
    for pair in obj.pair:
        for time1 in obj.time1:
            for time2 in obj.time2:

                # already done?
                if obj.status.loc[{
                    'pair': pair,
                    'time1': time1,
                    'time2': time2,
                }].all():
                    continue

                # reset
                psd1, psd2 = None, None

                # load cc and compute psd on-the-fly
                try:
                    psd1 = get_spectrogram(pair, time1, root)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    continue

                if psd1 is None:
                    continue

                try:
                    psd2 = get_spectrogram(pair, time2, root)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    continue

                if psd2 is None:
                    continue

                # correlate per freq range
                for freq in obj.freq:

                    # set (min, max) frequency
                    bw = obj.freq_bw.loc[{'freq': freq}]
                    fmin = (obj.freq - bw/2).values[0]
                    fmax = (obj.freq + bw/2).values[0]

                    # extract freq
                    in1 = psd1.where(
                        (psd1.freq >= fmin) & (psd1.freq < fmax), drop=True,
                    )
                    in2 = psd2.where(
                        (psd2.freq >= fmin) & (psd2.freq < fmax), drop=True,
                    )

                    # correlate psd's
                    cc2 = xcorr.signal.correlate2d(in1, in2)

                    # split dims
                    dim1, dim2 = cc2.dims[-2:]

                    # get max index
                    amax1, amax2 = np.unravel_index(cc2.argmax(), cc2.shape)

                    # store values in object
                    item = {
                        'pair': pair,
                        'freq': freq,
                        'time1': time1,
                        'time2': time2,
                    }
                    obj['status'].loc[item] = np.byte(1)
                    obj['cc2'].loc[item] = cc2.isel({dim1: amax1,
                                                     dim2: amax2})
                    obj[dim1].loc[item] = cc2[dim1][amax1]
                    obj[dim2].loc[item] = cc2[dim2][amax2]

    return obj


def _mask_upper_triangle(ds):
    """Mark upper triangle (one offset)
    """
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        ds.status.loc[{
            'time1': ds.time1[ind1[i]],
            'time2': ds.time2[ind2[i]],
        }] = np.byte(1)


def _fill_upper_triangle(ds):
    """Fill upper triangle (one offset)
    """
    # prevent Dask dataframe issues
    ds.load()
    # copy lower to upper triangle
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        time1 = ds.time1[ind1[i]]
        time2 = ds.time2[ind2[i]]
        triu = {'time1': time1, 'time2': time2}
        tril = {'time1': time2, 'time2': time1}
        ds.cc2.loc[triu] = ds.cc2.loc[tril]
        ds.delta_freq.loc[triu] = -ds.delta_freq.loc[tril]
        ds.delta_lag.loc[triu] = -ds.delta_lag.loc[tril]


def init_spectrogram_timelapse(pair, time, freq, root):
    """Init a spectrogram timelapse dataset.
    """
    # new dataset
    ds = xr.Dataset()

    # set global attributes
    nc = xcorr.util.ncfile(pair[0], time[0], os.path.join(root, 'cc'))
    cc = xcorr.read(nc, quick_and_dirty=True)
    ds.attrs = cc.attrs
    cc.close()
    ds.attrs['xcorr_version'] = xcorr.__version__
    ds.attrs['dependencies_version'] = xcorr.core.core.dependencies_version()

    # remove hashes
    item = ds.attrs.pop('sha256_hash_metadata', None)
    item = ds.attrs.pop('sha256_hash', None)
    del item

    # set coordinates
    ds['pair'] = pair
    ds.pair.attrs = pair.attrs

    ds['freq'] = xr.DataArray(
        np.mean(freq, axis=1),
        dims=('freq'),
        coords=(np.mean(freq, axis=1),),
        attrs={
            'long_name': 'Center Frequency',
            'standard_name': 'center_frequency',
            'units': 's-1',
        },
    )

    # set variables
    ds['freq_bw'] = xr.DataArray(
        np.diff(freq, axis=1).squeeze(),
        dims=('freq'),
        coords=(ds.freq,),
        attrs={
            'long_name': 'Frequency Bandwidth',
            'standard_name': 'frequency_bandwidth',
            'units': 's-1',
        },
    )

    ds['status'] = xr.DataArray(
        np.zeros((len(ds.pair), len(freq), time.size, time.size),
                 dtype=np.byte),
        dims=('pair', 'freq', 'time1', 'time2'),
        coords=(ds.pair, ds.freq, time, time),
        attrs={
            'long_name': 'Crosscorrelation status',
            'standard_name': 'crosscorrelation_status',
            'units': '-',
        },
    )

    ds['cc2'] = ds.status.astype(np.float64) * 0
    ds['cc2'].attrs = {
        'long_name': 'Two-dimensional Crosscorrelation Estimate',
        'standard_name': '2d_crosscorrelation_estimate',
        'units': '-',
        'add_offset': np.float64(0.),
        'scale_factor': np.float64(1.),
        'valid_range': np.float64([-1., 1.]),
        'normalize': np.byte(1),
    }

    ds['delta_freq'] = ds.status.astype(np.float64) * 0
    ds['delta_freq'].attrs = {
        'long_name': 'Delta Frequency',
        'standard_name': 'delta_frequency',
        'units': 's-1',
    }

    ds['delta_lag'] = ds.status.astype(np.float64) * 0
    ds['delta_lag'].attrs = {
        'long_name': 'Delta Lag',
        'standard_name': 'delta_lag',
        'units': 's',
    }

    return ds


def process_spectrogram_timelapse(ds: xr.Dataset, root: str,
                                  chunk: int = None, sparse: bool = True):
    """2-d correlate spectrograms on a Dask client
    """
    # ignore upper triangle
    if sparse:
        _mask_upper_triangle(ds)

    # chunk
    chunk = chunk or 10
    ds = ds.chunk({'time1': chunk, 'time2': chunk})

    # map and persists to client
    mapped = ds.map_blocks(
        correlate_spectrograms,
        args=[os.path.join(root, 'cc')],
        template=ds,
    ).persist()

    # force await on async
    distributed.wait(mapped)

    # load dask to xarray
    ds = mapped.load()

    # fill upper triangle
    if sparse:
        _fill_upper_triangle(ds)

    return ds


def create_locks(ds, root, client=None):
    """Initate distributed cc file access locking.
    """
    locks = []
    for pair in ds.pair:
        for time in ds.time1:
            nc = xcorr.util.ncfile(pair, time, root)
            if nc not in locks:
                locks.append(nc)
    locks = [distributed.Lock(nc, client=client) for nc in locks]
    return locks


###############################################################################
# Main function
# -------------

def help(e=None):
    """Return the help text.
    """
    _help = """
    Two-dimensional crosscorrelation of crosscorrelation spectrograms.

    Usage: xcorr-timelapse <snr_ct> [option] ... [arg] ...
    <snr_ct>         : Path to netcdf dataset with signal-to-noise ratio (snr)
                       and coincidence triggers (ct) indicating the active
                       periods of interest.
    Options and arguments:
    -f, --frequency= : Set psd frequency bands. Frequency should be a list of
                       tuple-pairs with start and end frequencies. Defaults to
                       --frequency="(3., 6.), (6., 12.)".
    """

    print('\n'.join([line[4:] for line in _help.splitlines()]))
    raise SystemExit(e)


def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-timelapse',
        description=('Two-dimensional crosscorrelation of crosscorrelation '
                     'spectrograms.'),
        epilog='See also xcorr-snr xcorr-ct xcorr-psd',
    )
    parser.add_argument(
        'snr_ct', metavar='path', type=str,
        help=('Path to netcdf dataset with signal-to-noise ratio (snr) '
              'and coincidence triggers (ct) indicating the active periods '
              'of interest')
    )
    parser.add_argument(
        '-s', '--start', metavar='start', type=str,
        help='Start date given format yyyy-mm-dd'
    )
    parser.add_argument(
        '-e', '--end', metavar='end', type=str,
        help='End date given format yyyy-mm-dd'
    )
    parser.add_argument(
        '-p', '--pair', metavar='pair', type=str, default='*',
        help='Filter pairs that contain the given string'
    )
    parser.add_argument(
        '-f', '--frequency', metavar='f', type=str, default=None,
        help=('Set psd frequency bands. Frequency should be a list of '
              'tuple-pairs with start and end frequencies (default: '
              '"(3., 6.), (6., 12.)")')
    )
    parser.add_argument(
        '-r', '--root', metavar='path', type=str, default=os.getcwd(),
        help=('Set crosscorrelation root directory (default: current '
              'working directory)')
    )
    parser.add_argument(
        '-n', '--nworkers', metavar='n', type=int, default=None,
        help=('Set number of dask workers for local client. If a scheduler '
              'is set the client will wait until the number of workers is '
              'available.')
    )
    parser.add_argument(
        '--scheduler', metavar='path', type=str, default=None,
        help='Connect to a dask scheduler by a scheduler-file.'
    )
    parser.add_argument(
        '--abundant', action='store_true',
        help=('Compute the entire timelapse 2d correlation matrix instead of '
              'mirroring along the diagonal')
    )
    parser.add_argument(
        '--chunk', metavar='c', type=int, default=10,
        help=('Set dask chunks for time dimension (default: 10)')
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

    # snr and ct files
    snr_ct = xr.open_dataset(args.snr_ct)

    # extract arguments
    root = os.path.abspath(args.root)
    pair = args.pair
    t0 = pd.to_datetime(args.start or snr_ct.time[0].values)
    t1 = pd.to_datetime(args.end or snr_ct.time[-1].values)
    freq = np.array(((3., 6.), (6., 12.)) or args.freq)

    # print header and core parameters
    print(f'xcorr-timelapse v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', t0))
    print('{:>20} : {}'.format('end', t1))
    print('{:>20} : {}'.format(
        'frequency', ', '.join([f'{f[0]}-{f[1]}' for f in freq])
    ))

    # init dask client
    client, cluster = init_dask(n_workers=args.n_workers,
                                scheduler_file=args.scheduler)

    # init timelapse
    print('.. init timelapse dataset', end=', ')
    ds = init_spectrogram_timelapse(
        pair=snr_ct.pair,
        time=snr_ct.time.where(snr_ct.ct >= 0, drop=True),
        freq=freq,
        root=root
    )
    print('dims: pair={pair}, freq={freq}, time={time}'.format(
        pair=ds.pair.size, freq=ds.freq.size, time=ds.time1.size,
    ))
    if args.debug:
        print(ds)

    # to netcdf
    nc = ncfile('timelapse', pair, t0, t1)
    print(f'.. write to "{nc}"')
    xcorr.write(ds, nc, force_write=True, verb=1 if args.debug else 0)

    # load dataset
    print(f'.. load from "{nc}"')
    ds = xr.open_dataset(nc, engine='h5netcdf')
    if args.debug:
        print(ds)

    # create all locks
    print('.. init locks', end=', ')
    locks = create_locks(ds, os.path.join(root, 'cc'), client)
    print(f'files = {len(locks)}')

    # persist to client
    print(f'.. map and compute blocks: chunk={args.chunk}, '
          f'sparse={args.sparse}')
    ds = process_spectrogram_timelapse(ds, root, args.chunk, args.sparse)

    # to netcdf
    print(f'.. write to "{nc}"')
    xcorr.write(ds, nc, variable_encoding=dict(zlib=True, complevel=9))

    # plot?
    if args.plot:
        # common plot settings
        plotset = dict(col='freq', yincrease=False, size=4, aspect=1)

        # plot cc2
        plt.figure()
        ds.cc2.isel(pair=-1).plot(vmin=0, **plotset)
        plt.show()

        # plot delta_lag
        plt.figure()
        ds.delta_lag.isel(pair=-1).plot(robust=True, **plotset)
        plt.show()

        # plot delta_freq
        plt.figure()
        ds.delta_freq.isel(pair=-1).plot(robust=True, **plotset)
        plt.show()

    # close dask client and cluster
    close_dask(client, cluster)

    print('.. done')


if __name__ == "__main__":
    main()
