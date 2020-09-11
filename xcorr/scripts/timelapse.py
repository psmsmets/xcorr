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
        ds = ds.sel(pair=pair, time=time)
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

    # extract time_offset and pair_offset
    delay = -xcorr.util.time.to_seconds(ds.pair_offset + ds.time_offset)

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag')
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2.5, padding_factor=4)

    return psd


def correlate_spectrograms(obj, root):
    """Correlate spectrograms.
    """
    # complete obj?
    if (obj.status != 0).all():
        sleep(.5)
        return obj

    # process per item
    for pair in obj.pair:
        for time1 in obj.time1:
            for time2 in obj.time2:

                # already done?
                if (obj.status.loc[{
                    'pair': pair, 'time1': time1, 'time2': time2,
                }] != 0).all():
                    continue

                # load cc and compute psd on-the-fly
                psd1 = get_spectrogram(pair, time1, root)
                if psd1 is None:
                    continue
                psd2 = get_spectrogram(pair, time2, root)
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
                    item = {'pair': pair, 'freq': freq,
                            'time1': time1, 'time2': time2}
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


def init_spectrogram_timelapse(pair: xr.DataArray, time: xr.DataArray,
                               freq: np.ndarray, **kwargs):
    """Init a spectrogram timelapse dataset.

    Parameters
    ----------
    pair : :class:`xr.DataArray`
        One-dimensional array of data with receiver couple SEED-id's separated
        with a dash ('-').

    time : :class:`xr.DataArray`
        One-dimensional array of data with selected crosscorrelation times to
        correlated.

    freq : :class:`np.ndarray`
        Two-dimensional array of size Nx2 with spectrogram minimum and maximum
        frequencies.

    Any additional keyword argument will be added to the global attributes of
    the dataset. Please provide 'institution', 'author' and 'source' to comply
    with COARDS and CF conventions.
    """
    # create dataset
    ds = xr.Dataset()

    # global attributes
    ds.attrs = {
        'title': (
            kwargs.pop('title', '') +
            'Timelapse Crosscorrelations - {} to {}'
            .format(
                time[0].dt.strftime('%Y.%j').item(),
                time[-1].dt.strftime('%Y.%j').item(),
            )
        ).strip(),
        'institution': kwargs.pop('institution', 'n/a'),
        'author': kwargs.pop('author', 'n/a'),
        'source': kwargs.pop('source', 'n/a'),
        'history': 'Created @ {}'.format(pd.to_datetime('now')),
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
        'comment': kwargs.pop('comment', 'n/a'),
        'Conventions': 'CF-1.9',
        'xcorr_version': xcorr.__version__,
        'dependencies_version': xcorr.core.core.dependencies_version(),
    }

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


def process_spectrogram_timelapse(
    ds: xr.Dataset, root: str, client: distributed.Client = None,
    chunk: int = None, sparse: bool = True
):
    """2-d correlate spectrograms on a Dask client
    """

    # client
    client = client or distributed.Client()
    if not isinstance(client, distributed.Client):
        raise TypeError('client should be a Dask distributed Client object')

    # ignore upper triangle
    if sparse:
        _mask_upper_triangle(ds)

    # chunk
    chunk = chunk or 10
    ds = ds.chunk({'time1': chunk, 'time2': chunk})

    # map and persist blocks
    mapped = ds.map_blocks(
        correlate_spectrograms,
        args=[root],
        template=ds,
    ).persist()

    # force await on async
    distributed.wait(mapped)

    # load blocks
    ds = client.gather(mapped)

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
        '-f', '--frequency', metavar='..', type=str, default=None,
        help=('Set psd frequency bands. Frequency should be a list of '
              'tuple-pairs with start and end frequencies (default: '
              '"(3., 6.), (6., 12.)")')
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
        '--abundant', action='store_true',
        help=('Compute the entire timelapse 2d correlation matrix instead of '
              'mirroring along the diagonal')
    )
    parser.add_argument(
        '-c', '--chunk', metavar='..', type=int, default=10,
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

    # snr and ct file
    snr_ct = xr.open_dataset(args.snr_ct)

    # update arguments
    args.root = os.path.abspath(args.root)
    args.start = pd.to_datetime(args.start or snr_ct.time[0].values)
    args.end = pd.to_datetime(args.end or snr_ct.time[-1].values)
    args.freq = np.array(((3., 6.), (6., 12.)) or args.freq)

    # print header and core parameters
    print(f'xcorr-timelapse v{xcorr.__version__}')
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))
    print('{:>20} : {}'.format(
        'frequency', ', '.join([f'{f[0]}-{f[1]}' for f in args.freq])
    ))

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

    # init timelapse
    print('.. init timelapse dataset', end=', ')
    ds = init_spectrogram_timelapse(
        pair=snr.pair,
        time=ct.time.where(ct >= 0, drop=True),
        freq=args.freq,
    )
    print('dims: pair={pair}, freq={freq}, time={time}'.format(
        pair=ds.pair.size, freq=ds.freq.size, time=ds.time1.size,
    ))
    if args.debug:
        print(ds)

    # to netcdf and lazy load dataset
    nc = ncfile('timelapse', args.pair, args.start, args.end)
    print(f'.. write and lazy reload dataset')
    xcorr.write(ds, nc, force_write=True, verb=1 if args.debug else 0)
    ds = xr.open_dataset(nc, engine='h5netcdf')
    if args.debug:
        print(ds)

    # create all locks
    print('.. init locks', end=', ')
    locks = create_locks(ds, args.root, client)
    print(f'files = {len(locks)}')

    # persist to client
    print(f'.. map and compute blocks: chunk={args.chunk}, '
          f'sparse={not args.abundant}')
    ds = process_spectrogram_timelapse(ds, args.root, client, chunk=args.chunk,
                                       sparse=not args.abundant)

    # to netcdf
    print(f'.. write to "{nc}"')
    xcorr.write(ds, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)
    # plot?
    if args.plot:
        plotset = dict(col='freq', yincrease=False, size=4, aspect=1)
        ds.cc2.isel(pair=-1).plot(vmin=0, **plotset)
        ds.delta_lag.isel(pair=-1).plot(robust=True, **plotset)
        ds.delta_freq.isel(pair=-1).plot(robust=True, **plotset)
        plt.show()

    # close dask client and cluster
    close_dask(client, cluster)

    print('.. done')


if __name__ == "__main__":
    main()
