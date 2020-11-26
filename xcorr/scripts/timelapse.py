"""
TIMELAPSE
===

Two-dimensional cross-correlation of cross-correlation spectrograms.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import distributed
from time import sleep
import os
import sys
import argparse

# relative imports
import xcorr
from .helpers import init_dask, ncfile

__all__ = []


###############################################################################
# Local functions
# ---------------

def init_spectrogram_timelapse(pair: xr.DataArray, time: xr.DataArray,
                               freq: np.ndarray, **kwargs):
    """Init a spectrogram timelapse dataset.

    Parameters
    ----------
    pair : :class:`xr.DataArray`
        One-dimensional array of data with receiver couple SEED-id's separated
        with a dash ('-').

    time : :class:`xr.DataArray`
        One-dimensional array of data with selected cross-correlation times to
        correlated.

    freq : :class:`np.ndarray`
        Two-dimensional array of size Nx2 with spectrogram minimum and maximum
        frequencies.

    Any additional keyword argument will be added to the global attributes of
    the dataset. Please provide 'institution', 'author' and 'source' to comply
    with COARDS and CF conventions.
    """
    ds = xr.Dataset()
    ds.attrs = xcorr.util.metadata.global_attrs({
        'title': (
            kwargs.pop('title', '') +
            'Timelapse Cross-correlations - {} to {}'
            .format(
                time[0].dt.strftime('%Y.%j').item(),
                time[-1].dt.strftime('%Y.%j').item(),
            )
        ).strip(),
        **kwargs,
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
    })

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
            'long_name': 'Cross-correlation status',
            'standard_name': 'cross-correlation_status',
            'units': '-',
        },
    )

    ds['cc2'] = ds.status.astype(np.float64) * 0
    ds['cc2'].attrs = {
        'long_name': 'Two-dimensional Cross-correlation Estimate',
        'standard_name': '2d_cross-correlation_estimate',
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


def get_spectrogram(pair, time, root):
    """Load spectrogram for a pair and time.
    """
    # construct abs path and filename
    nc = xcorr.util.ncfile(pair, time, root)

    # set lock
    lock = distributed.Lock(name=nc)
    lock.acquire(timeout='5s')

    # get data from disk
    ds, ok = False, False
    try:
        ds = xcorr.read(nc, fast=True, engine='h5netcdf')
        ds = ds.sel(pair=pair, time=time)
        ok = (ds.status == 1).all()
        if ok:
            ds.load()
        ds.close()
    except Exception:
        ds = None

    # release lock
    try:
        if lock.locked():
            lock.release()
    except Exception as e:
        print(e)

    # no data?
    if ds is None or not ok:
        return

    # extract cc
    cc = ds.cc.where(
        (ds.lag >= ds.distance/1.50) &
        (ds.lag <= ds.distance/1.46),
        drop=True,
    )

    # no valid data?
    if xr.ufuncs.isnan(cc).any():
        return

    # extract time_offset and pair_offset
    delay = -(ds.pair_offset + ds.time_offset) / pd.Timedelta('1s')

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase wrapping
    cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2.5, padding_factor=4)

    return psd


def correlate_spectrogram(obj, pair, time1, time2, root):
    """Correlate spectrogram.
    """
    # locs
    items = {'pair': pair, 'time1': time1, 'time2': time2}

    # already done?
    if (obj.status.loc[items] == 1).all():
        return

    # load cc and compute psd on-the-fly
    psd1, psd2 = None, None
    try:
        psd1 = get_spectrogram(pair, time1, root)
        psd2 = get_spectrogram(pair, time2, root)
    except Exception as e:
        print(e)
    if psd1 is None or psd2 is None:
        return

    # correlate per freq range
    for freq in obj.freq:

        # set (min, max) frequency
        bw = obj.freq_bw.loc[{'freq': freq}]
        fmin = (obj.freq - bw/2).values[0]
        fmax = (obj.freq + bw/2).values[0]

        # extract freq
        in1 = psd1.where((psd1.freq >= fmin) & (psd1.freq < fmax), drop=True)
        in2 = psd2.where((psd2.freq >= fmin) & (psd2.freq < fmax), drop=True)

        # correlate psd's
        cc2 = xcorr.signal.correlate2d(in1, in2)

        # split dims
        dim1, dim2 = cc2.dims[-2:]

        # get max index
        amax1, amax2 = np.unravel_index(cc2.argmax(), cc2.shape)

        # store values in object
        item = {**items, 'freq': freq}
        obj['status'].loc[item] = np.byte(1)
        obj['cc2'].loc[item] = cc2.isel({dim1: amax1, dim2: amax2})
        obj[dim1].loc[item] = cc2[dim1][amax1]
        obj[dim2].loc[item] = cc2[dim2][amax2]

    return


def correlate_spectrograms(obj, root):
    """Correlate spectrograms.
    """
    # complete obj?
    if (obj.status == 1).all():
        sleep(2.)
        return obj

    # process per item
    for pair in obj.pair:
        for time1 in obj.time1:
            for time2 in obj.time2:
                correlate_spectrogram(obj, pair, time1, time2, root)
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


def _all_ncfiles(pair, time, root):
    """Construct list of unique list of ncfiles
    """
    ncfiles = []
    for p in pair:
        for t in time:
            nc = xcorr.util.ncfile(p, t, root)
            if nc not in ncfiles:
                ncfiles.append(nc)
    return ncfiles


def spectrogram_timelapse_on_client(
    ds: xr.Dataset, root: str, client: distributed.Client,
    chunk: int = 10, sparse: bool = True, merge: bool = False, verb: int = 1
):
    """2-d correlate spectrograms on a Dask client

    Parameters:
    -----------

    ds: :class:`xr.Dataset`
        Timelapse dataset.

    root: `str`
        Cross-correlation root directory

    client: :class:`distributed.Client`
        Dask distributed client object.

    chunk: `int`, optional
        Dask map blocks chunk size for time1 and time2 (default: 10).

    sparse: `bool`, optional
        Only calculate the lower diagonal cross-correlation values
        (default: `True`).

    merge: `bool`
        Merge ``ds`` with the updated timelapse dataset. Avoid when calculating
        a freshly initiated dataset.

    verb: `int`, optional
        Set verbosity level (default: 1).
    """

    # parameters
    if verb > 0:
        print('.. map and compute blocks')
        print('{:>20} : {}'.format('chunk', chunk))
        print('{:>20} : {}'.format('sparse', sparse))
        print('{:>20} : {}'.format('merge', merge))

    # ignore upper triangle
    if sparse:
        _mask_upper_triangle(ds)

    # extract
    new = ds.drop_vars('freq_bw').where((ds.status != 1), drop=True)
    new['freq_bw'] = ds['freq_bw']

    # create locks
    locks = [distributed.Lock(nc)
             for nc in _all_ncfiles(new.pair, new.time1, root)]
    if verb > 0:
        print('{:>20} : {}'.format('locks', len(locks)))

    # chunk
    new = new.chunk({'time1': chunk, 'time2': chunk})

    # map and persist blocks
    mapped = new.map_blocks(
        correlate_spectrograms,
        args=[root],
        template=new,
    ).persist()

    # force await async
    distributed.wait(mapped)

    # compute blocks
    new = client.gather(mapped).load()

    # merge
    if merge:
        new = xr.merge([new, ds], join='outer', compat='override')
        try:
            ds = xr.merge([new, ds], join='outer')
        except Exception as e:
            print(e)
            ds = new
    else:
        ds = new
    # ds = xr.merge([new, ds], join='outer') if merge else new

    # fill upper triangle
    if sparse:
        _fill_upper_triangle(ds)

    return ds


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """

    parser = argparse.ArgumentParser(
        prog='xcorr-timelapse',
        description=('Two-dimensional cross-correlation of cross-correlation '
                     'spectrograms.'),
        epilog='See also xcorr-snr xcorr-ct xcorr-psd xcorr-beamform',
    )
    parser.add_argument(
        'paths', metavar='paths', type=str, nargs='+',
        help=('Paths to netcdf datasets to "--init" or "--update" timelapse')
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-i', '--init', action="store_true",
        help=('Paths to datasets with signal-to-noise ratio (snr) and '
              'coincidence triggers (ct) indicating the active periods of '
              'interest to initiate a new timelapse dataset')
    )
    group.add_argument(
        '-u', '--update', action="store_true",
        help=('Paths to timelapse datasets to be merged and updated')
    )
    parser.add_argument(
        '-s', '--start', metavar='..', type=str,
        help='Start date'
    )
    parser.add_argument(
        '-e', '--end', metavar='', type=str,
        help='End date'
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
        '-f', '--frequency', metavar='..', type=str, default=None,
        help=('Set psd frequency bands. Frequency should be a list of '
              'tuple-pairs with start and end frequencies (default: '
              '"(3., 6.), (6., 12.)")')
    )
    parser.add_argument(
        '-r', '--root', metavar='..', type=str, default=os.getcwd(),
        help=('Set cross-correlation root directory (default: current '
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

    # open and merge paths
    ds = xr.open_mfdataset(
        paths=args.paths,
        combine='by_coords',
        data_vars='minimal',
        join='outer',
    )
    ds.load()
    ds.close()

    if args.debug:
        print(ds)

    # update arguments
    args.root = os.path.abspath(args.root)
    if args.start:
        args.start = pd.to_datetime(args.start, format=args.format)
    if args.end:
        args.end = pd.to_datetime(args.end, format=args.format)
    if args.init:
        args.freq = np.array(((3., 6.), (6., 12.)) or args.freq)
    else:
        args.freq = None

    # print header and core parameters
    print(f'xcorr-timelapse v{xcorr.__version__}')
    print('{:>20} : {}'.format('action', 'update' if args.update else 'init'))
    print('{:>20} : {}'.format('root', args.root))
    print('{:>20} : {}'.format('pair', 'all' if args.pair in ('*', '')
                               else args.pair))
    print('{:>20} : {}'.format('start', args.start))
    print('{:>20} : {}'.format('end', args.end))
    if args.freq is not None:
        print('{:>20} : {}'.format(
            'frequency', (', '.join([f'{f[0]}-{f[1]}' for f in args.freq]))
        ))

    # init dask cluster and client
    cluster, client = init_dask(n_workers=args.nworkers,
                                scheduler_file=args.scheduler)

    # filter snr and ct
    if args.init:
        print('.. filter snr and ct')
        args.start = args.start or pd.to_datetime(ds.time[0])
        args.end = args.end or (pd.to_datetime(ds.time[-1])+pd.Timedelta('1s'))
        snr = ds.snr.where(
            (
                (ds.time >= args.start.to_datetime64()) &
                (ds.time < args.end.to_datetime64()) &
                (ds.pair.str.contains(args.pair))
            ),
            drop=True,
        )
        if args.debug:
            print(snr)
        ct = ds.ct.where(
            (
                (ds.time >= args.start.to_datetime64()) &
                (ds.time < args.end.to_datetime64())
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
        args.end = pd.to_datetime(ct.time[-1].item())

        # init timelapse
        print('.. init timelapse dataset')
        ds = init_spectrogram_timelapse(
            pair=snr.pair,
            time=ct.time.where(ct >= 0, drop=True),
            freq=args.freq,
        )
    else:
        print('.. update timelapse dataset')
        args.start = args.start or pd.to_datetime(ds.time1[0].item())
        args.end = args.end or (pd.to_datetime(ds.time1[-1].item()) +
                                pd.Timedelta('1s'))
        bw = ds.freq_bw
        ds = ds.drop_vars('freq_bw').where(
            (ds.time1 >= args.start.to_datetime64()) &
            (ds.time2 >= args.start.to_datetime64()) &
            (ds.time1 < args.end.to_datetime64()) &
            (ds.time2 < args.end.to_datetime64()),
            drop=True
        )
        for dim in ('pair', 'time1', 'time2'):
            if dim in bw.dims:
                bw = bw.loc[{dim: bw[dim][0]}].drop_vars(dim)
        ds['freq_bw'] = bw
        ds['status'] = ds['status'].fillna(0).astype(np.byte)

    nc = ncfile('timelapse', args.pair, args.start, args.end)
    print('{:>20} : {}'.format('pair', ds.pair.size))
    print('{:>20} : {}'.format('time', ds.time1.size))
    print('{:>20} : {}'.format('freq', ds.freq.size))
    if args.debug:
        print(ds)

    # process on client
    ds = spectrogram_timelapse_on_client(
        ds, args.root, client,
        merge=args.update,
        chunk=args.chunk,
        sparse=(not args.abundant),
    )

    # to netcdf
    print(f'.. write to "{nc}"')
    xcorr.write(ds, nc, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot?
    if args.plot:
        print('.. plot')
        plotset = dict(col='freq', yincrease=False, size=4, aspect=1)
        ds.cc2.isel(pair=-1).plot(vmin=0, **plotset)
        ds.delta_lag.isel(pair=-1).plot(robust=True, **plotset)
        ds.delta_freq.isel(pair=-1).plot(robust=True, **plotset)
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
