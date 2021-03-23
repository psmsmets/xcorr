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
import dask
import distributed
import os
import sys
import gc
import argparse

# relative imports
import xcorr
from ..scripts import helpers
from ..scripts.helpers import logging as log
from ..scripts.helpers import logging_arg_info as arg_info
from ..scripts.helpers import logging_arg_debug as arg_debug

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
        np.zeros((ds.pair.size, ds.freq.size, time.size, time.size),
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
    f = os.path.split(nc)[-1]

    # exists?
    if not os.path.isfile(nc):
        raise FileNotFoundError("cc dataset not found in" + f)

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
        pass

    # release lock
    try:
        if lock.locked():
            lock.release()
    except Exception:
        pass

    # no data?
    if ds is None or not ok:
        raise ValueError("cc trace load failed from" + f)

    # extract cc
    cc = ds.cc.where(
        (ds.lag >= ds.distance/1.50) &
        (ds.lag <= ds.distance/1.46),
        drop=True,
    )

    # no valid data?
    if xr.ufuncs.isnan(cc).any():
        raise ValueError("cc trace contains NaN in" + f)

    # extract time_offset and pair_offset
    delay = -(ds.pair_offset + ds.time_offset)

    # process cc
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.demean(cc)
    cc = xcorr.signal.taper(cc, max_length=5.)  # timeshift phase wrapping
    cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', fast=True)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=3/2)  # filter artefacts

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2.5, padding_factor=4)

    # clean
    del cc
    del ds

    return psd


def correlate_spectrogram(obj, loc, root):
    """Correlate spectrogram.
    """
    # already done?
    if (obj.status.loc[loc] == 1).all():
        return

    # no need to process?
    if xr.ufuncs.isnan(obj.status.loc[loc]).all():
        return

    # load cc and compute psd on-the-fly
    psd1, psd2 = None, None
    try:
        psd1 = get_spectrogram(loc['pair'], loc['time1'], root)
        psd2 = get_spectrogram(loc['pair'], loc['time2'], root)
    except Exception as e:
        print("Spectrogram computation failed:", e)
        return

    if psd1 is None or psd2 is None:
        print("Spectrogram computation returned None")
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
        locf = {**loc, 'freq': freq}
        obj['status'].loc[locf] = np.byte(1)
        obj['cc2'].loc[locf] = cc2.isel({dim1: amax1, dim2: amax2})
        obj[dim1].loc[locf] = cc2[dim1][amax1]
        obj[dim2].loc[locf] = cc2[dim2][amax2]

    # clean
    del psd1
    del psd2
    del cc2

    return


def correlate_spectrograms(obj, root):
    """Correlate spectrograms.
    """

    # complete or no need to process obj?
    if (obj.status == 1).all() or xr.ufuncs.isnan(obj.status).all():
        return obj

    # catch memory error
    obj = obj.copy(True)

    # process per item
    for time2 in obj.time2:
        for time1 in obj.time1:
            for pair in obj.pair:
                loc = dict(pair=pair, time1=time1, time2=time2)
                correlate_spectrogram(obj, loc, root)

    # clean
    gc.collect()

    return obj


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


def plot_timelapse(ds, data_vars: list = [], **kwargs):
    """Plot timelapse dataset
    """
    plotset = {
        'row': 'freq',
        'col': 'pair',
        'yincrease': False,
        'size': 3,
        'aspect': 1,
        'cbar_kwargs': dict(shrink=1/ds.freq.size),
        **kwargs,
    }

    data_vars = data_vars or ['status', 'cc2', 'delta_freq', 'delta_lag']

    gs = []

    if 'status' in data_vars:
        gs.append(ds.status.plot(**plotset))
    if 'cc2' in data_vars:
        gs.append(ds.cc2.plot(vmin=0, vmax=1, **plotset))
    if 'delta_lag' in data_vars:
        gs.append(ds.delta_lag.plot(robust=True, **plotset))
    if 'delta_freq' in data_vars:
        gs.append(ds.delta_freq.plot(robust=True, **plotset))

    for g in gs:
        [axis.set_aspect('equal') for axis in g.axes.ravel()]

    plt.show()

    return gs


def mask_upper_triangle(ds):
    """In-place mask the upper diagonal status (one off-diagonal).
    """
    mask = xcorr.signal.tri_mask(ds.time1, ds.time2, 1)
    ds['status'] = ds.status.where(mask, np.byte(1))


def fill_upper_triangle(ds):
    """In-place fill the upper diagonal (one off-diagonal).
    """
    mask = xcorr.signal.tri_mask(ds.time1, ds.time2, 1)
    ds['status'] = xcorr.signal.tri_mirror(ds.status, mask, False)
    ds['cc2'] = xcorr.signal.tri_mirror(ds.cc2, mask, False)
    ds['delta_freq'] = xcorr.signal.tri_mirror(ds.delta_freq, mask, True)
    ds['delta_lag'] = xcorr.signal.tri_mirror(ds.delta_lag, mask, True)


def spectrogram_timelapse_on_client(
    ds: xr.Dataset, client: distributed.Client, root: str,
    chunk: int = 10, debug: bool = False, plot: bool = False,
):
    """2-d correlate spectrograms on a Dask client

    Parameters:
    -----------

    ds: :class:`xr.Dataset`
        Timelapse dataset.

    client: :class:`distributed.Client`
        Dask distributed client object.

    root: `str`
        Cross-correlation root directory

    chunk: `int`, optional
        Dask map blocks chunk size for time1 and time2 (default: 10).

    debug: `bool`, optional
        Verbose more (defaults to `False`).

    plot: `bool`, optional
        Plot results (defaults to `False`).
    """

    # ignore upper triangle
    log.info("├── mask upper triangle")
    mask_upper_triangle(ds)

    # set all False to 1 (status completed)

    # extract unprocessed
    log.info("├── extract status != 1")
    new = ds.drop_vars('freq_bw').where((ds.status != 1), drop=True)
    new['freq_bw'] = ds.freq_bw

    # plot?
    if debug and plot:
        log.debug("├── plot extracted timelapse dataset")
        plot_timelapse(new)

    # create locks cc ncfiles locks
    log.info("├── create locks cc files")
    ncfiles = _all_ncfiles(new.pair, new.time1, root)
    locks = [distributed.Lock(nc) for nc in ncfiles]
    arg_debug('locks', len(locks))

    # chunk
    log.info("├── chunk")
    new = new.chunk({'pair': 1, 'time1': chunk, 'time2': chunk})
    arg_debug('pair', 1)
    arg_debug('time1', chunk)
    arg_debug('time2', chunk)

    # map blocks and persist
    log.info("├── map and persist blocks")
    mapped = new.map_blocks(
        correlate_spectrograms,
        args=[root],
        template=new,
    )
    new = client.persist(mapped)

    # await async
    log.info("├── await async")
    distributed.wait(new)

    # gather blocks
    log.info("├── gather blocks")
    new = client.gather(new)

    # load results from Dask-array
    # print('.. load results')
    # new.load()

    # plot?
    if debug and plot:
        log.info("├── plot processed dataset")
        plot_timelapse(new)

    # merge
    log.info("├── merge results")
    ds = new.combine_first(ds)

    # fill upper triangle
    log.info("├── fill mirrored upper triangle")
    fill_upper_triangle(ds)

    # load results from Dask-arrays
    log.info("└── load results")
    ds.load()

    return ds


###############################################################################
# Main function
# -------------

def main():
    """Main script function.
    """
    # arguments
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
        '-f', '--frequency', metavar='..', type=str,
        default="(3., 6.), (6., 12.)",
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
        '-c', '--chunk', metavar='..', type=int, default=10,
        help=('Set dask chunks for time dimension (default: 10)')
    )
    helpers.add_common_arguments(parser)
    helpers.add_attrs_group(parser)

    # parse arguments
    args = parser.parse_args()

    # init logging
    helpers.init_logging(args.debug)

    # header
    log.info(f"xcorr-timelapse v{xcorr.__version__}")

    # init dask cluster and client
    cluster, client = helpers.init_dask(
         n_workers=args.nworkers, scheduler_file=args.scheduler, logger=True
    )

    # open and merge paths
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(
            paths=args.paths,
            combine='by_coords',
            data_vars='minimal',
            join='outer',
        )
        ds.load()
        ds.close()
    log.debug(ds)

    # update arguments
    if args.start:
        args.start = pd.to_datetime(args.start, format=args.format)
    if args.end:
        args.end = pd.to_datetime(args.end, format=args.format)
    if args.init:
        args.frequency = np.array(eval(args.frequency))
    else:
        args.frequency = np.array([ds.freq.values - ds.freq_bw.values/2,
                                   ds.freq.values + ds.freq_bw.values/2])
    args.attrs = helpers.parse_attrs_group(args)

    # print core parameters
    log.info("Parameters:")
    arg_info('action', 'update' if args.update else 'init')
    arg_info('root', args.root)
    arg_info('pair', 'all' if args.pair in ('*', '') else args.pair)
    arg_info('start', args.start)
    arg_info('end', args.end)
    arg_info('frequency', ', '.join([f'{f[0]}-{f[1]}'
                                     for f in args.frequency]))

    # init timelapse dataset
    if args.init:
        # load snr and ct to init
        log.info("Filter snr and ct")
        args.start = args.start or pd.to_datetime(ds.time[0])
        args.end = args.end or (pd.to_datetime(ds.time[-1])+pd.Timedelta('1s'))

        # extract signal-to-noise ratio
        snr = ds.snr.where(
            (
                (ds.time >= args.start.to_datetime64()) &
                (ds.time < args.end.to_datetime64()) &
                (ds.pair.str.contains(args.pair))
            ),
            drop=True,
        )
        log.debug(snr)

        # extract coincidence triggers
        ct = ds.ct.where(
            (
                (ds.time >= args.start.to_datetime64()) &
                (ds.time < args.end.to_datetime64())
            ),
            drop=True,
        )
        log.debug(ct)

        # plot extracted snr and coincidence triggers
        if args.plot:
            log.info("Plot snr and coincidence triggers")
            snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                          add_legend=False)
            xcorr.signal.trigger.plot_trigs(snr, ct)
            plt.tight_layout()
            plt.show()

        # set end time
        args.end = pd.to_datetime(ct.time[-1].item())

        # init time lapse dataset
        log.info("Init time lapse dataset")
        ds = init_spectrogram_timelapse(
            pair=snr.pair,
            time=ct.time.where(ct >= 0, drop=True),
            freq=args.frequency,
            **args.attrs,
        )

    else:
        # plot merged dataset
        if args.plot:
            log.info('Plot original dataset')
            plot_timelapse(ds)

        # update timelapse dataset
        log.info("Update merged dataset")

        # set start and end times
        args.start = args.start or pd.to_datetime(ds.time1[0].item())
        args.end = args.end or (pd.to_datetime(ds.time1[-1].item()) +
                                pd.Timedelta('1s'))

        # trim merged dataset to start and end times
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

        # empty status to zero
        ds['status'] = ds['status'].fillna(0).astype(np.byte)

    # set output file
    args.out = helpers.ncfile('timelapse', args.pair, args.start, args.end,
                              args.prefix, args.suffix)

    # log
    log.info("Time lapse dimensions:")
    arg_info('pair', ds.pair.size)
    arg_info('time', ds.time1.size)
    arg_info('freq', ds.freq.size)
    log.info("Prepare output:")
    arg_info('outfile', args.out)
    arg_info('overwrite', args.overwrite)
    log.debug(ds)

    # check if output file exists
    if os.path.exists(args.out) and not args.overwrite:
        log.critical("Output file already exists and overwrite is False.")
        raise SystemExit()

    # process on client
    log.info("Start spectrogram time lapse:")
    ds = spectrogram_timelapse_on_client(ds, client, args.root,
                                         args.chunk, args.debug, args.plot)
    log.info("Spectrogram time lapse completed")

    # to netcdf
    log.info(f"Write to '{args.out}'")
    xcorr.write(ds, args.out, variable_encoding=dict(zlib=True, complevel=9),
                verb=1 if args.debug else 0)

    # plot?
    if args.plot:
        log.info("Plot final dataset")
        plot_timelapse(ds)

    # close dask client and cluster
    log.info("Close Dask client")
    client.close()
    if cluster is not None:
        cluster.close()

    log.info("Done")
    sys.exit(0)


if __name__ == "__main__":
    main()
