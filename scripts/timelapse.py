"""
TIMELAPSE
===

Crosscorrelation of crosscorrelation spectrograms.


Check distributed.scheduler.KilledWorker for a long time series #236
https://github.com/spencerahill/aospy/issues/236

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import distributed
from glob import glob
import os
import sys
import getopt
import xcorr
# try:
#     import dask_jobqueue
# except ModuleNotFoundError:
#     dask_jobqueue = False


###############################################################################
# Local functions
# ---------------

def get_spectrogram(pair, time, root):
    """Load spectrogram for a pair and time.
    """
    # construct abs path and filename
    nc = xcorr.util.ncfile(pair, time, root)

    # select pair time location
    item = {'pair': pair, 'time': time}

    # set lock
    lock = distributed.Lock(nc)
    lock.acquire()

    # get data from disk
    ds = None
    ok = False
    try:
        # read file
        ds = xcorr.read(nc, fast=True, engine='h5netcdf')

        # success?
        if ds is not None:
            # status okay?
            ok = ds.status.loc[item] == 1
            # extract data
            if ok:
                cc = ds.cc.loc[item].load()
                lag = ds.lag.load()
                d_km = ds.distance.loc[{'pair': pair}].values
                delay = ds.pair_offset.loc[item] + ds.time_offset.loc[item]
            # close
            ds.close()
    except Exception:
        ds = None

    # release lock
    lock.release()

    # no data?
    if ds is None or not ok:
        return

    # no valid data?
    if xr.ufuncs.isnan(cc).any():
        return

    # process cc
    cc = cc.where((lag >= d_km/1.495) & (lag <= d_km/1.465), drop=True)
    cc = xcorr.signal.unbias(cc)
    cc = xcorr.signal.detrend(cc)
    cc = xcorr.signal.filter(cc, frequency=1.5, btype='highpass', order=4)
    cc = xcorr.signal.taper(cc, max_length=2/3.)

    # solve time_offset and pair_offset
    delay = xcorr.util.time.to_seconds(delay.values)
    if delay != 0.:
        cc = xcorr.signal.timeshift(cc, delay=delay, dim='lag', pad=True)

    # spectrogram
    psd = xcorr.signal.spectrogram(cc, duration=2., padding_factor=4)

    return psd


def correlate_spectrograms(obj, root):
    """Correlate spectrograms.
    """
    # already set?
    if obj.status.all():
        return obj

    # test if object is loaded
    if not (obj.freq.any() and obj.pair.any()):
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

                try:
                    # load cc and compute psd on-the-fly
                    psd1 = get_spectrogram(pair, time1, root)
                    psd2 = get_spectrogram(pair, time2, root)

                    if psd1 is None or psd2 is None:
                        continue

                    # correlate per freq range
                    for freq in obj.freq:

                        # set (min, max) frequency
                        bw = obj.freq_bw.loc[{'freq': freq}]
                        fmin = (obj.freq - bw/2).values[0]
                        fmax = (obj.freq + bw/2).values[0]

                        # extract freq
                        in1 = psd1.where(
                            (psd1.freq >= fmin) & (psd1.freq < fmax),
                            drop=True,
                        )
                        in2 = psd2.where(
                            (psd2.freq >= fmin) & (psd2.freq < fmax),
                            drop=True,
                        )

                        # correlate psd's
                        cc = xcorr.signal.correlate2d(in1, in2)

                        # split dims
                        dim1, dim2 = cc.dims[-2:]

                        # get max index
                        amax1, amax2 = np.unravel_index(cc.argmax(), cc.shape)

                        # store values in object
                        item = {
                            'pair': pair,
                            'freq': freq,
                            'time1': time1,
                            'time2': time2,
                        }
                        obj['status'].loc[item] = np.byte(1)
                        obj['cc'].loc[item] = cc.isel({dim1: amax1,
                                                       dim2: amax2})
                        obj[dim1].loc[item] = cc[dim1][amax1]
                        obj[dim2].loc[item] = cc[dim2][amax2]

                except Exception:
                    continue

    return obj


def mask_upper_triangle(ds):
    """Mark upper triangle (one offset)
    """
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        ds.status.loc[{
            'time1': ds.time1[ind1[i]],
            'time2': ds.time2[ind2[i]],
        }] = np.byte(1)


def fill_upper_triangle(ds):
    """Fill upper triangle (one offset)
    """
    ind1, ind2 = np.triu_indices(ds.time1.size, 1)
    for i in range(len(ind1)):
        time1 = ds.time1[ind1[i]]
        time2 = ds.time2[ind2[i]]
        triu = {'time1': time1, 'time2': time2}
        tril = {'time1': time2, 'time2': time1}
        ds.cc.loc[triu] = ds.cc.loc[tril]
        ds.delta_freq.loc[triu] = -ds.delta_freq.loc[tril]
        ds.delta_lag.loc[triu] = -ds.delta_lag.loc[tril]


def init_timelapse(snr, ct, pair, starttime, endtime, freq, root):
    """Init a timelapse dataset.
    """
    # extract times with activity
    time = ct.time.where(ct >= 0, drop=True)

    # new dataset
    ds = xr.Dataset()

    # set global attributes
    nc = xcorr.util.ncfile(snr.pair[0], time[0], os.path.join(root, 'cc'))
    cc = xcorr.read(nc, quick_and_dirty=True)
    ds.attrs = cc.attrs
    cc.close()
    ds.attrs['xcorr_version'] = xcorr.__version__
    ds.attrs['dependencies_version'] = xcorr.core.core.dependencies_version()

    # set coordinates
    ds['pair'] = snr.pair
    ds.pair.attrs = snr.pair.attrs

    ds['freq'] = xr.DataArray(
        np.mean(freq, axis=1),
        dims=('freq'),
        coords=(np.mean(freq, axis=1),),
        attrs={
            'long_name': 'Frequency',
            'standard_name': 'frequency',
            'units': 's-1',
        },
    )

    # set variables
    ds['freq_bw'] = xr.DataArray(
        np.diff(freq, axis=1).squeeze(),
        dims=('freq'),
        coords=(ds.freq,),
        attrs={
            'long_name': 'Frequency bandwidth',
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

    ds['cc'] = ds.status.astype(np.float64) * 0
    ds['cc'].attrs = {
        'long_name': 'Crosscorrelation Estimate',
        'standard_name': 'crosscorrelation_estimate',
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


def correlate_spectrograms_on_client(
    ds: xr.Dataset, client: distributed.Client, root: str,
    n_workers: int = None
):
    """Correlate spectrograms on a Dask client
    """

    # ignore upper diagonal
    mask_upper_triangle(ds)

    # total number of workers
    n_workers = n_workers or sum(client.ncores().values())

    # chunk
    chunk = int(np.floor(ds.time1.size/n_workers))
    ds = ds.chunk({'pair': 1, 'time1': chunk, 'time2': chunk})

    # map
    mapped = ds.map_blocks(
        correlate_spectrograms,
        args=[os.path.join(root, 'cc')],
        template=ds
    )

    # persist to client
    future = client.persist(mapped)

    # load dask to xarray (force await on async)
    result = future.load()

    # fill upper triangle
    fill_upper_triangle(result)

    return result


###############################################################################
# Main functions
# --------------

def help(e=None):
    """
    """
    print('timelapse.py -p <pair> -s <starttime>  -e <endtime> -r <root> '
          '-n <nworkers>')
    raise SystemExit(e)


def main(argv):
    """
    """
    # init args
    root = None
    pair = None
    starttime = None
    endtime = None
    freq = None
    n_workers = None
    plot = False
    debug = False
    scheduler = None
    cluster = None

    try:
        opts, args = getopt.getopt(
            argv,
            'hvp:s:e:f:r:n:c:',
            ['pair=', 'starttime=', 'endtime=', 'frequency=', 'root=',
             'nworkers=', 'help', 'plot', 'debug', 'scheduler=']
        )
    except getopt.GetoptError as e:
        help(e)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
        elif opt in ('-p', '--pair'):
            pair = arg
        elif opt in ('-s', '--starttime'):
            starttime = arg
        elif opt in ('-e', '--endtime'):
            endtime = arg
        elif opt in ('-f', '--frequency'):
            freq = np.array(eval(arg))
            if len(freq.shape) != 2 or freq.shape[-1] != 2:
                raise ValueError('frequency should be a list of tuple-pairs '
                                 'with start and end frequencies. Example: '
                                 '--frequencies="(3., 6.), (6., 12.)"')
        elif opt in ('-r', '--root'):
            root = arg
        elif opt in ('-n', '--nworkers'):
            n_workers = int(arg)
        elif opt in ('--plot'):
            plot = True
        elif opt in ('--debug'):
            debug = True
        elif opt in ('--scheduler'):
            scheduler = arg

    pair = pair or ''
    freq = np.array(((3., 6.), (6., 12.))) if freq is None else freq
    starttime = pd.to_datetime(starttime or '2015-01-15')
    endtime = pd.to_datetime(endtime or '2015-01-18')
    root = os.path.abspath(root) if root is not None else os.getcwd()

    if n_workers is None or n_workers < 1:
        raise ValueError('--nworkers should be defined!')

    if scheduler is not None:
        print('dask scheduler:', scheduler)
        client = distributed.Client(scheduler_file=scheduler)
        print(f'.. waiting for {n_workers} workers', end=', ')
        client.wait_for_workers(n_workers=n_workers)
        print('OK.')
    else:
        cluster = distributed.LocalCluster(
            processes=False, threads_per_worker=1, n_workers=n_workers,
        )
        print('dask LocalCluster:', cluster)
        client = distributed.Client(cluster)

    print('Dask client:', client)
    print('Dask dashboard:', client.dashboard_link)
    print('{:>25} : {}'.format('root', root))
    print('{:>25} : {}'.format('pair', pair))
    print('{:>25} : {}'.format('starttime', starttime))
    print('{:>25} : {}'.format('endtime', endtime))

    # snr
    print('.. load signal-to-noise ratio')
    snr = xr.merge([xr.open_dataarray(f) for f in
                    glob(os.path.join(root, 'snr', 'snr_20??.nc'))]).snr
    snr = snr.where(
        (
            (snr.time >= starttime.to_datetime64()) &
            (snr.time < endtime.to_datetime64()) &
            (snr.pair.str.contains(pair))
        ),
        drop=True,
    )
    if debug:
        print(snr)

    # get confindence triggers
    print('.. get coincidence triggers', end=', ')
    ct = xcorr.signal.coincidence_trigger(
        snr, thr_on=10., extend=0, thr_coincidence_sum=None,
    )
    print(f'periods = {ct.attrs["nperiods"]}')
    if debug:
        print(ct)
    if plot:
        snr.plot.line(x='time', hue='pair', aspect=2.5, size=3.5,
                      add_legend=False)
        xcorr.signal.trigger.plot_trigs(snr, ct)
        plt.tight_layout()
        plt.show()

    # init timelapse
    print('.. init timelapse dataset', end=', ')
    ds = init_timelapse(snr, ct, pair, starttime, endtime, freq,
                        root, n_workers)
    print('dims: pair={pair}, freq={freq}, time={time1}'.format(
        pair=ds.pair.size, freq=ds.freq.size, time1=ds.time1.size,
    ))
    if debug:
        print(ds)

    # create all locks
    print('.. init locks')
    locks = create_locks(ds, os.path.join(root, 'cc'), client)

    # persist to client
    print('.. map and compute blocks')
    ds = correlate_spectrograms_on_client(ds, client, root)

    # update metadata
    print('.. extend dataset with snr and triggers')
    ds['snr'] = snr
    ds['ct'] = ct
    if debug:
        print(ds)

    # to netcdf
    nc = os.path.join(root, 'timelapse', 'timelapse_{}_{}_{}.nc'.format(
        'all' if pair == '' else pair,
        str(ds.time[0].dt.strftime('%Y%j').values),
        str(ds.time[-1].dt.strftime('%Y%j').values),
    ))
    print(f'.. write to "{nc}"')
    xcorr.write(
        data=ds,
        path=nc,
        variable_encoding=dict(zlib=True, complevel=9),
        verb=1 if debug else 0,
    )

    # plot?
    if plot:
        # common plot settings
        plotset = dict(col='freq', yincrease=False, size=4, aspect=1)

        # plot cc
        plt.figure()
        ds.cc.isel(pair=-1).plot(vmin=0, **plotset)
        plt.show()

        # plot delta_lag
        plt.figure()
        ds.delta_lag.isel(pair=-1).plot(robust=True, **plotset)
        plt.show()

        # plot delta_freq
        plt.figure()
        ds.delta_freq.isel(pair=-1).plot(robust=True, **plotset)
        plt.show()

    # cleanup
    print('.. cleanup')
    client.close()
    if cluster is not None:
        cluster.close()

    print('.. done')


if __name__ == "__main__":
    main(sys.argv[1:])
