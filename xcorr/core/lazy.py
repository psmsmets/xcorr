r"""

:mod:`core.lazy` -- Lazy
========================

Lazy processing using `xcorr`.

"""

# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
from obspy import Inventory
from shutil import rmtree
import dask
from dask import distributed

# Relative imports
from ..version import version
from ..core import core
from ..client import Client
from .. import util


__all__ = ['lazy_process']


@dask.delayed
def single_threaded_process(
    pair: str, time: pd.Timestamp, init_args: dict,
    client: Client, inventory: Inventory,
    root: str, force_fresh: bool = False, verb: int = 0,
    **kwargs
):
    r"""Single thread xcorr processing sequence.

    Parameters
    ----------

    Returns
    -------
    processed : `bool`
        `True` if data is processed, otherwise `False`.

    """
    data = None

    # filename
    nc = util.ncfile(pair, time, root)

    if not force_fresh:

        # open
        data = core.read(nc, fast=True)

        # update
        if data and np.all(data.status.values == 1):

            data.close()

            return True

    # create
    if not data:
        data = core.init(
            pair=pair,
            starttime=time,
            endtime=time + pd.offsets.DateOffset(1),
            inventory=inventory,
            **init_args
        )

    # process
    core.process(
        data,
        inventory=inventory,
        client=client,
        retry_missing=True,
        download=False,
        verb=verb,
        **kwargs
    )

    # Save
    if data and np.any(data.status.values == 1):

        core.write(data, nc, verb=verb)

        return True

    return False


def lazy_processes(
    pairs: list, times: pd.DatetimeIndex, availability: xr.DataArray,
    preprocessing: xr.DataArray, init_args: dict, verb: int = 0,
    **kwargs
):
    """
    Construct list of lazy single processes for dask.compute

    Parameters
    ----------
    pairs : `list`
        Expects a list of `str`, with receiver couple SEED-id's separated
        by a dash ('-').

    times : `pandas.data_range`
        Date range from start to end with ``freq``='D'.

    availability : :class:`xarray.DataArray`
        Data availability status N-D labelled array.

    preprocessing : :class:`xarray.DataArray`
        Data preprocessing status N-D labelled array.

    init_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.init`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    **kwargs :
        Paramaters passed to :func:`single_threaded_process`.

    Returns
    -------
    lazy_processess : `list`
        List with :class:`dask.delayed` function calls.

    """
    results = []

    for pair in pairs:

        # check preprocessing
        pair_preprocessing = preprocessing.loc[{
            'receiver': util.receiver.split_pair(pair, substitute=False,
                                                 to_dict=False),
            'time': preprocessing.time[0],
        }] == 1
        preprocessing_passed = np.all(pair_preprocessing.values == 1)
        preprocessing_status = 'passed' if preprocessing_passed else 'failed'

        # substituted receivers
        receivers = util.receiver.split_pair(pair, substitute=True,
                                             to_dict=False)

        for time in times:

            if verb > 0:

                print('    Check {} {}'.format(pair, time), end='. ')

            # preprocessing status
            if verb > 2:

                print('Preprocessing', preprocessing_status)

            # check availability
            start = time - pd.offsets.DateOffset(
                seconds=init_args['window_length']/2, normalize=True
            )
            end = time + pd.offsets.DateOffset(
                seconds=init_args['window_length']*3/2, normalize=True
            )
            pair_availability = availability.loc[{
                'receiver': receivers,
                'time': pd.date_range(start, end, freq='D'),
            }] == 1

            availability_passed = np.any(
                pair_availability.sum(dim='receiver') == len(receivers)
            )

            # availability status
            if verb > 2:

                print('Availability',
                      'passed' if availability_passed else 'failed', end='. ')

            # preprocessing and availability passed
            if preprocessing_passed and availability_passed:

                if verb > 0:

                    print('Add lazy process.')
                result = single_threaded_process(
                    pair, time, init_args, verb=verb, **kwargs
                )
                results.append(result)

            else:

                if verb > 0:

                    print('Skip.')

    return results


def lazy_process(
    pairs: list, times: pd.DatetimeIndex, init_args: dict, client_args: dict,
    inventory: Inventory, root: str, threads: int = None,
    force_fresh: bool = False, download: bool = True,
    debug: bool = False, **kwargs
):
    """
    Lazy process a lot of data with xcorr and dask.

    Parameters
    ----------
    pairs : `list`
        Expects a list of `str`, with receiver couple SEED-id's separated
        by a dash ('-').

    times : `pandas.data_range`
        Date range from start to end with ``freq``='D'.

    init_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.init`.

    client_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.Client`.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    root : `str`
        Path to output base directory. A folder will be created for each pair.

    threads : `int`, optional
        Set the number of threads (if ``debug`` is `False`). Defaults to the
        maximum processors/threads available.

    progressbar : `bool`, optional
        If `True` (default), show a progressbar for each evaluation of lazy
        processes. If ``debug`` is enabled the progressbar is disabled.

    force_fresh : `bool`, optional
        If `True`, for a to iniate and process an xcorr data array each step.
        Defaults to `False`.

    download : `bool`, optional
        If `True` (default), download missing data on-the-fly during
        the data availability analysis.

    debug : `bool`, optional
        If `True`, process single threaded with some extra verbosity. Default
        is `False`.

    **kwargs :
        Parameters passed to :func:`lazy_processes`.

    """

    # -------------------------------------------------------------------------
    # Dask
    # -------------------------------------------------------------------------

    # dask compute arguments
    if debug:

        # allow verbose
        verb = kwargs['verb'] if 'verb' in kwargs else 1

        # single-threaded
        dcluster = distributed.LocalCluster(
            processes=False, scheduler='single-threaded',
        )

    else:

        # disable verbose globally
        verb = 0

        # init local cluster
        dcluster = distributed.LocalCluster(
            processes=False, threads_per_worker=1, n_workers=threads or -1,
        )

    # make sure there is no verb in kwargs
    if 'verb' in kwargs:

        kwargs.pop('verb')

    # dask client from cluster
    dclient = distributed.Client(dcluster)

    # -------------------------------------------------------------------------
    # Print some config parameters
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Config')
    print('    results root   :', root)
    print('    threads        :', 1 if debug else (threads or -1))
    print('    force_fresh    :', force_fresh)
    print('    download       :', download)
    print('    debug          :', debug)
    print('    xcorr version  :', version)
    print('    dask dashboard :', dclient.dashboard_link)

    print(dclient)

    # -------------------------------------------------------------------------
    # various inits
    # -------------------------------------------------------------------------

    # init the waveform client
    xclient = Client(**client_args)

    print(xclient)

    # Read and filter inventory
    inventory = util.receiver.get_pair_inventory(pairs, inventory, times)

    # -------------------------------------------------------------------------
    # Print main data parameters
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Data')
    print('    pairs : {}'.format(len(pairs)))

    extend_days = 1

    for p in pairs:

        print('        {}'.format(p))

    print('    times : {}'.format(len(times)))
    print('        start  : {}'.format(times[0]))
    print('        end    : {}'.format(times[-1]))
    print('        extend : {} day(s)'.format(extend_days))

    # -------------------------------------------------------------------------
    # Evaluate data availability (parallel), and try to download missing data
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Verify availability')

    availability = xclient.init_data_availability(
        pairs, times, extend_days=extend_days,
        substitute=True,
        download=download,
        verb=verb or 1,
    )

    # -------------------------------------------------------------------------
    # Data preprocessing (parallel)
    # -------------------------------------------------------------------------

    # init waveform preprocessing status for a day with max availability
    nofrec = len(availability.receiver)

    for time in availability.time[extend_days:-extend_days]:

        if np.sum(availability.loc[{'time': time}].values == 1) == nofrec:

            break

    else:

        raise RuntimeError(
            'Your pairs contain a receiver without data availability...'
        )

    time = pd.to_datetime(time.values)

    preprocessing = xclient.data_preprocessing(
        pairs, time,
        preprocess=init_args['preprocess'],
        inventory=inventory,
        substitute=True,
        duration=init_args['window_length'],
        sampling_rate=init_args['sampling_rate'],
        download=False,
        compute=False,
        verb=verb or 1,
    )

    # -------------------------------------------------------------------------
    # Evaluate lazy process list
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Process')
    results = dask.compute(
        lazy_processes(
            pairs, times, availability, preprocessing, init_args,
            client=xclient, inventory=inventory, root=root,
            force_fresh=force_fresh, verb=verb, **kwargs
        )
    )
    results = results[0]  # unpack tuple (only one variable returned)
    completed = sum(results)
    pcnt = 100 * completed / len(results) if len(results) > 0 else 0.
    print('    Completed : {} of {} ({:.1f}%)'
          .format(completed, len(results), pcnt))

    # -------------------------------------------------------------------------
    # Dask profiler
    # -------------------------------------------------------------------------
    if not debug:

        dclient.close()
        dcluster.close()
        rmtree('dask-worker-space')
