r"""

:mod:`core.lazy` -- Lazy
========================

Lazy processing using `xcorr`.

"""

# Mandatory imports
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from obspy import Inventory
import dask
from dask.diagnostics import ProgressBar


# Relative imports
from ..core import core
from ..clients.client import Client
from ..util import ncfile, split_pair, get_pair_inventory


__all__ = ['lazy_process']


@dask.delayed
def single_threaded_process(
    pair: str, time: pd.Timestamp, init_args: dict,
    client: Client, inventory: Inventory,
    root: str, force_fresh: bool = False, verb: int = 0,
    **kwargs
):
    r"""Single thread xcorr processing sequence.

    Parameters:
    ----------

    Returns:
    -------
    processed : `bool`
        `True` if data is processed, otherwise `False`.
    """
    data = None

    # filename
    nc = ncfile(pair, time, root)

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
    r"""Construct list of lazy single processes for dask.compute

    Parameters:
    -----------
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

    Returns:
    -------
    lazy_processess : `list`
        List with :class:`dask.delayed` function calls.
    """
    results = []

    for pair in pairs:

        # check preprocessing
        pair_preprocessing = preprocessing.loc[{
            'receiver': split_pair(pair, substitute=False, to_dict=False),
            'time': preprocessing.time[0],
        }] == 1
        preprocessing_passed = np.all(pair_preprocessing.values == 1)
        preprocessing_status = 'passed' if preprocessing_passed else 'failed'

        # substituted receivers
        receivers = split_pair(pair, substitute=True, to_dict=False)

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
    progressbar: bool = True, force_fresh: bool = False, debug: bool = False,
    **kwargs
):
    r'''Lazy process a lot of data with xcorr and dask.

    Parameters:
    -----------
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

    debug : `bool`, optional
        If `True`, process single threaded with some extra verbosity. Default
        is `False`.

    **kwargs :
        Parameters passed to :func:`lazy_processes`.
    '''
    # -------------------------------------------------------------------------
    # Print some config parameters
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Config')
    print('    results root :', root)
    print('    threads      :', 1 if debug else (threads or -1))
    print('    debug        :', debug)
    print('    force_fresh  :', force_fresh)

    # -------------------------------------------------------------------------
    # various inits
    # -------------------------------------------------------------------------

    # init the waveform client
    client = Client(**client_args)

    # Read and filter inventory
    inventory = get_pair_inventory(pairs, inventory, times)

    # minimize output to stdout in dask!
    warnings.filterwarnings("ignore")

    if debug:
        verb = 1
        compute_args = dict(scheduler='single-threaded')
    else:
        verb = 0
        compute_args = dict()
        if isinstance(threads, int):
            compute_args['num_workers'] = threads
        if progressbar:
            pbar = ProgressBar()
            pbar.register()

    # -------------------------------------------------------------------------
    # status inits
    # -------------------------------------------------------------------------
    # waveform availability status
    availability = client.init_data_availability(
        pairs, times, extend_days=1, substitute=True
    )
    lazy_availability = client.verify_data_availability(
        availability,
        download=True,
        compute=False
    )

    # waveform preprocessing status
    preprocessing = client.init_data_preprocessing(
        pairs, times[0],
        preprocess=init_args['preprocess'], substitute=True
    )
    lazy_preprocessing = client.verify_data_preprocessing(
        preprocessing,
        inventory=inventory,
        duration=init_args['window_length'],
        sampling_rate=init_args['sampling_rate'],
        download=False,
        compute=False
    )

    # -------------------------------------------------------------------------
    # Print main data parameters
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Data')
    print('    pairs : {}'.format(len(pairs)))
    for p in pairs:
        print('        {}'.format(p))
    print('    times : {} ({})'.format(len(times), len(availability.time)))
    print('        start : {}'.format(str(times[0])))
    print('        end   : {}'.format(str(times[-1])))

    # -------------------------------------------------------------------------
    # Evaluate data availability (parallel), and try to download missing data
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Verify availability')
    verified = lazy_availability.compute(**compute_args)
    pcnt = 100 * verified / availability.size
    print('    Verified : {} of {} ({:.1f}%)'
          .format(verified, availability.size, pcnt))
    print('    Overall availability : {:.2f}%'
          .format(100 * np.sum(availability.values == 1) / availability.size))
    print('    Receiver availability')
    for rec in availability.receiver:
        pcnt = 100*np.sum(
            availability.loc[{'receiver': rec}].values == 1
        ) / availability.time.size
        print('        {} : {:.2f}%'.format(rec.values, pcnt))

    # -------------------------------------------------------------------------
    # Evaluate data preprocessing (parallel)
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Verify preprocessing')
    verified = lazy_preprocessing.compute(**compute_args)
    pcnt = 100 * verified / preprocessing.size
    print('    Verified : {} of {} ({:.1f}%)'
          .format(verified, preprocessing.size, pcnt))
    print('    Overall preprocessing : {:.2f}% passed'
          .format(100*np.sum(preprocessing.values == 1) / preprocessing.size))
    print('    Receiver preprocessing')
    for rec in preprocessing.receiver:
        passed = np.all(preprocessing.loc[{'receiver': rec}].values == 1)
        print('        {} :'.format(rec.values),
              'passed' if passed else 'failed')

    # -------------------------------------------------------------------------
    # Evaluate lazy process list
    # -------------------------------------------------------------------------
    print('-'*79)
    print('Process')
    results = dask.compute(
        lazy_processes(
            pairs, times, availability, preprocessing, init_args,
            client=client, inventory=inventory, root=root,
            force_fresh=force_fresh, verb=verb, **kwargs
        ),
        **compute_args,
    )
    results = results[0]  # unpack tuple (only one variable returned)
    completed = sum(results)
    pcnt = 100 * completed / len(results) if len(results) > 0 else 0.
    print('    Completed : {} of {} ({:.1f}%)'
          .format(completed, len(results), pcnt))
