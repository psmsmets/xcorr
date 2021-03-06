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
import dask
import distributed

# Relative imports
from ..version import version
from .. import io, core, stream, util


__all__ = ['lazy_process']


@dask.delayed
def single_threaded_process(
    pair: str, time: pd.Timestamp, init_args: dict,
    client: stream.Client, inventory: Inventory,
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
    nc = io.ncfile(pair, time, root)

    # open and update
    if not force_fresh:
        data = io.read(nc, fast=True)
        if data and np.all(data.status.values == 1):
            data.close()
            return True

    # create
    if not data:
        data = core.init(
            pair=pair,
            starttime=time,
            endtime=time + time.freq,
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

    # save
    if data and np.any(data.status.values == 1):
        io.write(data, nc, verb=verb)
        return True

    return False


def lazy_processes(
    pairs: list, times: pd.DatetimeIndex, availability: xr.DataArray,
    preprocessing: xr.DataArray, init_args: dict, verb: int = 0,
    override_availability: bool = False,
    **kwargs
):
    """
    Construct list of lazy single processes for dask.compute

    Parameters
    ----------
    pairs : `list`
        Expects a list of `str`, with receiver couple SEED-id's separated
        by a dash ('-').

    times : `pandas.date_range`
        Date range from start to end with ``freq``='D'.

    availability : :class:`xarray.DataArray`
        Waveform availability status N-D labelled array.

    preprocessing : :class:`xarray.DataArray`
        Waveform preprocessing status N-D labelled array.

    init_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.init`.

    override_availability : `bool`
        If `True`, force processing even when the data availability failed.
        Defautls to `False`. Note: preprocessing should always be passed.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    **kwargs :
        Paramaters passed to :func:`single_threaded_process`.

    Returns
    -------
    lazy_processess : `list`
        List with :class:`dask.delayed` function calls.

    """
    tasks = []

    for pair in pairs:

        # check preprocessing
        pair_preprocessing = preprocessing.loc[{
            'receiver': util.receiver.split_pair(pair, substitute=False,
                                                 to_dict=False),
            'time': preprocessing.time[0],
        }] == 1
        preprocessing_passed = np.all(pair_preprocessing.values == 1)

        # substituted receivers
        receivers = util.receiver.split_pair(pair, substitute=True,
                                             to_dict=False)

        for time in times:

            # pair time
            if verb > 0:
                print('    {} {}'.format(pair, time), end='. ')

            # preprocessing status
            if verb > 2:
                print('Preprocessing:',
                      'ok' if preprocessing_passed else 'failed', end='. ')

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
            ) or override_availability

            # availability status
            if verb > 2:
                print('Availability:',
                      'ok' if availability_passed else 'failed', end='. ')

            # preprocessing and availability passed
            if preprocessing_passed and availability_passed:

                if verb > 0:
                    print('Add.')
                task = single_threaded_process(
                    pair, time, init_args, verb=verb, **kwargs
                )
                tasks.append(task)

            else:

                if verb > 0:
                    print('Skip.')

    if len(tasks) == 0:
        raise RuntimeError('No lazy processes added.')

    return tasks


def lazy_process(
    pairs: list, times: pd.DatetimeIndex, init_args: dict, client_args: dict,
    inventory: Inventory, root: str, force_fresh: bool = False,
    download: bool = True, verb: int = 1,
    dask_client: distributed.Client = None, **kwargs
):
    """
    Lazy process a lot of data with xcorr and dask.

    Parameters
    ----------
    pairs : `list`
        Expects a list of `str`, with receiver couple SEED-id's separated
        by a dash ('-').

    times : `pandas.DatetimeIndex`
        Date range from start to end with ``freq``='D'.

    init_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.init`.

    client_args : `dict`
        Dictionary with input arguments passed to :class:`xcorr.stream.Client`.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    root : `str`
        Path to output base directory. A separate directory will be created
        for each receiver pair.

    force_fresh : `bool`, optional
        If `True`, for a to iniate and process an xcorr data array each step.
        Defaults to `False`.

    download : `bool`, optional
        If `True` (default), download missing data on-the-fly during
        the data availability analysis.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    dask_client : :class:`distributed.Client`, optional
        Specify a Dask client. If `None` (default), the currently active client
        will be used.

    **kwargs :
        Parameters passed to :func:`lazy_processes`.

    """

    # -------------------------------------------------------------------------
    # Dask
    # -------------------------------------------------------------------------

    dask_client = dask_client or distributed.Client()

    # -------------------------------------------------------------------------
    # Print main config parameters
    # -------------------------------------------------------------------------
    if verb > 0:
        print('-'*79)
        print('Config')
        print('    root           :', root)
        print('    force_fresh    :', force_fresh)
        print('    download       :', download)
        print('    xcorr version  :', version)

    # -------------------------------------------------------------------------
    # various inits
    # -------------------------------------------------------------------------

    # activate client parallel locking
    client_args['parallel'] = True

    # init the waveform client
    xcorr_client = stream.Client(**client_args)

    if verb > 0:
        print('-'*79)
        print('Stream Client')
        print(xcorr_client)

    # Read and filter inventory
    inventory = util.receiver.get_pair_inventory(pairs, inventory, times)

    # -------------------------------------------------------------------------
    # Print main data parameters
    # -------------------------------------------------------------------------
    extend_days = 1

    if verb > 0:
        print('-'*79)
        print('Data')
        print('    pairs : {}'.format(len(pairs)))

        for p in pairs:
            print('        {}'.format(p))

        print('    times : {}'.format(len(times)))
        print('        start  : {}'.format(times[0]))
        print('        end    : {}'.format(times[-1]))
        print('        extend : {} day(s)'.format(extend_days))

    # -------------------------------------------------------------------------
    # Evaluate waveform availability (parallel) and try to add missing data
    # -------------------------------------------------------------------------
    if verb > 0:
        print('-'*79)

    availability = xcorr_client.verify_waveform_availability(
        pairs, times, extend_days=extend_days,
        substitute=True,
        parallel=True,
        download=download,
        verb=verb,
    )

    # -------------------------------------------------------------------------
    # Waveform preprocessing (parallel)
    # -------------------------------------------------------------------------
    if verb > 0:
        print('-'*79)

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

    preprocessing = xcorr_client.verify_waveform_processing(
        pairs, time,
        operations=init_args['preprocess'],
        inventory=inventory,
        substitute=True,
        duration=init_args['window_length'],
        sampling_rate=init_args['sampling_rate'],
        download=False,
        parallel=True,
        verb=verb,
    )

    # -------------------------------------------------------------------------
    # Evaluate lazy process list
    # -------------------------------------------------------------------------
    if verb > 0:
        print('-'*79)
        print('Process')
    results = dask_client.compute(
        lazy_processes(
            pairs, times, availability, preprocessing, init_args,
            client=xcorr_client, inventory=inventory, root=root,
            force_fresh=force_fresh, verb=verb-1, **kwargs
        ),
        allow_other_workers=True,
    )
    distributed.wait(results)
    results = dask_client.gather(results)

    completed = sum(results) if len(results) > 0 else 0
    pcnt = 100 * completed / len(results) if len(results) > 0 else 0.
    if verb > 0:
        print('    Completed : {} of {} ({:.1f}%)'
              .format(completed, len(results), pcnt))

    return pcnt == 100.
