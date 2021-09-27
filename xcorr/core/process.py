r"""

:mod:`core.process` -- Process
==============================

Process an N-D labelled arrays of data.

"""

# Mandatory imports
import xarray as xr
import pandas as pd
import obspy


# Relative imports
from .. import stream, util


__all__ = ['process']


def process(
    dataset: xr.Dataset, client: stream.Client,
    inventory: obspy.Inventory = None,
    retry_missing: bool = False, test_run: bool = False,
    hash_waveforms: bool = True, metadata_hash: str = None,
    verb: int = 1, **kwargs
):
    """
    Process the xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        The `xcorr` N-D labelled set of data arrays to process.

    client : :class:`xcorr.stream.Client`
        The initiated client to the local and remote data archives.

    inventory : :class:`obspy.Inventory`
        The inventory object with instrument responses.

    retry_missing : `bool`, optional
        If `True`, ``x.status`` with flag "-1" are reprocessed, otherwise only
        flag "0". Defaults is `False`.

    test_run : `bool`, optional
        If `True` the function is aborted after the first iteration.
        Default is `False`.

    hash_waveforms : `bool`, optional
        Compute the sha256 hash of the preprocessed waveforms used for each
        correlation step if `True` (default), and ``x`` contains the variable
        ``hash``. Caution: hashing can take some time (~10s per step).

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 1.

    **kwargs :
        Arguments passed to :meth:``client.get_pair_processed_waveforms``.

    """
    # History
    dataset.attrs['history'] += f", Process started @ {pd.to_datetime('now')}"

    # extract and validate preprocess operations
    if isinstance(dataset.pair.preprocess, dict):
        o = dataset.pair.preprocess
        stream.process.check_operations_hash(o, raise_error=True)

    else:
        o = stream.process.operations_to_dict(dataset.pair.preprocess)

    # check lag indices and update if necessary
    util.time.update_lag_indices(dataset.lag)

    # hash?
    hash_waveforms = hash_waveforms and 'hash' in dataset.variables

    # process each pair per time step
    for p in dataset.pair:

        for t in dataset.time:

            # set location
            pt = {'pair': p, 'time': t}

            if verb > 0:
                print(str(p.values), str(t.values)[:19], end=': ')

            # skip processed
            if dataset.status.loc[pt].values != 0:

                if not (
                    retry_missing and
                    dataset.status.loc[pt].values == -1
                ):

                    if verb > 0:
                        print('Has status "{}". Skip.'.format(
                              dataset.status.loc[pt].values))

                    continue

            # waveforms
            if verb > 0:
                print('Waveforms', end='. ')

            st = client.get_pair_processed_waveforms(
                pair=p.values,
                time=t.values,
                operations=o,
                duration=t.window_length,
                inventory=inventory,
                verb=verb-1,
                strict=True,
                **kwargs
            )

            if not isinstance(st, obspy.Stream) or len(st) != 2:

                if verb > 0:
                    print('Missing data. Set status "-1" and skip.')

                dataset.status.loc[pt] = -1

                if test_run:
                    break

                continue

            # track timing offsets
            dataset.pair_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) -
                pd.to_datetime(st[1].stats.starttime.datetime)
            ) / pd.Timedelta('1s')
            dataset.time_offset.loc[pt] = (
                pd.to_datetime(st[0].stats.starttime.datetime) +
                pd.to_timedelta(dataset.time.window_length / 2, unit='s') -
                dataset.time.loc[{'time': t}].values
            ) / pd.Timedelta('1s')

            # hash
            if hash_waveforms:
                if verb > 0:
                    print('Hash', end='. ')
                dataset.hash.loc[pt] = util.hash_Stream(st)

            # cc
            if verb > 0:
                print('CC', end='. ')

            dataset.cc.loc[pt] = util.cc.cc(
                x=st[0].data[:dataset.lag.attrs['npts']],
                y=st[1].data[:dataset.lag.attrs['npts']],
                normalize=dataset.cc.attrs['normalize'] == 1,
                pad=True,
                unbiased=False,  # apply correction for full dataset!
                dtype=dataset.cc.dtype,
            )[dataset.lag.attrs['index_min']:dataset.lag.attrs['index_max']]

            # Status
            dataset.status.loc[pt] = 1

            # Finish
            if verb > 0:
                print('Done.')

            if test_run:
                break

    # update history
    dataset.attrs['history'] += f", Process ended @ {pd.to_datetime('now')}"

    # bias correct?
    if dataset.cc.attrs['bias_correct'] == 1:
        dataset['cc'] = dataset.cc.signal.unbias()
        dataset.attrs['history'] += f", Unbiased CC @ {pd.to_datetime('now')}"

    # update metadata hash
    dataset.attrs['sha256_hash_metadata'] = util.hasher.hash_Dataset(
        dataset, metadata_only=True, debug=False
    )
