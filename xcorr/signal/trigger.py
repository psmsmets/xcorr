r"""

:mod:`signal.trigger` -- Trigger
================================

Estimate the coincidence trigger of an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal import trigger


# Relative imports
from ..util.time import to_datetime, to_seconds
from ..util.convert import to_stream
from ..util.history import historicize


__all__ = ['coincidence_trigger', 'trigger_periods', 'trigger_values',
           'plot_trigs']


def coincidence_trigger(
    x: xr.DataArray, thr_on: float = None, thr_off: float = None,
    thr_coincidence_sum: int = None, similarity_threshold: float = None,
    extend: int = None, dim: str = None
):
    """
    Compute the triggered periods of an N-D labelled array of data with
    precomputed characteristic functions such as, for example, the
    signal-to-noise ratio.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data with the precomputed characteristic functions to
        compute the coincidence trigger for.

    thr_on : `float`, optional
        Threshold for switching single element trigger on.
        Defaults to 10.

    thr_off : `float`, optional
        Threshold for switching single element trigger off.
        Defaults to ``thr_on``.

    thr_coincidence_sum : `int`, optional
        Threshold for coincidence sum. The network coincidence sum has to be
        at least equal to this value for a single trigger to be included in
        the returned trigger list.
        Defaults to the total non-time-dimensional number of elements.

    similarity_threshold : `float`, optional
        Similarity threshold (0.0-1.0) at which a single element trigger gets
        included in the combined trigger list.
        Defaults to 0.7.

    extend : `int`, optional
        Extend the triggered period with ``extend`` time steps.
        Defaults to 0.

    dim : `str`, optional
        The time coordinates name of ``x`` to compute the trigger for.
        Defaults to the last dimension of ``x``.

    Returns
    -------
    trigs : :class:`xarray.DataArray`
        The triggers of ``x``.

    """

    # check time dimension
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # check attributes
    if 'window_length' not in x[dim].attrs:
        raise ValueError(f'Coordinate {dim} has no expected time attribute '
                         '"window_length".')

    if 'window_overlap' not in x[dim].attrs:
        raise ValueError(f'Coordinate {dim} has no expected time attribute '
                         '"window_overlap".')

    # window step
    win_step = (
        x[dim].attrs['window_length'] * (1 - x[dim].attrs['window_overlap'])
    )
    min_step = 3 * win_step

    # trigger params
    thr_on = thr_on or 10.
    thr_off = thr_off or thr_on

    if not isinstance(thr_on, float):
        raise TypeError('On threshold should be of type float.')

    if not isinstance(thr_off, float):
        raise TypeError('Off threshold should be of type float.')

    # similarity_threshold
    similarity_threshold = similarity_threshold or .7

    if (
        not isinstance(similarity_threshold, float) or
        similarity_threshold < 0 or
        similarity_threshold > 1
    ):
        raise TypeError('Similarity threshold should be of type float '
                        'within (0.0-1.0).')

    # extend triggered window
    extend = extend or 0

    if not isinstance(extend, int) or extend < 0:
        raise TypeError('extend should be a postive integer.')

    # convert dataarray to stream
    st = to_stream(x)

    # replace gaps with -1
    for tr in st:
        tr.data = tr.data.filled()

    # thr coincidence sum
    thr_coincidence_sum = thr_coincidence_sum or len(st)

    if (
        not isinstance(thr_coincidence_sum, int) or
        thr_coincidence_sum < 1 or
        thr_coincidence_sum > len(st)
    ):
        raise TypeError('Threshold coincidence sum should be of type int '
                        'within (1 to number of elements).')

    # get coincidence triggers
    trigs = trigger.coincidence_trigger(
        None, thr_on, thr_off, st, thr_coincidence_sum,
        similarity_threshold=similarity_threshold,
    )

    # init ct dataarray to store periods
    ct = x[dim].astype(np.int)
    ct.name = f'ct_{x.name}'
    ct.attrs = {
        'long_name': f'Coincidence trigger for {x.long_name}',
        'standard_name': f'coincidence_trigger_{x.standard_name}',
        'units': '-',
    }

    ct_index = -1
    ct.values[:] = ct_index

    # fill ct with periods
    for trig in trigs:

        if trig['duration'] < min_step:
            continue

        # get period start and end
        start = to_datetime(trig['time'] - extend * win_step).to_datetime64()
        end = to_datetime(
            trig['time'] + trig['duration'] + extend * win_step
        ).to_datetime64()

        # check existing periods
        period = (ct[dim] >= start) & (ct[dim] <= end)

        if np.all(ct[period] == -1):
            ct_index += 1

        ct[period] = ct_index

    # add number of periods
    ct.attrs['nperiods'] = ct_index + 1

    # mask non-periods
    ct = ct.where(ct > -1)

    historicize(ct, f='coincidence_trigger', a={
        'x': x.name,
        'thr_on': thr_on,
        'thr_off': thr_off,
        'thr_coincidence_sum': thr_coincidence_sum,
        'similarity_threshold': similarity_threshold,
        'extend': extend,
        'dim': dim,
    })

    return ct


def trigger_periods(trigs: xr.DataArray):
    """
    Extract the triggered periods into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    trigs : :class:`array.DataArray`
        The triggers of ``x``.

    Returns
    -------
    periods : :class:`pandas.DataFrame`
        The start and end times for each trigger.

    """

    periods = []

    for (index, period) in trigs.groupby(trigs):
        start = period.time[0].values
        end = period.time[-1].values
        periods.append(
            pd.DataFrame(
                data={
                    'start': [start],
                    'end': [end],
                    'days': [to_seconds(end-start)/86400.],
                },
                index=[int(index)]
            )
        )

    return pd.concat(periods)


def trigger_values(x: xr.DataArray, trigs: xr.DataArray):
    """
    Extract the triggered values into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data with the precomputed characteristic functions to
        compute the coincidence trigger for.

    trigs : :class:`array.DataArray`
        The triggers of ``x``.

    Returns
    -------
    values : :class:`pandas.DataFrame`
        Triggered values for each coordinate in ``x``.

    """

    val = []

    for i in range(trigs.max(skipna=True).astype(int).item()):
        trig = trigs.time.where(trigs == i, drop=True)
        tmp = x.sel(time=trig).to_dataframe()
        tmp['period'] = i
        val.append(tmp)

    return pd.concat(val).reset_index()


def plot_trigs(x: xr.DataArray, trigs: xr.DataArray, ax: plt.Axes = None, ylim=None):
    """
    Plot the triggered periods and values.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data with the precomputed characteristic functions to
        compute the coincidence trigger for.

    trigs : :class:`array.DataArray`
        The triggers of ``x``.

    ax : :class:`matplotlib.axes.Axes`
        Specify the axes to plot into. If `None` (default) the current active
        axes is used.

    """
    ax = ax or plt.gca()
    ymin, ymax = ylim or (0., x.max(skipna=True).item())

    imin = trigs.min(skipna=True).astype(int).item()
    imax = trigs.max(skipna=True).astype(int).item()

    for i in range(imin, imax+1):
        sel = trigs == i
        if sel.any():
            ax.fill_between(
                trigs.time.where(sel, drop=True).values, ymin, ymax,
                alpha=0.2, color='black'
            )
