r"""

:mod:`core.postprocess` -- xcorr postprocess
============================================

Predefined cc postproccessing

"""


# Mandatory imports
import numpy as np
import pandas as pd
import xarray as xr
import warnings


__all__ = ['postprocess']


def postprocess(
    ds: xr.Dataset, lag_lim: tuple = None, time_lim: tuple = None,
    clim: tuple = None, cmin: float = None, cmax: float = None,
    time_min: float = None, time_max: float = None, lag_min: float = None,
    lag_max: float = None, filter_kwargs: dict = None
):
    """Postprocess an xcorr CCF dataset

    Parameters
    ----------
    ds : :class:`xr.Dataset`
        Xcorr cc dataset.

    cmin : `float`, optional
        Set the lower celerity, in m/s, to set the maximal time lag.

    cmax : `float`, optional
        Set the upper celerity, in m/s, to set the minimal time lag.

    clim : `tuple`, optional
        Set the time lag given a celerity range, in m/s.
        Overrules ``cmin`` and ``cmax``.

    lag_lim : `tuple`, optional
        Set the time lag lower and upper limit.
        Overrules ``lag_min`` and ``lag_max``.

    lag_min : `float`, optional
        Set the lower time lag of interest.

    lag_max : `float`, optional
        Set the upper time lag of interest.

    time_lim : `tuple`, optional
        Set the time lower and upper limit.
        Overrules ``time_min`` and ``time_max``.

    time_min : `np.datetime64`, optional
        Set the lower time of interest.

    time_max : `np.datetime64`, optional
        Set the upper time of interest.

    filter_kwargs : `dict`, optional
        Dictionary of keyword arguments to pass to the filter.
        Defaults to a 2nd order highpass filter with a 3Hz corner frequency.

    Returns
    -------
    ds2 : `xr.Dataset`
        Dataset with ``ds.cc`` extracted for the valid signal window,
        postprocessed and replaced.

    """
    # check
    if 'postprocess' in ds.cc.attrs:
        warnings.warn(f"Dataset already postprocessed on {ds.cc.postprocess}")
        return ds

    # tuple limits given?
    if lag_lim is not None:
        lag_min, lag_max = lag_lim
    if time_lim is not None:
        time_min, time_max = time_lim
    if clim is not None:
        cmin, cmax = clim

    # extract distance and set SI-unit factor
    d = ds.distance
    d_fact = 1000 if (d.units == 'km' and (cmin > 10 and cmax > 10)) else 1

    # time range?
    time_min = time_min or ds.time.min().values
    time_max = time_max or ds.time.max().values
    if (
        not isinstance(time_min, np.datetime64) or
        not isinstance(time_max, np.datetime64)
    ):
        raise TypeError('time min and max should be of type numpy.datetime64')

    # extract valid times only
    m = (ds.status == 1) & (ds.time >= time_min) & (ds.time <= time_max)
    if not m.any():
        raise ValueError("No data after extracting valid times")
    ds = ds.drop_vars('distance').where(m, drop=True)
    ds['distance'] = d  # avoids adding extra dimensions!

    # set filter arguments
    filter_kwargs = {
        'frequency': 3.,
        'btype': 'highpass',
        'order': 2,
        **(filter_kwargs or dict()),
    }

    # extract time_offset and pair_offset
    delay = -(ds.pair_offset + ds.time_offset)

    # time lag range?
    lag_min = lag_min or ds.lag.min().item()
    lag_max = lag_max or ds.lag.max().item()

    # update with celerity range?
    lag_min = max((lag_min, d.min().values*d_fact/cmax)) if cmax else lag_min
    lag_max = min((lag_max, d.max().values*d_fact/cmin)) if cmin else lag_max

    # extract
    m = (ds.lag >= lag_min) & (ds.lag <= lag_max)
    if not m.any():
        raise ValueError("No data after extracting time lag")

    # postprocess
    cc = (
        ds.cc.where(m, drop=True)
        .signal.unbias()
        .signal.demean()
        .signal.taper(max_length=5.)  # timeshift phase wrapping
        .signal.timeshift(delay=delay, dim='lag', fast=True)
        .signal.filter(**filter_kwargs)
        .signal.taper(max_length=3/2)  # filter artefacts
    )
    cc.attrs['postprocess'] = f"{pd.to_datetime('now')}"
    cc.lag.attrs['cmin'] = cmin
    cc.lag.attrs['cmax'] = cmax

    # extract valid time and replace raw with processed cc
    ds = ds.drop_vars(('cc', 'lag', 'status'))
    ds['cc'] = cc

    return ds
