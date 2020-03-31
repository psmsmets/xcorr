r"""

:mod:`signal.stack` -- Stack
============================

Stack an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
import pandas as pd


# Relative imports
from ..util import get_dpm


__all__ = ['stack_all', 'stack_year_month', 'stack_year_dayofyear']


def list_dim_variables(
    dataset: xr.Dataset, dim: str = 'time'
):
    """
    List all variables in `xr.Dataset` with dimension `dim`.
    """
    if isinstance(dim, str):
        d = dim
    elif isinstance(dim, xr.DataArray):
        d = dim.name
    else:
        raise TypeError('Only xr.Dataset and str are allowed.')
    var = []
    for v in dataset.data_vars:
        if d in dataset[v].dims:
            var.append(v)
    return var


def stack_all(
    dataset: xr.Dataset, dim: xr.DataArray = None, **kwargs
):
    """
    Stack `xr.Dataset` over the dimension `dim` (default = time).
    """
    ds = dataset.mean(
        dim='time' if dim is None else dim.name,
        keep_attrs=True,
        **kwargs
    )
    return ds.assign_coords(dim or {'time': dataset.time[0]})


def stack_year_month(
    dataset: xr.Dataset, **kwargs
):
    """
    Stack `xr.Dataset` per year and month.
    """
    year_month_idx = pd.MultiIndex.from_arrays(
        [dataset['time.year'], dataset['time.month']]
    )
    dataset.coords['year_month'] = ('time', year_month_idx)
    month_length = xr.DataArray(
        get_dpm(dataset.time.to_index()),
        coords=[dataset.time],
        name='month_length'
    )
    weights = (
        (month_length.groupby('time.month') / month_length)
        .groupby('time.month')
        .sum()
    )
    return (
        (dataset[list_dim_variables(dataset, **kwargs)] * weights)
        .groupby('year_month')
        .sum(dim='time', keep_attrs=True)
    )


def stack_year_dayofyear(
    dataset: xr.Dataset, **kwargs
):
    """
    Stack `xr.Dataset` per year and doy.
    """
    year_doy_idx = pd.MultiIndex.from_arrays(
        [dataset['time.year'], dataset['time.dayofyear']]
    )
    dataset.coords['year_dayofyear'] = ('time', year_doy_idx)
    return (
        dataset[list_dim_variables(dataset, **kwargs)]
        .groupby('year_dayofyear')
        .mean(dim='time', keep_attrs=True)
    )
