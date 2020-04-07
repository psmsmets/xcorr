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


__all__ = ['stack']

_grouping = ['all', 'year_month', 'year_doy']


def stack(
    x: xr.Dataset, group: str = None, dim: str = None, **kwargs
):
    r"""Stack a an N-D labeled array of data over a (grouped) dimension.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be stacked.

    group : {'all', 'year_month', 'year_doy'}, optional
       The method to group ``dim``.

    dim : `str`, optional
        The coordinates name of ``x`` to be stacked over. Default is 'time'.

    Returns:
    --------
    y : :class:`xarray.DataArray` or `None`
        The stacked output of ``x``.
    """
    group = group or 'all'
    assert group in _grouping, f'"{group}" is not a group method!'

    return eval(f'stack_{group}(x, dim=dim, **kwargs)')


def list_dim_variables(
    x: xr.Dataset, dim: str = None
):
    """
    List all variables in `xr.Dataset` with dimension `dim`.
    """
    dim = dim or 'time'
    if isinstance(dim, xr.DataArray):
        dim = dim.name
    elif not isinstance(dim, str):
        raise TypeError('Only xr.Dataset and str are allowed.')
    var = []
    for v in x.data_vars:
        if dim in x[v].dims:
            var.append(v)
    return var


def stack_all(
    x: xr.Dataset, coord: xr.DataArray = None, dim: str = None,
    **kwargs
):
    """
    Stack `xr.Dataset` over the dimension `dim` (default = time).
    """
    dim = dim or 'time'
    assert isinstance(dim, str), '"dim" should be of type `str`'
    assert coord is None or isinstance(coord, xr.DataArray), (
        '"coord" should be of type :class:`xarray.DataArray`'
    )
    if coord:
        assert coord.name in x.dims, f'"{coord}" is not a coordinate of "x"!'
        dim = coord.name
    else:
        assert dim in x.dims, f'"{dim}" is not a dimension of "x"!'
        coord = x[dim]
    y = x.mean(dim=dim, keep_attrs=True, **kwargs)
    return y.assign_coords({dim: coord[0]})


def stack_year_month(
    x: xr.Dataset, dim: str = None
):
    """
    Stack `xr.Dataset` per year and month.
    """
    dim = dim or 'time'
    assert isinstance(dim, str), '"dim" should be of type `str`'
    assert dim in x.dims, f'"{dim}" is not a dimension of "x"!'

    # extract all variables depending on dim
    y = x[list_dim_variables(x, dim=dim)]

    # construct multi-index
    year = dim + '.year'
    month = dim + '.month'
    year_month_idx = pd.MultiIndex.from_arrays([y[year], y[month]])

    # add new coordinate
    y.coords['year_month'] = (dim, year_month_idx)

    # month weight (correct for varying number of days)
    month_length = xr.DataArray(
        get_dpm(y[dim].to_index()),
        coords=[y[dim]],
        name='month_length'
    )
    w = (month_length.groupby(month)/month_length).groupby(month).sum()

    # apply weight and group
    y = (w*y).groupby('year_month').sum(dim=dim, keep_attrs=True)

    return y


def stack_year_doy(
    x: xr.Dataset, dim: str = None
):
    """
    Stack `xr.Dataset` per year and doy (day of year).
    """
    dim = dim or 'time'
    assert isinstance(dim, str), '"dim" should be of type `str`'
    assert dim in x.dims, f'"{dim}" is not a dimension of "x"!'

    # extract all variables depending on dim
    y = x[list_dim_variables(x, dim=dim)]

    # construct multi-index
    year = dim + '.year'
    doy = dim + '.doyofyear'
    year_doy_idx = pd.MultiIndex.from_arrays([y[year], y[doy]])

    # add new coordinate
    y['year_dayofyear'] = (dim, year_doy_idx)

    # apply weight and group
    y = y.groupby('year_dayofyear').mean(dim=dim, keep_attrs=True)

    return y
