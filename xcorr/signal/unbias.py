"""

:mod:`signal.unbias` -- Unbias
==============================

Bias correct an N-D labelled cc array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
from warnings import warn
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..cc import weight
from ..util.time import update_lag_indices
from ..util.history import historicize


__all__ = ['unbias']


def unbias(
    x: xr.DataArray, w: xr.DataArray = None, dim: str = None, name: str = None
):
    """
    Bias correct the xcorr N-D labelled cc array of data.

    Parameters
    ----------
    x: :class:`xarray.Dataset`
        The biased cc data array.

    w :class:`xarray.Dataset`, optional
        The unbiased cc weight data array.

    dim : `str`, optional
        The dimension to unbias. Defaults to the last dimension of ``x``.

    name : `str`, optional
        Unbiased cc variable name. Defaults to 'x.name'.

    Returns
    -------
    y: :class:`xarray.Dataset`
        The unbiased cc data array.

    """

    # check x
    dim = dim or x.dims[-1]
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')
    if 'unbiased' not in x.attrs:
        raise ValueError('x has no unbiased flag attribute!')

    if x.unbiased != 0:
        warn('No need to bias correct again.')
        return x

    # get weight vector
    wv = get_weights(x[dim]) if w is None else w

    # check w
    if dim not in wv.dims:
        raise ValueError(f'w has no dimension "{dim}"!')
    if len(wv.coords) != 1:
        raise ValueError('w should have a single dimension!')

    # mask wv for safety
    wv = wv.where((wv[dim] >= x[dim][0]) & (wv[dim] <= x[dim][-1]), drop=True)

    # apply weight to x
    y = x * wv.astype(x.dtype)

    # transfer name and attributes
    y.name = name or x.name
    y.attrs = x.attrs

    # update attributes
    y.attrs['unbiased'] = np.byte(True)
    y.attrs['long_name'] = 'Unbiased ' + x.attrs['long_name']
    y.attrs['standard_name'] = 'unbiased_' + x.attrs['standard_name']

    # log workflow
    historicize(y, f='unbias', a={
        'x': x.name,
        'w': w,
        'dim': dim,
    })

    return y


def get_weights(
    lag: xr.DataArray, name: str = 'w'
):
    """Construct the unbiased crosscorrelation weight vector from the lag vector.

    Parameters
    ----------
    lag: :class:`xarray.DataArray`
        The lag coordinate.

    name : `str`, optional
        Weight variable name. Defaults to 'w'.

    Returns
    -------
       w : :class:`DataArray`
           Unbiased crosscorrelation weight vector.

    """

    update_lag_indices(lag)

    for attr in ['sampling_rate', 'delta', 'npts', 'index_min', 'index_max']:
        if attr not in lag.attrs:
            raise ValueError(f'Lag has no attribute "{attr}"!')

    w = xr.DataArray(
        data=weight(
            lag.attrs['npts'], pad=True
        )[lag.attrs['index_min']:lag.attrs['index_max']],
        dims=(lag.name),
        coords={lag.name: lag},
        name=name,
        attrs={
            'long_name': 'Unbiased CC estimate scale factor',
            'units': '-',
        }
    )

    return w
