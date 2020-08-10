r"""

:mod:`signal.rms` -- RMS
========================

Root-mean-square and N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr


# Relative imports
from ..util.history import historicize


__all__ = ['rms']


def rms(
    x: xr.DataArray, dim: str = None
):
    """
    Root-mean-square an N-D labeled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to be root-mean-squared.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is the
        last dimension.

    Returns
    -------
    y : :class:`xarray.DataArray` or `None`
        The rms of ``x``.

    """
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # square
    y = xr.ufuncs.square(x)

    # mean
    y = y.mean(dim=dim, skipna=True, keep_attrs=True)

    # root
    y = xr.ufuncs.sqrt(y)

    # log workflow
    historicize(y, f='rms', a={
        'x': x.name,
        'dim': dim,
    })

    return y
