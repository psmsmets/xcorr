r"""

:mod:`signal.taper` -- Taper
============================

Filter an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr


# Relative imports
from ..signal.window import window
from ..util.history import historicize


__all__ = ['taper']


def taper(
    x: xr.DataArray, wtype: str = None, max_percentage: float = None,
    max_length: float = None, side: str = None, dim: str = 'lag',

):
    """
    Taper an N-D labeled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to be windowed.

    wtype : `str`, optional
       The type of window. Default is ‘hann’. Should be a method of
       `scipy.signal.windows`.

    max_percentage : `float`, optional
        Decimal percentage of taper at one end (ranging from 0. to 0.5).

    max_length : `float`, optional
        Length of taper at one end in seconds.

    side : {'left', 'right', 'both'}, optional
        Specify if both sides should be tapered (default, 'both') or if only
        the left half ('left') or right half ('right') should be tapered.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is 'lag'.

    Returns
    -------
    y : :class:`xarray.DataArray` or `None`
        The windowed output of ``x`` if ``inplace`` is `False`.

    """
    assert dim in x.dims, (
        f'x has no dimension "{dim}"!'
    )

    w = window(x[dim], wtype, max_percentage, max_length, side)

    y = x * w

    y.attrs = x.attrs

    historicize(y, f='filter', a={
        'x': x.name,
        'wtype': wtype,
        'max_percentage': max_percentage,
        'max_length': max_length,
        'side': side,
        'dim': dim,
    })

    return y
