r"""

:mod:`signal.window` -- Window
==============================

Create a window for an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
import scipy as sp


# Relative imports
from ..util.history import historicize


__all__ = ['window']


_sides = ['left', 'right', 'both']


def window(
    x: xr.DataArray, wtype: str = None, max_percentage: float = None,
    max_length: float = None, side: str = None
):
    """
    Return a window for the given coordinate of N-D labelled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The coordinate to be windowed.

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

    Returns
    -------
    w : :class:`xarray.DataArray` or `None`
        The window for coordinates ``x``.

    """
    if len(x.dims) != 1 or x.name not in x.dims:
        raise ValueError('x should be a coordinate or variable with '
                         'a single dimension!')

    if 'sampling_rate' not in x.attrs:
        raise ValueError('x has no attribute "sampling_rate"!')

    wtype = wtype or 'hann'
    if wtype not in sp.signal.windows.__all__:
        raise ValueError('wtype should be a :func:`scipy.signal.windows`!')

    side = side or 'both'
    if side not in _sides:
        raise ValueError('side should be either "left", "right" or "both"!')

    nmax = 0.5 * len(x)

    if max_percentage:
        if not isinstance(max_percentage, float):
            raise TypeError('max_percentage should be of type `float`.')
        if max_percentage < 0. or max_percentage > .5:
            raise ValueError('max_percentage should be within [0, .5].')
        nmax = max_percentage * len(x)

    if max_length:
        if not isinstance(max_length, float):
            raise TypeError('max_length should be of type `float`.')
        if max_length < 0. or max_length > np.max(x.values):
            raise ValueError('max_length should be within [0, x].')
        nmax = max_length * x.attrs['sampling_rate']

    nmax = int(round(nmax))

    win = eval(f'sp.signal.windows.{wtype}({nmax}*2, False)')

    w = xr.DataArray(
        data=np.ones((len(x)), dtype=np.float64),
        dims=(x.name),
        coords={x.name: x},
        name='win',
        attrs={
            'long_name': 'Taper window',
            'standard_name': 'taper_window',
            'units': '-',
        }
    )
    w.values[0:nmax] = win[0:nmax]
    w.values[-nmax:] = win[-nmax:]

    historicize(w, f='window', a={
        'x': x.name,
        'wtype': wtype,
        'max_percentage': max_percentage,
        'max_length': max_length,
        'side': side
    })

    return w
