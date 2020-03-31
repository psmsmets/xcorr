r"""

:mod:`util.history` -- History Utilities
========================================

Utilities for ``xcorr`` for `xarray.DataArray` history attributes.

"""


# Mandatory imports
from xarray import DataArray
import json


## Relative imports
from ..version import version
from ..util.hasher import _to_serializable
from .. import __name__ as name


__all__ = ['historicize']


def historicize(x: DataArray, f: str, a: dict = {}):
    r"""Append an applied function and its arguments to the history attrs of
    an N-D labeled array of data.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data. The history attribute is updated in-place.

    f : `str`
        The applied function to ``x``.

    a : `dict`
        The arguments of ``f``.

    """
    h = x.history.split('; ') if 'history' in x.attrs else []
    h.append('{n}-{v}: {f}({a})'.format(
        n=name,
        v=version,
        f=f,
        a=json.dumps(
            a,
            separators=(',', ':'),
            sort_keys=False,
            indent=None,
            skipkeys=False,
            default=_to_serializable
        ),
    ))
    x.attrs['history'] = '; '.join(h)
