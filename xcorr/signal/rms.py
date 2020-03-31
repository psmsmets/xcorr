r"""

:mod:`signal.extract` -- Extract
================================

Extract lag time windows of interest.

"""


# Mandatory imports
import xarray as xr


# Relative imports
from ..util.history import historicize


__all__ = ['rms']


def rms(
    x: xr.DataArray, dim: str = 'lag', inplace: bool = False
):
    r"""Root-mean-square an N-D labeled array of data.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The array of data to be root-mean-squared.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is 'lag'.

    inplace : `bool`, optional
        If `True`, filter in place and avoid a copy. Default is `False`.

    Returns:
    --------
    y : :class:`xarray.DataArray` or `None`
        The windowed output of ``x`` if ``inplace`` is `False`.
    """
    assert dim in x.dims, (
        'x has no dimension "{}"!'.format(dim)
    )
    y = x if inplace else x.copy()
    # square
    y = xr.ufuncs.square(y)
    # mean
    y = y.mean(dim=dim, skipna=True, keep_attrs=True)
    # root
    y = xr.ufuncs.sqrt(y)

    historicize(y, f='rms', a={
        'x': x.name,
        'dim': dim,
        'inplace': inplace,
    })

    return None if inplace else y
