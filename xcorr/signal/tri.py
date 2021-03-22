r"""

:mod:`signal.tri` -- Tri
========================

Triangle array opperations for an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np


__all__ = ['tri_mask', 'tri_mirror']


def tri_mask(
    coord1: xr.DataArray, coord2: xr.DataArray, k: int = None
):
    """
    Triangular mask an array.

    Return a mask for a two-dimensional array ignoring the elements either
    below or above the k-th diagonal.

    Parameters
    ----------

    coord1 : :class:`xarray.DataArray`
        The first coordinate of size N to apply the triangular mask.

    coord2 : :class:`xarray.DataArray`
        The second coordinate of size M to apply the triangular mask.

    k : `int`, optional
        Diagonal above which to ignore elements. k = 0 (the default) is the
        main diagonal, k < 0 is below it and k > 0 is above.

    Returns
    -------
    m : :class:`xarray.DataArray`
        The mask for coordinates ``coord1`` and ``coord2`` of shape (N, M).

    """

    k = k or 0
    m = xr.DataArray(
        np.tri(coord1.size, coord2.size, k=k, dtype=bool),
        dims=(coord1.name, coord2.name),
        coords=(coord1, coord2),
    )

    return m


def tri_mirror(
    x: xr.DataArray, m: xr.DataArray = None, sign_inverse: bool = False,
    dims: tuple = None, **kwargs
):
    """
    Triangular mirror an array.

    Mirror the values of the off-diagonal elements either below or above the
    k-th diagonal. Values masked False are filled using `.where(m, <other>)`.

    Parameters
    ----------

    x : :class:`xarray.DataArray`
        The data array to apply the triangular mirroring.

    m : :class:`xarray.DataArray`, optional
        The  two-dimensional triangular mask. If `None` (default) a mask is
        created using the last two dimensions of ``x``.

    sign_inverse : `bool`, optional
        Change the sign of ``x`` for the mirrored values.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`tri_mask`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The mirrored fill of data array ``x``.

    """

    # mask provided?
    if not isinstance(m, xr.DataArray):
        dims = dims or x.status.dims[-2:]
        m = tri_mask(coord1=x[dims[-2]], coord2=x[dims[-1]], **kwargs)
    else:
        # verify mask
        dims = m.dims
        if len(dims) != 2:
            raise ValueError('mask should contain two coordinates')
        for d in dims:
            if d not in x.dims:
                raise ValueError(f'mask dim "{d}" not found in "x"')

    # get base coordinates (not dims)
    c = list(filter(None, [d if d not in dims else None for d in x.dims]))

    # set dims for transpose
    t = c + list(reversed(dims))

    xT = x.transpose(*t, transpose_coords=False)
    xT = xT.rename({dims[0]: dims[1], dims[1]: dims[0]})

    # check dtype preserving

    return x.where(m, (-1 if sign_inverse else 1) * xT)
