r"""

:mod:`signal.mask` -- Mask
==========================

Construct a labeled mask array.

"""


# Mandatory imports
import xarray as xr
import numpy as np


# Relative imports
from ..util.time import to_seconds
from ..util.history import historicize


__all__ = ['mask', 'multi_mask']


def mask(
    x: xr.DataArray, lower=None, upper=None, scalar = None,
    name: str = None, invert: bool = False, to_where: bool = False, **kwargs
):
    r"""Construct a one-dimensional N-D labeled mask array.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`
        The one-dimensional array of data or coordinate to be masked.

    lower : `various`, optional
        Specify the lower bound of ``coord``. If `None`, no bound is set.

    upper : `various`, optional
        Specify the upper bound of ``coord``. If `None`, no bound is set.

    scalar : `various`, optional
        Multiply ``lower`` and ``uper`` input parameters with a scaler. If
        `None` (default), 1 is used.

    invert : `bool`, optional
        Invert and and swap ``lower`` and ``upper`` bounds if `True` to
        ``1/upper`` and ``1/lower``. Only bounds that are not `None` are
        inverted and swapped. Default is `False`.

    name : `str`, optional
        Sets ``mask.name``. If `None` (default), the name is set to
        'mask_{coord.name}'.

    to_where : `bool`, optional
        Apply the masked output ``x.where(mask)``.

    Returns:
    --------
    mask : :class:`xarray.DataArray`
        The masked output of ``coord``. Values inside the mask criteria become
        `True`, else `False`. If ``to_where`` is `True`, masked values are
        returned whereas non-masked values become `NaN`.
    """
    assert len(x.coords) == 1, (
        '``x`` should be a coordinate or variable with a single dimension!'
    )
    x = to_seconds(x)
    scalar = get_scalar_value(scalar, 1.)
    lower = get_scalar_value(lower) if lower else None
    upper = get_scalar_value(upper) if upper else None

    # Invert
    l = 1/upper if invert else lower
    u = 1/lower if invert else upper

    # Multiply or replace None
    l = l * scalar if l else x.min().values
    u = u * scalar if u else x.max().values

    mask = (x >= l) & (x <= u)
    mask.name = name or 'mask_{}'.format(x.name)

    historicize(mask, f='mask', a={
        'x': '{} ({})'.format(x.name, x.dims[0]),
        'lower': lower,
        'upper': upper,
        'scalar': scalar,
        'invert': invert,
        'name': name,
        'to_where': to_where,
        '**kwargs': kwargs,
    })

    return coord.where(mask, **kwargs) if to_where else mask


def get_scalar_value(x: xr.DataArray, x0=None, timedelta_to_seconds: bool = True):
    r"""Convert a dimensionless array to a scalar.
    """
    assert not (not x and not x0), (
        'Input parameter cannot be empty without a default value!'
    )
    if isinstance(x, np.ndarray) or isinstance(x, xr.DataArray):
        y = np.asscalar(x) if x else x0
    else:
        y = x if x else x0
    return to_seconds(y) if timedelta_to_seconds else y


def multi_mask(
    x: xr.DataArray, y: xr.DataArray, lower=None, upper=None,
    invert: bool = False, name: str = None
):
    r"""Construct a two-dimensional N-D labeled mask array.

    Parameters:
    -----------
    x : :class:`xarray.DataArray`

    y : :class:`xarray.DataArray`
        The secondary one-dimensional array of data or coordinate to be masked.
        ``y`` is used as ``scalar`` in :func:`mask` and multiplied  with each
        ``lower`` and ``uper`` input parameter. ``y`` should have on different
        coordinate than ``x``.

    lower : `various`, optional
        Specify the lower bound of ``x`` as a function of ``y``.
        If `None`, no bound is set.

    upper : `various`, optional
        Specify the upper bound of ``x`` as a function of ``y``.
        If `None`, no bound is set.

    invert : `bool`, optional
        Invert and and swap ``lower`` and ``upper`` bounds if `True` to
        ``1/upper`` and ``1/lower``. Only bounds that are not `None` are
        inverted and swapped. Default is `False`.

    name : `str`, optional
        Sets ``mask.name``. If `None` (default), the name is set to
        'mask_{x.name}_{y.name}'.

    Returns:
    --------
    mask : :class:`xarray.DataArray`
        The masked output of ``x``. Values inside the mask criteria
        become `True`, else `False`.
    """
    assert len(x.coords) == 1, (
        '``x`` should be a coordinate or variable with a single dimension!'
    )
    assert len(y.coords) == 1, (
        '``y`` should be a coordinate or variable with a single dimension!'
    )
    assert y.name != x.name and x.dims != y.dims, (
        '``x`` and ``y`` should have a different coordinate !'
    )
    assert lower or upper, (
        'At least ``lower`` or ``upper`` is required!'
    )
    lower = get_scalar_value(lower) if lower else None
    upper = get_scalar_value(upper) if upper else None

    dim = y.dims[0]
    mask = []
    for y0 in y:
        m = mask_coord(x, lower=lower, upper=upper, scalar=y0.values, invert=invert)
        mask.append(m.assign_coords(coord=y0[dim]))

    mask = xr.concat(mask, dim=dim)
    mask.name = 'mask_{}_{}'.format(x.name, y.name)
    mask.attrs.pop('history')

    historicize(mask, f='multi_mask', a={
        'x': '{} ({})'.format(x.name, x.dims[0]),
        'y': '{} ({})'.format(y.name, y.dims[0]),
        'lower': lower,
        'upper': upper,
        'invert': invert,
        'name': name,
    })

    return mask
