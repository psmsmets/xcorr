r"""

:mod:`signal.beamform` -- Beamform
==================================

Least-squares plane wave beamforming to an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np

# Relative imports
from ..signal.correlate import correlate1d
from ..util.metadata import global_attrs


__all__ = ['plane_wave']


def plane_wave(
    s: xr.DataArray, x: xr.DataArray, y: xr.DataArray, dim: str = None,
    **kwargs
):
    """
    Return the least-squares estimated plane wave given a signal and
    xy-coordinates, all N-D labeled data arrays.

    Parameters
    ----------
    s : :class:`xarray.DataArray`
        The signal data array to estimate the plane wave.

    x : :class:`xarray.DataArray`
        The x-coordinate data array. Should be one-dimensional.

    y : :class:`xarray.DataArray`
        The y-coordaine data array. Should be one-dimensional.

    dim : `str`, optional
        The coordinate name of ``s`` defining the signal time dimension.
        Defaults to the last dimension of ``s``.
        Other dimensions (excluding the xy-coordinate dimension) will be
        broadcasted.

    **kwargs :
        Any additional keyword arguments are used to set the dataset global
        metadata attributes.

    Returns
    -------

    ds : :class:`xarray.Dataset`
        Data set with the estimated plane wave direction of arrival, the
        horizontal propagation velocity across the array and the LSE error
        minimum value.

    """
    # signal dim
    dim = dim or s.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in s.dims:
        raise ValueError(f's has no dimensions "{dim}"')

    # x-coordinate
    if len(x.dims) != 1:
        raise ValueError('x should be a coordinate or a variable with '
                         'a single dimension!')
    rdim = x.dims[0]
    r = s[rdim].drop_vars(tuple(d for d in s[rdim].coords if d != rdim))
    if rdim not in s.dims:
        raise ValueError(f's has no dimension "{x.dims[0]}"')
    if not x[rdim].equals(r):
        raise ValueError(f'dimension "{rdim}" of s and x do not agree')

    # y-coordinate
    if len(y.dims) != 1:
        raise ValueError('y should be a coordinate or a variable with '
                         'a single dimension!')
    if not y[rdim].equals(r):
        raise ValueError(f'dimension "{rdim}" of s and y do not agree')

    # output dims
    out_dims = tuple(c for c in s.coords if c not in (dim, rdim))
    out_shape = tuple(s[c].size for c in out_dims)

    # construct co-array receiver couples
    N = r.size
    coAi0, coAi1 = np.triu_indices(N, 1)
    M = coAi0.size

    # construct location matrix
    A = np.array([
        x.isel({rdim: coAi1}).values - x.isel({rdim: coAi0}).values,
        y.isel({rdim: coAi1}).values - y.isel({rdim: coAi0}).values,
    ]).T

    # prepare LSE
    ATAinvAT = np.linalg.inv(A.T.dot(A)).dot(A.T)

    # obtain lag times per co-array receiver couple
    tau = xr.DataArray(
        data=np.zeros(out_shape + (M,), dtype=x.dtype),
        dims=out_dims + ('M',),
        coords={d: s[d] for d in out_dims},
    )
    for i in range(M):
        cc = correlate1d(
            in1=s.isel({rdim: coAi0[i]}),
            in2=s.isel({rdim: coAi1[i]}),
            dim=dim
        )
        argmax = xr.ufuncs.fabs(cc).argmax(dim='delta_'+dim)
        tau.loc[{'M': i}] = cc['delta_'+dim][argmax]

    # estimate plane wave (broadcast-like)
    def LSE(tau):
        s = ATAinvAT.dot(tau)
        ns = np.linalg.norm(s)
        if ns > 0:
            vel = 1/ns
            doa = (np.arctan2(s[0], s[1]).item() * 180./np.pi) % 360.
        else:
            vel = np.inf
            doa = 0.
        e = tau - A.dot(s)
        err = e.T.dot(e)
        return doa, vel, err
    av = np.apply_along_axis(LSE, -1, tau.values)

    # output
    ds = xr.Dataset()
    ds.attrs = global_attrs({
        'title': (
            kwargs.pop('title', '') +
            'Signal-to-noise ratio - {} to {}'
            .format(
                x.time[0].dt.strftime('%Y.%j').item(),
                x.time[-1].dt.strftime('%Y.%j').item(),
            )
        ).strip(),
        **kwargs,
        'references': (
             'Bendat, J. Samuel, & Piersol, A. Gerald. (1971). '
             'Random data : analysis and measurement procedures. '
             'New York (N.Y.): Wiley-Interscience.'
        ),
    })

    ds['doa'] = xr.DataArray(
        data=np.take(av, 0, axis=-1),
        dims=out_dims,
        coords={d: s[d] for d in out_dims},
        attrs={
            'long_name': 'Direction of arrival',
            'standard_name': 'direction_of_arrival',
            'units': 'degree',
            'reference': 'clockwise from north',
        },
        name='doa',
    )

    ds['vel'] = xr.DataArray(
        data=np.take(av, 1, axis=-1),
        dims=out_dims,
        coords={d: s[d] for d in out_dims},
        attrs={
            'long_name': 'Horizontal velocity',
            'standard_name': 'horizontal_velocity',
            'units': 'm s-1',
        },
        name='vel',
    )

    ds['err'] = xr.DataArray(
        data=np.take(av, 2, axis=-1),
        dims=out_dims,
        coords={d: s[d] for d in out_dims},
        attrs={
            'long_name': 'Error minimum value',
            'standard_name': 'error_min_value',
            'units': 's2',
        },
        name='err',
    )

    return ds
