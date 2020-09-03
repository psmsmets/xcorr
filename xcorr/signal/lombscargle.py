r"""

:mod:`signal.lombscargle` -- Lomb-Scargle periodogram
=====================================================

Compute the Lomb-Scargle periodogram of an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
from scipy import signal
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['lombscargle']


def lombscargle(
    x: xr.DataArray, f: xr.DataArray, dim: str = None, invert: bool = False,
    normalize: bool = False, rescale: bool = False, precenter: bool = False,
):
    """
    Computes the Lomb-Scargle periodogram of an N-D labelled array of data.

    Implementation of :func:`scipy.signal.lombscargle`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The data array with measurement values.

    f : :class:`xarray.DataArray`
        Frequencies of oscillation, i.e., number of cycles per unit time,
        to compute the periodogram (NOT angular frequencies).

    dim : `str`, optional
        The coordinate name of ``x`` specifying the sample times to compute
        the periodogram. NaN are discarded automatically.
        Defaults to the last dimension of ``x``.

    invert : `bool`, optional
        Invert ``f`` to compute the periodogram, e.g., given period.

    normalize : `bool`, optional
        Compute normalized periodogram.

    rescale : `bool`, optional
        Rescale the computed unnormalized periodogram is unnormalized to the
        harmonic signal with amplitude A given a sufficient large number of
        measurement N (excluding NaN).
        If `False` (default), the periodogram takes the value (A**2) * N/4.

    precenter : `bool`, optional
        Pre-center amplitudes by subtracting the mean.

    Returns
    -------
    pgram : :class:`xarray.DataArray`
        The Lomb-Scargle periodogram of ``x`` and ``y``.

    """
    # measurements and samples
    if not isinstance(x, xr.DataArray):
        raise TypeError('x should be an xarray.DataArray.')

    # frequencies
    if not isinstance(f, xr.DataArray):
        raise TypeError('f should be an xarray.DataArray.')
    if len(f.dims) != 1:
        raise ValueError('f should be a coordinate.')
    if f.name is None:
        raise ValueError('f.name cannot be empty.')

    print(f)
    # invert
    if invert:
        f = 1./f

    print(f)

    # dim
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # rescale
    rescale = False if normalize else rescale

    # lombscargle wrapper to simplify ufunc input
    def _lombscargle(t, x, f, **kwargs):
        def _pgram_axis(x, t, w, rescale=False, **kwargs):
            valid = ~np.isnan(x)
            pgram = signal.lombscargle(t[valid], x[valid], w, **kwargs)
            return np.sqrt(pgram*4/sum(valid)) if rescale else pgram
        pgram = np.apply_along_axis(_pgram_axis, -1, x, t, 2*np.pi*f, **kwargs)
        return pgram

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[x.dtype])

    # apply sosfiltfilt as ufunc (and optional dask distributed)
    pgram = xr.apply_ufunc(_lombscargle, x[dim], x, f,
                           input_core_dims=[[dim], [dim], [f.name]],
                           output_core_dims=[[f.name]],
                           kwargs={
                               'normalize': normalize,
                               'rescale': rescale,
                               'precenter': precenter
                           },
                           keep_attrs=False,
                           vectorize=False,
                           **dargs)

    # set coordinate and attributes
    pgram.name = 'pgram'
    pgram.attrs = {
        'units': x.attrs['units'],
        'long_name': 'Lomb-Scargle periodogram',
        'standard_name': 'periodogram',
        'from_variable': x.name,
    }
    pgram = pgram.assign_coords({f.name: 1./f if invert else f})

    # log workflow
    historicize(pgram, f='lombscargle', a={
        'x': x.name,
        'f': f.name,
        'dim': dim,
        'invert': invert,
        'normalize': normalize,
        'rescale': rescale,
        'precenter': precenter,
    })

    return pgram