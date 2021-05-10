r"""

:mod:`signal.cwt` -- Continuous Wavelet Transform
=================================================

Perform the continuous wavelet transform of an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
import scipy as sp
try:
    import dask
except ModuleNotFoundError:
    dask = False

# Relative import
from ..util.history import historicize


__all__ = ['cwt']


def cwt(
    x: xr.DataArray, wavelet: callable = None, freqs=None, widths=None,
    dim: str = None, dtype: np.dtype = None, **kwargs
):
    """
    Compute the continuous wavelet transform of an N-D labelled dataarray.

    Implementation of :func:`scipy.signal.cwt` to a
    :class:`xarray.DataArray`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data for which to perform the wavelet transform.

    wavelet : `function`
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.

    freqs : `sequence`
        Frequencies to use for transform.

    widths : `sequence`
        Widths to use for transform.

    dim : `str`, optional
        The time dimension of ``x`` used to compute the wavelet transform.
        Defaults to the last dimension.

    dtype : :class:`np.dtype`, optional
        The desired data type of output. Defaults to `np.float64` if the output
        of wavelet is real and `np.complex128` if it is complex.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`scipy.signal.cwt`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The wavelet transform of ``x``.

    """
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # set wavelet
    wavelet = wavelet or sp.signal.ricker

    # widths and frequencies
    if widths is None:
        fs = x[dim].attrs['sampling_rate']
        if freqs is None:
            freqs = np.linspace(1/fs, fs/2, 100)
        elif isinstance(freqs, int):
            freqs = np.linspace(1/fs, fs/2, np.abs(freqs))
        elif isinstance(freqs, tuple):
            if len(freqs) != 3:
                raise ValueError('Freqs should be a tuple with '
                                 '(min, max, steps)')
            f_min, f_max, f_steps = freqs
            if f_min <= 0:
                raise ValueError('Minimum frequency cannot be <= 0')
            if f_max > fs/2:
                raise ValueError('Maximum frequency cannot be > Nyquist')
            freqs = np.linspace(f_min, f_max, f_steps)
        elif isinstance(freqs, xr.DataArray):
            freqs = freqs.values
        else:
            freqs = np.array(freqs)
        w = kwargs['w'] if 'w' in kwargs else 1.
        widths = w*fs/(2*np.pi*freqs)
        new_dim = xr.DataArray(
            data=freqs,
            dims=('freq',),
            name='freq',
            attrs={
                'long_name': 'Frequency',
                'standard_name': 'frequency',
                'units': 'Hz',
                'w': w,
                'fs': fs,
            },
        )
    else:
        freqs = None
        widths = (widths.values if isinstance(widths, xr.DataArray)
                  else np.array(widths))
        new_dim = xr.DataArray(
            data=widths,
            dims=('width',),
            name='width',
            attrs={
                'long_name': 'Continuous Wavelet Transform Width',
                'standard_name': 'continuous_wavelet_transform_width',
                'units': '-',
            },
        )

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(
            dask='allowed',
            output_dtypes=[dtype or np.float64],  # or np.complex128
            output_sizes={new_dim.name: len(widths)}
        )

    # apply cwt as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(sp.signal.cwt, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[[new_dim.name, dim]],
                       keep_attrs=True,
                       vectorize=True,
                       kwargs=dict(wavelet=wavelet, widths=widths,
                                   dtype=dtype, **kwargs),
                       **dargs)

    # add frequency coordinate
    y = y.assign_coords(width=new_dim) if freqs is None else y.assign_coords(freq=new_dim)

    # set attributes
    y.name = 'cwt'
    y.attrs = {
        **x.attrs,
        'long_name': f"{x.long_name} Continuous Wavelet Transform",
        'standard_name': f"{x.standard_name}_continuous_wavelet_transform",
        'units': '-',
        'wavelet': repr(wavelet),
    }
    y.encoding = {'zlib': True, 'complevel': 9}

    # log workflow
    historicize(y, f='cwt', a={
        'x': x.name,
        'wavelet': repr(wavelet),
        '**kwargs': kwargs,
    })

    return y
