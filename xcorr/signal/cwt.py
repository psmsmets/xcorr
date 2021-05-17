r"""

:mod:`signal.cwt` -- Continuous Wavelet Transform
=================================================

Perform the continuous wavelet transform of an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr
import pywt
try:
    import dask
except ModuleNotFoundError:
    dask = False

# Relative import
from ..util.history import historicize
from .absolute import absolute


__all__ = ['cwt', 'scaleogram']


def cwt(
    x: xr.DataArray, scales=None, wavelet=None, method: str = None,
    dim: str = None
):
    """
    Compute the continuous wavelet transform of an N-D labelled data array.

    Implementation of :func:`pywt.cwt` to a :class:`xarray.DataArray`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data for which to perform the wavelet transform.

    scales : `array_like`
        The wavelet scales to use.

    wavelet : `Wavelet object or name`, optional
        Wavelet to use. Defaults to 'cmor1.5-1.0'.
        See ``pywt.wavelist(kind='continuous')`` for a list of continuous
        wavelet families.

    method : {'conv', 'fft'}, optional
        The method used to compute the CWT. Can be any of:
            o ``conv`` uses `numpy.convolve`.
            o ``fft`` uses frequency domain convolution.
            o ``auto`` uses automatic selection based on an estimate of the
              computational complexity at each scale.
        The conv method complexity is ``O(len(scale) * len(data))``. The fft
        method is ``O(N * log2(N))`` with ``N = len(scale) + len(data) - 1``.
        It is well suited for large size signals but slightly slower than
        ``conv`` on small ones

    dim : `str`, optional
        The time dimension of ``x`` used to compute the wavelet transform.
        Defaults to the last dimension.

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

    # sampling rate
    if "sampling_rate" in x[dim].attrs:
        fs = x[dim].attrs['sampling_rate']
    elif "sampling_period" in x[dim].attrs:
        fs = 1/x[dim].attrs['sampling_period']
    else:
        fs = 1.

    # wavelet
    w = wavelet or "cmor1.5-1.0"
    if isinstance(w, str):
        w = pywt.ContinuousWavelet(w)
    elif not isinstance(w, pywt.ContinuousWavelet):
        raise TypeError("Wavelet should be of type 'pywt.ContinuousWavelet'")

    # scales
    smin, smax = w.center_frequency*2, w.center_frequency*fs*fs/2
    s, s0, s1, sn = None, smin*1, smax*1, 200
    if isinstance(scales, int):
        sn = scales
    elif isinstance(scales, tuple) and len(scales) == 3:
        s0, s1, sn = scales
    elif isinstance(scales, xr.DataArray):
        s = scales.values
    elif scales is not None:
        s = np.array(scales)
    s = s or np.logspace(np.log10(s0), np.log10(s1), sn)

    # frequency
    f = pywt.scale2frequency(w, s) * fs

    # method
    method = method or 'fft'

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(
            dask='allowed',
            output_dtypes=[np.complex128 if w.complex_cwt else np.float64],
            output_sizes={"freq": len(scales)}
        )

    # ufunc wrapper (catch frequency return)
    def _cwt(x, **kwargs):
        c, f = pywt.cwt(x, **kwargs)
        return np.moveaxis(c, 0, -2)

    # apply cwt as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_cwt, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[["freq", dim]],
                       keep_attrs=True,
                       vectorize=False,
                       kwargs=dict(
                           scales=s,
                           wavelet=w,
                           method=method,
                           axis=-1,
                       ),
                       **dargs)

    # add width coordinate
    y = y.assign_coords(
        freq=xr.DataArray(
            data=f,
            dims=("freq",),
            name="freq",
            attrs={
                'long_name': 'Frequency',
                'standard_name': 'Frequency',
                'units': '-',
                'sampling_rate': fs,
                'wavelet': w.name,
            },
        )
    )

    # set attributes
    y.name = 'cwt'
    y.attrs = {
        **x.attrs,
        'long_name': f"{x.long_name} Continuous Wavelet Transform",
        'standard_name': f"{x.standard_name}_continuous_wavelet_transform",
        'units': '-',
        'sampling_rate': fs,
        'wavelet': w.name,
        'wavelet_bandwidth_frequency': w.bandwidth_frequency,
        'wavelet_center_frequency': w.center_frequency,
        'wavelet_family_name': w.family_name,
    }
    y.encoding = {'zlib': True, 'complevel': 9}

    # log workflow
    historicize(y, f='cwt', a={
        'x': x.name,
        'wavelet': w.name,
        'method': method,
        'dim': dim,
    })

    return y


def scaleogram(
    x: xr.DataArray, magnitude: bool = True, dim: str = None,
    freq_lim: tuple = None, dtype: np.dtype = None,
    **kwargs
):
    """
    Compute the scaleogram using the continuous wavelet transform of an N-D
    labelled data array.

    Implementation of :func:`pywt.cwt` to a :class:`xarray.DataArray`.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data for which to compute the scaleogram.

    magnitude : `bool`, optional
        Compute the magnitude (norm) of the (complex) wavelet transform.

    dim : `str`, optional
        The time dimension of ``x`` used to compute the wavelet transform.
        Defaults to the last dimension.

    freq_lim : `tuple`, optional
        Limit the frequency dimension given the minimum and maximum frequency.

    dtype : :class:`np.dtype`, optional
        The desired data type of output. Defaults to `np.float64` if the output
        of wavelet is real and `np.complex128` if it is complex.

    **kwargs :
        Any additional keyword arguments will be passed to :func:`cwt`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The scaleogram of ``x``.

    """
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')

    # cwt
    y = cwt(x, **kwargs)

    # freq limit?
    if freq_lim is not None:
        try:
            f0, f1 = freq_lim
        except Exception:
            raise ValueError("freq_lim should be a tuple with min and max")
        y = y.where((y.freq >= f0) & (y.freq <= f1), drop=True)

    # magnitude?
    if magnitude:
        y = absolute(y).astype(dtype or x.dtype)
        lname = 'Magnitude Scaleogram'
        sname = 'magnitude_'
    else:
        lname = 'Scaleogram'
        sname = 'scaleogram'

    # update attributes
    y.name = 'Scwt'
    y.attrs['long_name'] = f"{x.long_name} {lname}"
    y.attrs['standard_name'] = f"{x.standard_name}_{sname}",

    # log workflow
    historicize(y, f='scaleogram', a={
        'x': x.name,
        'magnitude': magnitude,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
