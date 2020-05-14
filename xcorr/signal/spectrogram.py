r"""

:mod:`signal.spectrogram` -- Spectrogram
========================================

Generate a spectrogram of an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
from scipy import signal
try:
    import dask
except ModuleNotFoundError:
    dask = False

# Relative import
from ..util.history import historicize


__all__ = ['psd']


def psd(
    x: xr.DataArray, duration: float = None, padding_factor: int = 2,
    dim: str = 'lag', **kwargs
):
    """
    Compute an N-D labelaled spectrogram with consecutive Fourier transforms.

    Implementation of :func:`scipy.signal.spectrogram` to a
    :class:`xarray.DataArray`.

    The dimension ``dim`` for which to compute the spectrogram should contain
    both sampling attributes ``sampling_rate`` and ``delta``.
    The computed spectrogram contains the coordinates of ``x`` and a new
    coordinate 'frequency' with sample frequencies ranging from dc up to
    Nyquist ``dim.sampling_rate/2``. Spectrogram time segments correspond to
    the samples in ``dim``, with `NaN` at the outer edges (``duration/2``).

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data for which to compute the spectrogram.

    duration : `float`
       The duration of each segment, in seconds.

    padding_factor : `int`, optional
        Factor applied to duration, if a zero padded FFT is desired. If `None`,
        the FFT length is duration. Defaults to None. Default is 2.

    dim : `str`, optional
        The time coordinates name of ``x`` used to compute the spectrogram.
        Defaults to 'lag'.

    **kwargs
        Extra arguments passed on to :func:`scipy.signal.spectrogram`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The computed spectrogram for ``x``.

    """
    assert dim in x.dims, (
        f'x has no dimension "{dim}"!'
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension has no attribute "{sampling_rate}"!'
    )
    assert 'delta' in x[dim].attrs, (
        'Dimension has no attribute "{delta}"!'
    )

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    duration = duration if (
        duration and duration > x[dim].delta
    ) else x[dim].delta

    sampling_rate = x[dim].attrs['sampling_rate']

    win_len = int(duration * sampling_rate)
    assert win_len >= 16, 'Change duration to have at least 16 sample points!'

    # static dimensions
    nfft = int(win_len * padding_factor)
    edge = int(np.rint(win_len/2-1))

    # expand x with frequency coordinate
    freq = np.linspace(0., sampling_rate/2, int(nfft/2 + 1))

    # get lag index
    axis = x.dims.index(dim)
    axis = -1 if axis == len(x.dims) - 1 else axis

    def _spectrogram(lag):
        _f, _t, Sxx = signal.spectrogram(
            x=lag,
            fs=sampling_rate,
            nperseg=win_len,
            noverlap=win_len-1,
            nfft=nfft,
            scaling='density',
            mode='psd',
            axis=axis,
            **kwargs
        )
        # get basic shape (in loop for dask!)
        shape = list(Sxx.shape)

        # make top sponge for lag
        shape[axis] = len(x[dim]) - shape[axis] - edge
        e1 = np.zeros(shape, dtype=x.dtype)

        # make bot sponge for lag
        shape[axis] = edge
        e0 = np.zeros(shape, dtype=x.dtype)

        return np.concatenate((e0, Sxx, e1), axis=axis)

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(
            dask='parallelized',
            output_dtypes=[x.dtype],
            output_sizes={'freq': len(freq)}
        )

    # apply spectrogram as ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_spectrogram, x,
                       input_core_dims=[[dim]],
                       output_core_dims=[['freq', dim]],
                       keep_attrs=True,
                       vectorize=False,
                       **dargs,
                       **kwargs)

    # add frequency coordinate
    y = y.assign_coords(freq=freq)
    y.freq.attrs = {
        'name': 'freq',
        'long_name': 'Frequency',
        'standard_name': 'frequency',
        'units': 'Hz',
    }

    # set attributes
    y.name = 'psd'
    y.attrs = {
        **x.attrs,
        'long_name': 'Power Spectral Density',
        'standard_name': 'power_spectral_density',
        'units': 'Hz**-1',
        'from_variable': x.name,
        'scaling': 'density',
        'mode': 'psd',
        'duration': duration,
        'padding_factor': padding_factor,
        'centered': np.byte(True),
        **kwargs
    }

    # log workflow
    historicize(y, f='psd', a={
        'x': x.name,
        'duration': duration,
        'padding_factor': padding_factor,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def psd_f_t(
    x: xr.DataArray, duration: float = None, padding_factor: int = None,
    overlap: float = None, dim: str = 'lag', **kwargs
):
    """
    Compute an N-D labelaled spectrogram with consecutive Fourier transforms
    with manual time segment control.

    Implementation of :func:`scipy.signal.spectrogram` to a
    :class:`xarray.DataArray`.

    The dimension ``dim`` for which to compute the spectrogram should contain
    both sampling attributes ``sampling_rate`` and ``delta``.

    The computed spectrogram contains the coordinates of ``x`` minus ``dim``,
    and two new coordinates, 'psd_f' and 'psd_t`, with sample frequencies
    ranging from dc up to Nyquist ``dim.sampling_rate/2`` and with time
    segments depending on ``dim``, ``duration`` and ``overlap``, respectively.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to be filtered.

    duration : `float`
       The duration of each segment, in seconds.

    padding_factor : `int`, optional
        Factor applied to duration, if a zero padded FFT is desired. If `None`,
        the FFT length is duration. Defaults to None. Default is 2.

    overlap : `float`
       The decimal overlap between two consecutive segment: [0,1).

    dim : `str`, optional
        The coordinates name if ``x`` to be filtered over. Default is 'lag'.

    **kwargs
        Extra arguments passed on to :func:`scipy.signal.spectrogram`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The computed spectrogram for ``x``.
    """
    assert dim in x.dims, (
        f'x has no dimension "{dim}"!'
    )
    assert 'sampling_rate' in x[dim].attrs, (
        'Dimension has no attribute "{sampling_rate}"!'
    )
    assert 'delta' in x[dim].attrs, (
        'Dimension has no attribute "{delta}"!'
    )

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    duration = duration if (
        duration and duration > x[dim].attrs['delta']
    ) else x[dim].attrs['delta']

    win_len = int(duration * x[dim].attrs['sampling_rate'])

    assert win_len >= 16, 'Change duration to have at least 16 sample points!'

    overlap = overlap if overlap and (0. < overlap < 1.) else .9
    win_shift = int(win_len*overlap)

    f, t, Sxx = signal.spectrogram(
        x=x.values,
        fs=x[dim].attrs['sampling_rate'],
        nperseg=win_len,
        noverlap=win_shift,
        nfft=int(win_len*padding_factor),
        scaling='density',
        mode='psd',
        axis=x.dims.index(dim),
        **kwargs
    )

    t += x[dim].values[0]

    coords = {}
    for d in x.dims:
        if d != dim:
            coords[d] = x[d]
    coords['psd_f'] = (
        'psd_f',
        f,
        {
            'long_name': 'Frequency',
            'standard_name': 'frequency',
            'units': 'Hz',
        }
    )
    coords['psd_t'] = (
        'psd_t',
        t,
        {
            'long_name': 'Time',
            'standard_name': 'time',
        }
    )

    y = xr.DataArray(
        data=Sxx,
        dims=coords.keys(),
        coords=coords,
        name='psd',
        attrs={
            'long_name': 'Power Spectral Density',
            'standard_name': 'power_spectral_density',
            'units': 'Hz**-1',
            'from_variable': x.name,
            'scaling': 'density',
            'mode': 'psd',
            'duration': duration,
            'padding_factor': padding_factor,
            'overlap': overlap,
            'centered': np.byte(True),
            **kwargs
        },
    )

    historicize(y, f='psd_f_t', a={
        'x': x.name,
        'duration': duration,
        'padding_factor': padding_factor,
        'overlap': overlap,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y
