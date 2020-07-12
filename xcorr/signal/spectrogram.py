r"""

:mod:`signal.spectrogram` -- Spectrogram
========================================

Generate a spectrogram of an N-D labelled array of data.

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


__all__ = ['spectrogram']


def spectrogram(
    x: xr.DataArray, duration: float = None, padding_factor: int = 2,
    scaling: str = None, dim: str = None, **kwargs
):
    """
    Compute an N-D labelled spectrogram with consecutive Fourier transforms.

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

    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density (‘density’) where
        ``y`` has units of V**2/Hz and computing the power spectrum
        ('spectrum') where ``y`` has units of V**2, if ``x`` is measured in V.
        Defaults to ‘density’.

    dim : `str`, optional
        The time coordinates name of ``x`` used to compute the spectrogram.
        Defaults to the last dimension.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`scipy.signal.spectrogram`.

    Returns
    -------
    y : :class:`xarray.DataArray`
        The computed spectrogram for ``x``.

    """
    dim = dim or x.dims[-1]
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    if 'sampling_rate' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{sampling_rate}"!')

    if 'delta' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{delta}"!')

    duration = duration if (
        duration and duration > x[dim].delta
    ) else x[dim].delta

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    scaling = scaling or 'density'
    if scaling not in ['density', 'spectrum']:
        raise ValueError('Scaling should be either "density" or "spectrum"!')

    if scaling == 'density':
        units = f'{x[dim].units}-1'
        units = f'{x.units}2 {units}' if x.units != '-' else units
        long_name = 'Power Spectral Density'
        standard_name = 'power_spectral_density'
    else:
        units = f'{x.units}2' if x.units != '-' else '-'
        long_name = 'Power Spectrum'
        standard_name = 'power_spectrum'

    sampling_rate = x[dim].attrs['sampling_rate']

    win_len = int(duration * sampling_rate)
    if win_len < 16:
        raise ValueError('Change duration to have at least 16 sample points!')

    # static dimensions
    nfft = int(win_len * padding_factor)
    edge = int(np.rint(win_len/2))

    # expand x with frequency coordinate
    freq = np.linspace(0., sampling_rate/2, int(nfft/2 + 1))

    def _spectrogram(lag):
        # scipy spectrogram
        _f, _t, Sxx = signal.spectrogram(
            x=lag,
            fs=sampling_rate,
            nperseg=win_len,
            noverlap=win_len-1,
            nfft=nfft,
            scaling=scaling,
            mode='psd',
            axis=-1,
            return_onesided=True,
            **kwargs
        )
        # extend spectrogram at edges
        npad = ([(0, 0)]*(len(Sxx.shape)-1) +
                [(edge, lag.shape[-1] - Sxx.shape[-1] - edge)])

        return np.pad(Sxx, pad_width=npad, mode='constant', constant_values=0)

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
                       **dargs)

    # add frequency coordinate
    y = y.assign_coords(freq=freq)
    y.freq.attrs = {
        'name': 'freq',
        'long_name': 'Frequency',
        'standard_name': 'frequency',
        'units': f'{x[dim].units}-1',
    }

    # set attributes
    y.name = 'psd'
    y.attrs = {
        **x.attrs,
        'long_name': f'{x.long_name} {long_name}',
        'standard_name': f'{x.standard_name}_{standard_name}',
        'units': units,
        'scaling': scaling,
        'mode': 'psd',
        'duration': duration,
        'padding_factor': padding_factor,
        'centered': np.byte(True),
        **kwargs
    }
    y.encoding = {'zlib': True, 'complevel': 9}

    # log workflow
    historicize(y, f='psd', a={
        'x': x.name,
        'duration': duration,
        'padding_factor': padding_factor,
        'scaling': scaling,
        'dim': dim,
        '**kwargs': kwargs,
    })

    return y


def spectrogram_mtc(
    x: xr.DataArray, duration: float = None, padding_factor: int = None,
    overlap: float = None, dim: str = 'lag', **kwargs
):
    """
    Compute an N-D labelled spectrogram with consecutive Fourier transforms
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
        The array of data for which to compute the spectrogram.

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
    if dim not in x.dims:
        raise ValueError(f'x has no dimension "{dim}"')

    if 'sampling_rate' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{sampling_rate}"!')

    if 'delta' not in x[dim].attrs:
        raise ValueError('Dimension has no attribute "{delta}"!')

    padding_factor = padding_factor if (
        padding_factor and padding_factor >= 1
    ) else 1

    duration = duration if (
        duration and duration > x[dim].attrs['delta']
    ) else x[dim].attrs['delta']

    win_len = int(duration * x[dim].attrs['sampling_rate'])

    if win_len < 16:
        raise ValueError('Change duration to have at least 16 sample points!')

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
