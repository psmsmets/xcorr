r"""

:mod:`signal.snr` -- SNR
========================

Estimate the signal-to-noise ratio of an N-D labelled array of data.

"""


# Mandatory imports
import numpy as np
import xarray as xr


# Relative imports
from ..util.history import historicize
from ..util.metadata import global_attrs
from .rms import rms
from .hilbert import hilbert
from .absolute import absolute


__all__ = ['snr']


def snr(
    x: xr.DataArray, signal: xr.DataArray, noise: xr.DataArray,
    dim: str = None, power: bool = False, decibels: bool = False,
    extend: bool = False, envelope: bool = False, **kwargs
):
    """
    Compute the signal-to-noise ratio (SNR or S/N) of an N-D labelled array of
    data given a signal mask and a noise mask.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to compute the snr ratio.

    signal : :class:`xarray.DataArray`
        The boolean array masking the signal window of ``x``.

    noise : :class:`xarray.DataArray`
        The boolean array masking the noise window of ``x``.

    dim : `str`, optional
        The coordinates name of ``x`` to compute the snr for. Defaults to the
        last dimension of ``x``.

    power : `bool`, optional
        Calculate the ratio of powers instead of the ratio of energies.
        Defaults to `False`.

    decibels : `bool`, optional
        Return the ``snr`` in decibels. Defaults to `False`.

    extend : `bool`, optional
        Return all parameters such as the maximum absolute signal and
        corresponding coordinate value and the root-mean-square noise value.
        Defaults to `False`.

    envelope : `bool`, optional
        Calculate the amplitude envelope of the signal part before locating the
        peak amplitude. The envelope is given by magnitude of the analytic
        signal. Defaults to `False`.

    **kwargs :
        Any additional keyword arguments are used to set the dataset global
        metadata attributes.

    Returns
    -------
    snr : :class:`xarray.DataArray` or :class:`xarray.Dataset`
        The computed snr for ``x`` or dataset with extended parameters if
        ``extend``.

    """
    dim = dim or x.dims[-1]
    if not isinstance(dim, str):
        raise TypeError('dim should be a string')
    if dim not in x.dims:
        raise ValueError(f'x has no dimensions "{dim}"')
    argmax = f'{dim}_s_max'

    if np.isnan(x).any() and envelope:
        raise ValueError('x contains NaN values')

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

    ds['n'] = rms(x.where(noise, drop=True), dim=dim)

    x = absolute(
        hilbert(x, dim=dim) if envelope else x
    ).where(signal, drop=True)

    ds[argmax] = x[dim].isel(**{dim: x.argmax(dim=dim).compute()}, drop=True)
    ds['s'] = x.sel(**{dim: ds[argmax]}, drop=True)

    ds['snr'] = ds.s/ds.n
    if decibels:
        ds['snr'] = 20 * np.log10(ds.snr)
        power = True
    elif power:
        ds['snr'] = np.square(ds.snr)

    ds.s.attrs = {
        **x.attrs,
        'long_name': f'{x.long_name} Signal',
        'standard_name': f'{x.standard_name}_signal',
        'units': x.units,
        'description': ('Signal energy is defined as the maximum absolute '
                        'value amplitude in the signal window'),
        'envelope': np.byte(envelope),
    }

    ds.n.attrs = {
        **x.attrs,
        'long_name': f'R{x.long_name} Noise',
        'standard_name': f'{x.standard_name}_noise',
        'units': x.units,
        'description': ('Noise energy is defined as the root mean square of '
                        'the noise window'),
    }

    ds.snr.attrs = {
        'long_name': 'Signal-to-noise ratio',
        'standard_name': 'signal_to_noise_ratio',
        'units': 'dB' if decibels else '-',
        'from_variable': f'{x.long_name} ({x.units})',
        'description': ('SNR is defined as the ratio of the signal {0} to the '
                        'noise {0}'.format('power' if power else 'energy')),
        'power': np.byte(power),
        'decibels': np.byte(decibels),
        'envelope': np.byte(envelope),
    }
    if 'history' in x.attrs:
        ds.snr.attrs['history'] = x.history

    historicize(ds.snr, f='snr', a={
        'x': x.name,
        'signal': signal.name,
        'noise': noise.name,
        'dim': dim,
        'power': power,
        'decibels': decibels,
        'extend': extend,
        'envelope': envelope,
        '**kwargs': kwargs,
    })

    return ds if extend else ds['snr']
