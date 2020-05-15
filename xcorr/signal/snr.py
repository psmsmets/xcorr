r"""

:mod:`signal.snr` -- SNR
========================

Estimate the signal-to-noise ratio of an N-D labeled array of data.

"""


# Mandatory imports
import xarray as xr


# Relative imports
from ..util.history import historicize
from ..signal.rms import rms


__all__ = ['snr']


def snr(
    x: xr.DataArray, signal: xr.DataArray, noise: xr.DataArray,
    dim: str = 'lag'
):
    """
    Compute the signal-to-noise ratio of an N-D labeled array of data given a
    signal mask and a noise mask.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to compute the snr for.

    signal : :class:`xarray.DataArray`
        The boolean array masking the signal window of ``x``.

    noise : :class:`xarray.DataArray`
        The boolean array masking the noise window of ``x``.

    dim : `str`, optional
        The coordinates name of ``x`` to be filtered over. Default is 'lag'.

    Returns
    -------
    snr : :class:`xarray.DataArray`
        The snr output of ``x``.

    """
    s = xr.ufuncs.fabs(x.where(signal, drop=True))
    n = x.where(noise, drop=True)

    snr = s.max(dim=dim) / rms(n, dim=dim)
    snr.name = 'snr'
    snr.attrs = {
        'long_name': 'signal-to-noise ratio',
        'standard_name': 'signal_to_noise_ratio',
        'units': '-',
        'from_variable': f'{x.long_name} ({x.units})',
    }
    if 'history' in x.attrs:
        snr.attrs['history'] = x.history

    historicize(snr, f='snr', a={
        'x': x.name,
        'signal': signal.name,
        'noise': noise.name,
        'dim': dim,
    })

    return snr
