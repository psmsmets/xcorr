r"""

:mod:`signal.accessors` -- Accessors
====================================

Xarray Dataset and DataArray accessors for xcorr.signal.

"""


# Mandatory imports
import xarray as xr
from functools import wraps


# Relative imports
from .. import signal


__all__ = []


@xr.register_dataarray_accessor('signal')
class SignalAccessor():
    """DataArray signal accessors
    """
    def __init__(self, da: xr.DataArray):
        self._da = da

    @wraps(signal.absolute)
    def absolute(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.absolute`.
        Do not edit here.
        """
        return signal.absolute(self._da, *args, **kwargs)

    @wraps(signal.coincidence_trigger)
    def coincidence_trigger(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.coincidence_trigger`.
        Do not edit here.
        """
        return signal.coincidence_trigger(self._da, *args, **kwargs)

    @wraps(signal.correlate2d)
    def detrend(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate2d`.
        Do not edit here.
        """
        return signal.correlate2d(self._da, *args, **kwargs)

    @wraps(signal.correlate1d)
    def detrend(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate1d`.
        Do not edit here.
        """
        return signal.correlate1d(self._da, *args, **kwargs)

    @wraps(signal.correlate2d)
    def detrend(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate2d`.
        Do not edit here.
        """
        return signal.correlate2d(self._da, *args, **kwargs)

    @wraps(signal.fft)
    def fft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.fft`.
        Do not edit here.
        """
        return signal.fft(self._da, *args, **kwargs)

    @wraps(signal.ifft)
    def ifft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.ifft`.
        Do not edit here.
        """
        return signal.ifft(self._da, *args, **kwargs)

    @wraps(signal.rfft)
    def rfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rfft`.
        Do not edit here.
        """
        return signal.rfft(self._da, *args, **kwargs)

    @wraps(signal.irfft)
    def irfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.irfft`.
        Do not edit here.
        """
        return signal.irfft(self._da, *args, **kwargs)

    @wraps(signal.filter)
    def filter(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.filter`.
        Do not edit here.
        """
        return signal.filter(self._da, *args, **kwargs)

    @wraps(signal.hilbert)
    def hilbert(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.hilbert`.
        Do not edit here.
        """
        return signal.hilbert(self._da, *args, **kwargs)

    @wraps(signal.lombscargle)
    def lombscargle(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.lombscargle`.
        Do not edit here.
        """
        return signal.lombscargle(self._da, *args, **kwargs)

    @wraps(signal.norm1d)
    def norm1d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm1d`.
        Do not edit here.
        """
        return signal.norm1d(self._da, *args, **kwargs)

    @wraps(signal.norm2d)
    def norm2d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm2d`.
        Do not edit here.
        """
        return signal.norm2d(self._da, *args, **kwargs)

    @wraps(signal.rms)
    def rms(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rms`.
        Do not edit here.
        """
        return signal.rms(self._da, *args, **kwargs)

    @wraps(signal.snr)
    def snr(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.snr`.
        Do not edit here.
        """
        return signal.snr(self._da, *args, **kwargs)

    @wraps(signal.spectrogram)
    def spectrogram(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.spectrogram`.
        Do not edit here.
        """
        return signal.spectrogram(self._da, *args, **kwargs)

    @wraps(signal.taper)
    def taper(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.taper`.
        Do not edit here.
        """
        return signal.taper(self._da, *args, **kwargs)

    @wraps(signal.timeshift)
    def timeshift(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.timeshift`.
        Do not edit here.
        """
        return signal.timeshift(self._da, *args, **kwargs)

    @wraps(signal.tri_mask)
    def tri_mask(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.tri_mask`.
        Do not edit here.
        """
        return signal.tri_mask(self._da, *args, **kwargs)

    @wraps(signal.tri_mirror)
    def tri_mirror(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.tri_mirror`.
        Do not edit here.
        """
        return signal.tri_mirror(self._da, *args, **kwargs)

    @wraps(signal.unbias)
    def unbias(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.unbias`.
        Do not edit here.
        """
        return signal.unbias(self._da, *args, **kwargs)

    @wraps(signal.window)
    def window(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.window`.
        Do not edit here.
        """
        return signal.window(self._da, *args, **kwargs)
