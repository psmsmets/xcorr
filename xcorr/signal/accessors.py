r"""

:mod:`signal.accessors` -- Accessors
====================================

Xarray Dataset and DataArray accessors for xcorr.signal.

"""


# Mandatory imports
import xarray as xr
from functools import wraps


# Relative imports
from ..signal.absolute import abs, absolute
from ..signal.beamform import plane_wave
from ..signal.correlate import correlate1d, correlate2d
from ..signal.detrend import detrend, demean
from ..signal.fft import fft, ifft, rfft, irfft
from ..signal.filter import filter
from ..signal.hilbert import hilbert
from ..signal.lombscargle import lombscargle
from ..signal.normalize import norm1d, norm2d
from ..signal.rms import rms
from ..signal.snr import snr
from ..signal.spectrogram import spectrogram
from ..signal.taper import taper
from ..signal.timeshift import timeshift
from ..signal.tri import tri_mask, tri_mirror
from ..signal.trigger import coincidence_trigger
from ..signal.unbias import unbias
from ..signal.window import window


__all__ = []


@xr.register_dataarray_accessor('signal')
class SignalAccessor():
    """DataArray signal accessors
    """
    def __init__(self, da: xr.DataArray):
        self._da = da

    @wraps(absolute)
    def abs(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.abs`.
        Do not edit here.
        """
        return abs(self._da, *args, **kwargs)

    @wraps(absolute)
    def absolute(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.absolute`.
        Do not edit here.
        """
        return absolute(self._da, *args, **kwargs)

    @wraps(coincidence_trigger)
    def coincidence_trigger(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.coincidence_trigger`.
        Do not edit here.
        """
        return coincidence_trigger(self._da, *args, **kwargs)

    @wraps(correlate1d)
    def correlate1d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate1d`.
        Do not edit here.
        """
        return correlate1d(self._da, *args, **kwargs)

    @wraps(correlate2d)
    def correlate2d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate2d`.
        Do not edit here.
        """
        return correlate2d(self._da, *args, **kwargs)

    @wraps(demean)
    def demean(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.demean`.
        Do not edit here.
        """
        return demean(self._da, *args, **kwargs)

    @wraps(detrend)
    def detrend(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.detrend`.
        Do not edit here.
        """
        return detrend(self._da, *args, **kwargs)

    @wraps(fft)
    def fft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.fft`.
        Do not edit here.
        """
        return fft(self._da, *args, **kwargs)

    @wraps(ifft)
    def ifft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.ifft`.
        Do not edit here.
        """
        return ifft(self._da, *args, **kwargs)

    @wraps(rfft)
    def rfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rfft`.
        Do not edit here.
        """
        return rfft(self._da, *args, **kwargs)

    @wraps(irfft)
    def irfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.irfft`.
        Do not edit here.
        """
        return irfft(self._da, *args, **kwargs)

    @wraps(filter)
    def filter(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.filter`.
        Do not edit here.
        """
        return filter(self._da, *args, **kwargs)

    @wraps(hilbert)
    def hilbert(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.hilbert`.
        Do not edit here.
        """
        return hilbert(self._da, *args, **kwargs)

    @wraps(lombscargle)
    def lombscargle(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.lombscargle`.
        Do not edit here.
        """
        return lombscargle(self._da, *args, **kwargs)

    @wraps(norm1d)
    def norm1d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm1d`.
        Do not edit here.
        """
        return norm1d(self._da, *args, **kwargs)

    @wraps(norm2d)
    def norm2d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm2d`.
        Do not edit here.
        """
        return norm2d(self._da, *args, **kwargs)

    @wraps(plane_wave)
    def plane_wave_estimate(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.plane_wave`.
        Do not edit here.
        """
        return plane_wave(self._da, *args, **kwargs)

    @wraps(rms)
    def rms(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rms`.
        Do not edit here.
        """
        return rms(self._da, *args, **kwargs)

    @wraps(snr)
    def snr(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.snr`.
        Do not edit here.
        """
        return snr(self._da, *args, **kwargs)

    @wraps(spectrogram)
    def spectrogram(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.spectrogram`.
        Do not edit here.
        """
        return spectrogram(self._da, *args, **kwargs)

    @wraps(taper)
    def taper(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.taper`.
        Do not edit here.
        """
        return taper(self._da, *args, **kwargs)

    @wraps(timeshift)
    def timeshift(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.timeshift`.
        Do not edit here.
        """
        return timeshift(self._da, *args, **kwargs)

    @wraps(tri_mask)
    def tri_mask(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.tri_mask`.
        Do not edit here.
        """
        return tri_mask(self._da, *args, **kwargs)

    @wraps(tri_mirror)
    def tri_mirror(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.tri_mirror`.
        Do not edit here.
        """
        return tri_mirror(self._da, *args, **kwargs)

    @wraps(unbias)
    def unbias(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.unbias`.
        Do not edit here.
        """
        return unbias(self._da, *args, **kwargs)

    @wraps(window)
    def window(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.window`.
        Do not edit here.
        """
        return window(self._da, *args, **kwargs)
