r"""

:mod:`signal.accessor` -- Accessor
==================================

Xarray DataArray accessor for xcorr.signal.

"""


# Mandatory imports
import xarray as xr
from functools import wraps


# Relative imports
from .absolute import abs as abs_
from .absolute import absolute as absolute_
from .beamform import plane_wave as plane_wave_
from .correlate import correlate1d as correlate1d_
from .correlate import correlate2d as correlate2d_
from .detrend import detrend as detrend_
from .detrend import demean as demean_
from .fft import fft as fft_
from .fft import ifft as ifft_
from .fft import rfft as rfft_
from .fft import irfft as irfft_
from .filter import filter as filter_
from .hilbert import hilbert as hilbert_
from .lombscargle import lombscargle as lombscargle_
from .normalize import norm1d as norm1d_
from .normalize import norm2d as norm2d_
from .rms import rms as rms_
from .snr import snr as snr_
from .spectrogram import spectrogram as spectrogram_
from .taper import taper as taper_
from .timeshift import timeshift as timeshift_
from .tri import tri_mirror as tri_mirror_
from .trigger import coincidence_trigger as coincidence_trigger_
from .unbias import unbias as unbias_
from .unbias import unbias_weights as unbias_weights_
from .window import window as window_


__all__ = []


@xr.register_dataarray_accessor('signal')
class SignalAccessor():
    """DataArray signal accessor
    """
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @wraps(abs_)
    def abs(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.abs`.
        Do not edit here.
        """
        return abs_(self._obj, *args, **kwargs)

    @wraps(absolute_)
    def absolute(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.absolute`.
        Do not edit here.
        """
        return absolute_(self._obj, *args, **kwargs)

    @wraps(coincidence_trigger_)
    def coincidence_trigger(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.coincidence_trigger`.
        Do not edit here.
        """
        return coincidence_trigger_(self._obj, *args, **kwargs)

    @wraps(correlate1d_)
    def correlate1d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate1d`.
        Do not edit here.
        """
        return correlate1d_(self._obj, *args, **kwargs)

    @wraps(correlate2d_)
    def correlate2d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.correlate2d`.
        Do not edit here.
        """
        return correlate2d_(self._obj, *args, **kwargs)

    @wraps(demean_)
    def demean(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.demean`.
        Do not edit here.
        """
        return demean_(self._obj, *args, **kwargs)

    @wraps(detrend_)
    def detrend(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.detrend`.
        Do not edit here.
        """
        return detrend_(self._obj, *args, **kwargs)

    @wraps(fft_)
    def fft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.fft`.
        Do not edit here.
        """
        return fft_(self._obj, *args, **kwargs)

    @wraps(ifft_)
    def ifft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.ifft`.
        Do not edit here.
        """
        return ifft_(self._obj, *args, **kwargs)

    @wraps(rfft_)
    def rfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rfft`.
        Do not edit here.
        """
        return rfft_(self._obj, *args, **kwargs)

    @wraps(irfft_)
    def irfft(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.irfft`.
        Do not edit here.
        """
        return irfft_(self._obj, *args, **kwargs)

    @wraps(filter_)
    def filter(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.filter`.
        Do not edit here.
        """
        return filter_(self._obj, *args, **kwargs)

    @wraps(hilbert_)
    def hilbert(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.hilbert`.
        Do not edit here.
        """
        return hilbert_(self._obj, *args, **kwargs)

    @wraps(lombscargle_)
    def lombscargle(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.lombscargle`.
        Do not edit here.
        """
        return lombscargle_(self._obj, *args, **kwargs)

    @wraps(norm1d_)
    def norm1d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm1d`.
        Do not edit here.
        """
        return norm1d_(self._obj, *args, **kwargs)

    @wraps(norm2d_)
    def norm2d(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.norm2d`.
        Do not edit here.
        """
        return norm2d_(self._obj, *args, **kwargs)

    @wraps(plane_wave_)
    def plane_wave_estimate(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.plane_wave`.
        Do not edit here.
        """
        return plane_wave_(self._obj, *args, **kwargs)

    @wraps(rms_)
    def rms(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.rms`.
        Do not edit here.
        """
        return rms_(self._obj, *args, **kwargs)

    @wraps(snr_)
    def snr(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.snr`.
        Do not edit here.
        """
        return snr_(self._obj, *args, **kwargs)

    @wraps(spectrogram_)
    def spectrogram(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.spectrogram`.
        Do not edit here.
        """
        return spectrogram_(self._obj, *args, **kwargs)

    @wraps(taper_)
    def taper(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.taper`.
        Do not edit here.
        """
        return taper_(self._obj, *args, **kwargs)

    @wraps(timeshift_)
    def timeshift(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.timeshift`.
        Do not edit here.
        """
        return timeshift_(self._obj, *args, **kwargs)

    @wraps(tri_mirror_)
    def tri_mirror(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.tri_mirror`.
        Do not edit here.
        """
        return tri_mirror_(self._obj, *args, **kwargs)

    @wraps(unbias_)
    def unbias(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.unbias`.
        Do not edit here.
        """
        return unbias_(self._obj, *args, **kwargs)

    @wraps(unbias_weights_)
    def unbias_weights(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.unbias_weights`.
        Do not edit here.
        """
        return unbias_weights_(self._obj, *args, **kwargs)

    @wraps(window_)
    def window(self, *args, **kwargs):
        """
        The docstring and signature are taken from
        :func:`~xcorr.signal.window`.
        Do not edit here.
        """
        return window_(self._obj, *args, **kwargs)
