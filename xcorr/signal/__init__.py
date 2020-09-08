# -*- coding: utf-8 -*-
"""
xcorr.signal init
"""

# Import all main functions
from ..signal.correlate import correlate2d
from ..signal.detrend import detrend, demean
from ..signal.fft import fft, ifft
from ..signal.filter import filter
from ..signal.lombscargle import lombscargle
from ..signal.mask import mask, multi_mask
from ..signal.normalize import norm1d, norm2d
from ..signal.rms import rms
from ..signal.snr import snr
from ..signal.spectrogram import spectrogram
from ..signal.taper import taper
from ..signal.timeshift import timeshift
from ..signal.trigger import coincidence_trigger
from ..signal.unbias import unbias
from ..signal.window import window


__all__ = ['coincidence_trigger', 'correlate2d', 'detrend', 'demean',
           'fft', 'ifft', 'filter', 'lombscargle', 'mask', 'multi_mask',
           'norm1d', 'norm2d', 'rms', 'snr', 'spectrogram', 'taper',
           'timeshift', 'unbias', 'window']
