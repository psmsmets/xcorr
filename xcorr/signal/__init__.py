# -*- coding: utf-8 -*-
"""
xcorr.signal init
"""

# Import all main functions
from ..signal.detrend import detrend, demean
from ..signal.filter import filter
from ..signal.mask import mask, multi_mask
from ..signal.rms import rms
from ..signal.snr import snr
from ..signal.spectrogram import spectrogram
from ..signal.taper import taper
from ..signal.trigger import coincidence_trigger
from ..signal.unbias import unbias
from ..signal.window import window


__all__ = ['coincidence_trigger', 'detrend', 'demean', 'filter',
           'mask', 'multi_mask', 'rms', 'snr', 'spectrogram',
           'taper', 'unbias', 'window']
