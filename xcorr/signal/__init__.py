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
from ..signal.spectrogram import psd
from ..signal.stack import stack
from ..signal.taper import taper
from ..signal.unbias import unbias
from ..signal.window import window


__all__ = ['detrend', 'demean', 'filter', 'mask', 'multi_mask', 'psd', 'rms',
           'snr', 'stack', 'taper', 'unbias', 'window']
