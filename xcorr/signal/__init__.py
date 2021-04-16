# -*- coding: utf-8 -*-
"""
xcorr.signal init
"""

# Import all main functions
from .absolute import abs, absolute
from .beamform import plane_wave
from .correlate import correlate1d, correlate2d
from .detrend import detrend, demean
from .fft import fft, ifft, rfft, irfft
from .filter import filter
from .hilbert import hilbert, envelope
from .lombscargle import lombscargle
from .normalize import norm1d, norm2d
from .peak_local_max import peak_local_max
from .rms import rms
from .snr import snr
from .spectrogram import spectrogram
from .taper import taper
from .timeshift import timeshift
from .tri import tri_mask, tri_mirror
from .trigger import coincidence_trigger
from .unbias import unbias, unbias_weights
from .window import window

# Import accessor class
from .accessor import SignalAccessor


__all__ = []
