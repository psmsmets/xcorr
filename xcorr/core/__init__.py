# -*- coding: utf-8 -*-
"""
xcorr.core init
"""

# Import main functions
from .accessors import XcorrAccessor
from .init import init
from .merge import merge
from .process import process
from .postprocess import postprocess
from . import plot

try:
    from ..core.lazy import lazy_process
except ModuleNotFoundError:
    lazy_process = None


__all__ = ['init', 'merge', 'process', 'lazy_process', 'postprocess', 'plot']
