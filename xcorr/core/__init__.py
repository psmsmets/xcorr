# -*- coding: utf-8 -*-
"""
xcorr.core init
"""

# Import main functions
from ..core.core import init, read, write, merge, mfread, process, bias_correct
from ..core.lazy import lazy_process


__all__ = ['init', 'read', 'write', 'merge', 'mfread', 'process',
           'bias_correct', 'lazy_process']
