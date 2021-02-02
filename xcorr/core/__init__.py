# -*- coding: utf-8 -*-
"""
xcorr.core init
"""

# Import main functions
from ..core.core import (init, read, write, merge, mfread, process, 
                         validate, validate_list, bias_correct)
try:
    from ..core.lazy import lazy_process
except ModuleNotFoundError:
    lazy_process = None


__all__ = ['init', 'read', 'write', 'merge', 'mfread', 'process',
           'validate', 'validate_list', 'bias_correct', 'lazy_process']
