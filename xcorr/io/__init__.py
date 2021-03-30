# -*- coding: utf-8 -*-
"""
xcorr.io init
"""

# import all functions
from .filesystem import ncfile
from .read import read, mfread
from .validate import validate, validate_list
from .write import write


__all__ = ['ncfile', 'read', 'mfread', 'validate', 'validate_list', 'write']
