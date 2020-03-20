# -*- coding: utf-8 -*-
"""
ccf

**ccf** is an open-source project containing tools to calculate
crosscorrelation functions. Results and meta data are stored as
xarray/netCDF4 following COARDS and CF standards.
`ccf` contains waveform preprocessing and crosscorrelation postprocessing
routines as well as a client wrapper to retrieve waveforms from local
and remote services.

:author:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

# import all modules
from ccf import utils
from ccf import clients
from ccf import cc 
from ccf import preprocess 
from ccf import postprocess 
from ccf import core

# Import all functions
from ccf.core import *

# Import some classes
from ccf.clients.clients import Client

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from ccf import *`.
__all__ = ['Client', 'core', 'postprocess']

# Version
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncom. changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from ccf.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version.
    # We could throw a warning here, but this case *should* be
    # rare. empymod should be installed properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')

