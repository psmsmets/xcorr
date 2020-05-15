# -*- coding: utf-8 -*-
"""
xcorr

**xcorr** is an open-source project containing tools to crosscorrelate
waveform timeseries as :class:`obspy.Stream` and stored as a self describing
`xarray.Dataset`.

Results and metadata are stored netCDF4 following CF-1.9 Conventions and
FAIR data guidelines.

`xcorr` contains various modules such as waveform preprocessing, a client
waterfall-based wrapping various getters from local archives as well as
remote services, frequency domain crosscorrelationm and postprocessing/
analysis tools.

:author:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

# Import main modules
from xcorr import cc, preprocess, signal

# Import all core functions
from xcorr.core import (init, validate, read, write, merge, mfread, process,
                        lazy_process, bias_correct)

# Import client class
from xcorr.client import Client

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from xcorr import *`.
__all__ = ['Client', 'cc', 'signal', 'preprocess', 'init', 'validate', 'read',
           'write', 'merge', 'mfread', 'process', 'lazy_process',
           'bias_correct']

# Version
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncom. changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from xcorr.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version.
    # We could throw a warning here, but this case *should* be
    # rare. empymod should be installed properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
