# -*- coding: utf-8 -*-
"""
xcorr

**xcorr** is an open-source project containing tools to cross-correlate
waveform timeseries as :class:`obspy.Stream` and stored as a self describing
`xarray.Dataset`.

Results and metadata are stored netCDF4 following CF-1.9 Conventions and
FAIR data guidelines.

`xcorr` contains various modules such as waveform preprocessing, a client
waterfall-based wrapping various getters from local archives as well as
remote services, frequency domain cross-correlation and postprocessing/
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
# Check for Dask
import importlib as _importlib
__dask_spec = ((_importlib.util.find_spec("dask") is not None) and
               (_importlib.util.find_spec("distributed") is not None))

# Import main modules
from . import signal, stream, util, io

# Import all core functions
from .core import (init, process, postprocess, merge)
from .io import (read, write, mfread)

# Import client class
from .stream.client import Client

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from xcorr import *`.
__all__ = ['Client', 'signal', 'stream', 'util', 'io', 'postprocess',
           'init', 'read', 'write', 'merge', 'mfread', 'process']

# Load optional functions related to Dask
if __dask_spec:
    from .core import lazy_process
    __all__ += ['lazy_process']

# Version
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncom. changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version.
    # We could throw a warning here, but this case *should* be
    # rare. empymod should be installed properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
