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
from ..preprocess import operations
from ..preprocess import running_rms

# Import __all__ functions
from ..preprocess.operations import *
from ..preprocess.running_rms import *
