# -*- coding: utf-8 -*-
"""
ccf

**ccf** is an open-source project containing tools to calculate
crosscorrelation functions. Results and meta data are stored as
xarray/netCDF4 following COARDS and CF standards.
`ccf` contains waveform postprocessing and crosscorrelation postprocessing
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
from ..clients import datafetch
from ..clients import clients

# Import all functions
from ..clients.datafetch import *
from ..clients.clients import *

# Import some classes
from ..clients.clients import Client
