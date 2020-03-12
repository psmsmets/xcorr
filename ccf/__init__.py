# -*- coding: utf-8 -*-
"""
ccf

**ccf** is an open-source project containing tools to calculate
crosscorrelation functions.
Results and meta data are stored as xarray/netCDF4 following COARDS
and CF-1.7 standards.
It contains pre- and postprocess routines and various clients to retrieve
waveforms.

:author:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from ccf.helpers import Helpers as helpers
from ccf.process import CC as cc
from ccf.process import Preprocess as preprocess
from ccf.process import Postprocess as postprocess
from ccf.clients import Clients as clients
from ccf.core import (
    write_dataset, open_dataset, init_dataset, cc_dataset,
    bias_correct_dataset, get_dataset_weights
)

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

