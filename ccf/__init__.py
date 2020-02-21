# -*- coding: utf-8 -*-
"""
ccf

**ccf** is an open-source project containing tools to calculate cross-correlation functions.
Results and meta data are stored as xarray/netCDF4 following COARDS and CF-1.7 standards.
It contains pre- and postprocess routines and various clients to retrieve waveforms.

:author:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from __future__ import absolute_import, print_function, division

#from ccf.core import toUTCDateTime, init_dataset, cc_dataset, write_dataset, bias_correct_dataset, get_dataset_weights
from ccf.core import *
from ccf.clients import Clients as clients
from ccf.process import Preprocess as preprocess
from ccf.process import Postprocess as postprocess
from ccf.process import CC as cc

__version__ = '0.1.0'
