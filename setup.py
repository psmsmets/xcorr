#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ccf - Cross-Correlation Functions

**ccf** is an open-source project containing tools to calculate and store cross-correlation functions
automatically with proper documentation. Results, parameters and meta data are stored as xarray dataset
and stored to disk as a netCDF4 file, following COARDS and CF-1.7 conventions.
It contains pre- and postprocess routines and various clients to retrieve waveforms, all based on
the xarray dataset metadata.

:author:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:copyright:
    Pieter Smets (p.s.m.smets@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import os
import sys
from glob import glob
from setuptools import setup, find_namespace_packages


SETUP_DIRECTORY = os.path.abspath('./')
name = 'ccf'

DOCSTRING = __doc__.split('\n')

KEYWORDS = [
    'ccf', 'cross-correlation', 'signal-processing', 'waveform', 'stacking', 'preprocess', 'postprocess'
]

INSTALL_REQUIRES = [
    'numpy>=1.16',
    'obspy>=1.1.0',
    'xarray>=0.14.0',
    'pandas>=0.24.0',
    'pyfftw>=0.11.0',
]

ENTRY_POINTS = {
}

INSTALL_SCRIPTS = [
]

def setup_package():

    # setup package
    setup(
        name=name,
        python_requires='>3.5.2',
        description=DOCSTRING[1],
        long_description='\n'.join(DOCSTRING[3:]),
        author=[
            'Pieter Smets'
        ],
        author_email='p.s.m.smets@tudelft.nl',
        url='https://gitlab.com/psmsmets/ccf',
        download_url='https://gitlab.com/psmsmets/ccf.git',
        install_requires=INSTALL_REQUIRES,
        keywords=KEYWORDS,
        packages=find_namespace_packages(include=['ccf.*']),
        entry_points=ENTRY_POINTS,
        scripts=INSTALL_SCRIPTS,
        zip_safe=False,
        use_scm_version={
            'root': '.',
            'relative_to': __file__,
            'write_to': os.path.join(name, 'version.py'),
        },
        setup_requires=['setuptools_scm'],
    )


def clean():
    """
    Make sure to start with a fresh install
    """
    import shutil

    # delete complete build directory and egg-info
    path = os.path.join(SETUP_DIRECTORY, 'build')
    try:
        shutil.rmtree(path)
        print('removed ', path)
    except Exception:
        pass
    # delete complete build directory and egg-info
    path = os.path.join(SETUP_DIRECTORY, 'dist')
    try:
        shutil.rmtree(path)
        print('removed ', path)
    except Exception:
        pass
    # delete egg-info dir
    path = os.path.join(SETUP_DIRECTORY, name + '.egg-info')
    try:
        shutil.rmtree(path)
        print('removed ', path)
    except Exception:
        pass
    # delete __pycache__
    for path in glob(os.path.join(name, '**', '__pycache__'),
                     recursive=True):
        try:
            shutil.rmtree(path)
            print('removed ', path)
        except Exception:
            pass


if __name__ == '__main__':
    if 'clean' in sys.argv:
        clean()
    else:
        clean()
        setup_package()
