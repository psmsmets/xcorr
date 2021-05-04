# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup, find_namespace_packages

# Get README and remove badges.
README = open('README.rst').read()
README = re.sub('----.*marker', '----', README, flags=re.DOTALL)

DESCRIPTION = 'Correlate timeseries in a self-describing N-D labelled dataset'

NAME = 'xcorr'

setup(
    name=NAME,
    python_requires='>=3.7.0',
    description=DESCRIPTION,
    long_description=README,
    author='Pieter Smets',
    author_email='mail@pietersmets.be',
    url='https://gitlab.com/psmsmets/xcorr',
    download_url='https://gitlab.com/psmsmets/xcorr.git',
    license='GNU General Public License v3 (GPLv3)',
    packages=find_namespace_packages(include=['xcorr.*']),
    keywords=[
        'xcorr', 'correlation', 'cross-correlation', 'signal-processing',
        'timeseries', 'waveforms', 'obspy', 'xarray'
    ],
    entry_points={
        'console_scripts': [
           f'{NAME}-snr=xcorr.scripts.snr:main',
           f'{NAME}-psd=xcorr.scripts.psd:main',
           f'{NAME}-psdmax=xcorr.scripts.psdmax:main',
           f'{NAME}-ct=xcorr.scripts.ct:main',
           f'{NAME}-timelapse=xcorr.scripts.timelapse:main',
           f'{NAME}-beamform=xcorr.scripts.beamform:main',
           f'{NAME}-swresp=xcorr.scripts.swresp:main',
        ],
    },
    scripts=[],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        ('License :: OSI Approved :: '
         'GNU General Public License v3 (GPLv3)'),
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.0',
        'obspy>=1.2',
        'xarray>=0.16',
        'pandas>=1.0',
        'pyproj>=2.0',
        'tabulate>=0.8',
        'tables>=3.2.3',
    ],
    use_scm_version={
        'root': '.',
        'relative_to': __file__,
        'write_to': os.path.join('xcorr', 'version.py'),
    },
    setup_requires=['setuptools_scm'],
)
