# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup, find_namespace_packages

# Get README and remove badges.
README = open('README.rst').read()
README = re.sub('----.*marker', '----', README, flags=re.DOTALL)

DESCRIPTION = 'Correlate timeseries in a self-describing N-D labeled dataset'

setup(
    name='xcorr',
    python_requires='>3.5.0',
    description=DESCRIPTION,
    long_description=README,
    author='Pieter Smets',
    author_email='mail@pietersmets.be',
    url='https://gitlab.com/psmsmets/xcorr',
    download_url='https://gitlab.com/psmsmets/xcorr.git',
    license='MIT License',
    packages=find_namespace_packages(include=['xcorr.*']),
    keywords=[
        'xcorr', 'correlation', 'signal-processing', 'timeseries',
        'stacking', 'obspy', 'xarray'
    ],
    entry_points={},
    scripts=[],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.0.0',
        'obspy>=1.1.0',
        'xarray>=0.14.0',
        'pandas>=0.24.0',
    ],
    use_scm_version={
        'root': '.',
        'relative_to': __file__,
        'write_to': os.path.join('xcorr', 'version.py'),
    },
    setup_requires=['setuptools_scm'],
)
