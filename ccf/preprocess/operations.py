# -*- coding: utf-8 -*-
"""
Python module with crosscorrelation processing, waveform preprocessing and
crosscorrelation postprocessing routines.

.. module:: process

:author:
    Pieter Smets (P.S.M.Smets@tudelft.nl)

:copyright:
    Pieter Smets

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

# Mandatory imports
import warnings
import xarray as xr
import json
from obspy import Inventory


# Relative imports
from ..preprocess import running_rms
from ..utils import to_UTCDateTime


__all__ = ['stream_operations', 'is_stream_operation', '_inject_parameters',
           'apply_stream_operation', 'preprocess', 'example_operations',
           'add_operations_to_DataArray', 'get_operations_from_DataArray']


_stream_operations = {
    'decimate': {'method': 'self', 'inject': []},
    'detrend': {'method': 'self', 'inject': []},
    'filter': {'method': 'self', 'inject': []},
    'interpolate': {'method': 'self', 'inject': []},
    'merge': {'method': 'self', 'inject': []},
    'normalize': {'method': 'self', 'inject': []},
    'remove_response': {'method': 'self', 'inject': ['inventory']},
    'remove_sensitivity': {'method': 'self', 'inject': ['inventory']},
    'resample': {'method': 'self', 'inject': []},
    'rotate': {'method': 'self', 'inject': ['inventory']},
    'select': {'method': 'self', 'inject': []},
    'taper': {'method': 'self', 'inject': []},
    'trim': {'method': 'self', 'inject': ['starttime', 'endtime']},
    'running_rms': {'method': 'running_rms', 'inject': []},
}


def stream_operations():
    r"""
    Returns a list with all implemented stream operations.
    """
    return _stream_operations


def is_stream_operation(operation: str):
    r"""Verify if the operation is an implemented stream operation.

    Parameters
    ----------
    operation : str
        Operation name. See :func:`stream_operations` for a list of
        implemented operations.

    Returns
    -------
        Returns True of the `operation` is part of the implemented stream
        operations. Otherwise returns False.

    """
    return operation in _stream_operations


def _inject_parameters(
    operation: str, parameters: dict, inventory: Inventory = None,
    starttime=None, endtime=None
):
    r"""
    Inject starttime, endtime and inventory to the parameters dictionary
    when needed.
    """
    if (
        'inventory' in
        _stream_operations[operation]['inject']
    ):
        parameters['inventory'] = inventory
    if (
        'starttime' in
        _stream_operations[operation]['inject']
    ):
        parameters['starttime'] = to_UTCDateTime(starttime)
    if (
        'endtime' in
        _stream_operations[operation]['inject']
    ):
        parameters['endtime'] = to_UTCDateTime(endtime)
    return parameters


def apply_stream_operation(
    stream, operation: str, parameters: dict, verbose: bool = False
):
    """
    Apply a stream operation with the provided parameters.
    """
    if not is_stream_operation(operation):
        return
    method = _stream_operations[operation]['method']
    if verbose:
        print(operation, ': ', parameters)
    if method == 'self':
        return eval(f'stream.{operation}(**parameters)')
    else:
        return eval(f'{method}(stream,**parameters)')


def preprocess(
    stream, operations: list, inventory: Inventory = None,
    starttime=None, endtime=None, verbose: bool = False,
    debug: bool = False
):
    """
    Preprocess a `stream` (~obspy.Stream or ~obspy.Trace) given a list
    of operations.
    Optionally provide the `inventory`, `starttime`, `endtime` to inject
    the parameters.
    """
    st = stream.copy()
    for operation_params in operations:
        if (
            not(
                isinstance(operation_params, tuple) or
                isinstance(operation_params, list)
            ) or
            len(operation_params) != 2
        ):
            warnings.warn(
                (
                    'Provided operation should be a tuple or list with '
                    'length 2 (method:str,params:dict).'
                ),
                UserWarning
            )
            continue
        operation, parameters = operation_params
        if not is_stream_operation(operation):
            warnings.warn(
                'Provided operation "{}" is invalid thus ignored.'
                .format(operation),
                UserWarning
            )
            continue
        try:
            st = apply_stream_operation(
                stream=st,
                operation=operation,
                parameters=_inject_parameters(
                    operation, parameters, inventory, starttime, endtime
                ),
                verbose=verbose,
            )
        except Exception as e:
            warnings.warn(
                'Failed to execute operation "{}".'.format(operation),
                RuntimeWarning
            )
            if verbose:
                print(e)
        if debug:
            print(st)
    return st


def example_operations(to_json: bool = False):
    operations = {
        'BHZ': [
            ('merge', {
                'method': 1,
                'fill_value': 'interpolate',
                'interpolation_samples': 0,
            }),
            ('filter', {
                'type': 'highpass',
                'freq': .05,
            }),
            ('detrend', {
                'type': 'demean',
            }),
            ('remove_response', {
                'output': 'VEL',
            }),
            ('filter', {
                'type': 'highpass',
                'freq': 3.,
            }),
            ('interpolate', {
                'sampling_rate': 50,
                'method': 'lanczos',
                'a': 20,
            }),
            ('filter', {
                'type': 'lowpass',
                'freq': 20.,
            }),
            ('trim', {}),
            ('detrend', {
                'type': 'demean',
            }),
            ('taper', {
                'type': 'cosine',
                'max_percentage': 0.05,
                'max_length': 30.,
            }),
        ],
        'BHR': [
            ('merge', {
                'method': 1,
                'fill_value': 'interpolate',
                'interpolation_samples': 0,
            }),
            ('filter', {
                'type': 'highpass',
                'freq': .05,
            }),
            ('detrend', {
                'type': 'demean',
            }),
            ('remove_response', {
                'output': 'VEL',
            }),
            ('rotate', {
                'method': '->ZNE',
            }),
            ('rotate', {
                'method': 'NE->RT',
                'back_azimuth': 250.30,
            }),
            ('select', {
                'channel': 'BHR',
            }),
            ('filter', {
                'type': 'highpass',
                'freq': 3.,
            }),
            ('interpolate', {
                'sampling_rate': 50,
                'method': 'lanczos',
                'a': 20,
            }),
            ('filter', {
                'type': 'lowpass',
                'freq': 20.,
            }),
            ('trim', {}),
            ('detrend', {
                'type': 'demean',
            }),
            ('taper', {
                'type': 'cosine',
                'max_percentage': 0.05,
                'max_length': 30.,
            }),
        ],
        'EDH': [
            ('merge', {
                'method': 1,
                'fill_value': 'interpolate',
                'interpolation_samples': 0,
            }),
            ('detrend', {
                'type': 'demean',
            }),
            ('remove_sensitivity', {}),
            ('filter', {
                'type': 'bandpass',
                'freqmin': 3.,
                'freqmax': 20.,
            }),
            ('decimate', {
                'factor': 5,
            }),
            ('trim', {}),
            ('detrend', {
                'type': 'demean',
            }),
            ('taper', {
                'type': 'cosine',
                'max_percentage': 0.05,
                'max_length': 30.,
            }),
        ],
    }
    return json.dumps(operations) if to_json else operations


def add_operations_to_DataArray(
    darray: xr.DataArray, operations, channel: str = None,
    attribute: str = 'preprocess'
):
    """
    Add the preprocess dict{ channel : [operations] } `DataArray`.
    Optionally change the `attribute` name (default is `preprocess`).
    """
    if not (isinstance(operations, dict) or isinstance(operations, list)):
        raise TypeError('`operations` should be of type dict or list!')
    if isinstance(operations, list):
        if channel is None:
            raise ValueError('`channel` is not defined!')
        dops = dict()
        if attribute in darray.attrs:
            dops = json.loads(darray.attrs[attribute])
        dops[channel] = operations
        darray.attrs[attribute] = json.dumps(dops)
    else:
        darray.attrs[attribute] = json.dumps(operations)


def get_operations_from_DataArray(
    darray: xr.DataArray, channel: str, attribute: str = 'preprocess'
):
    """
    Get the list of preprocess operations from the `DataArray`
    attribute given the `channel`.
    Optionally provide the `attribute` name (default is `preprocess`).
    """
    assert attribute in darray.attrs, (
        'Attribue "{}" not found in DataArray!'.format(attribute)
    )
    return json.loads(darray.attrs[attribute][channel])
