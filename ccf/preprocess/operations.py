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
from hashlib import sha256


# Relative imports
from ..preprocess.running_rms import running_rms
from ..utils import to_UTCDateTime


__all__ = ['help', 'stream_operations', 'is_stream_operation',
           'apply_stream_operation', 'preprocess', 'example_operations',
           'filter_operations', 'hash_operations', 'check_operations_hash',
           'operations_to_dict', 'operations_to_json',
           'preprocess_operations_to_json', 'preprocess_operations_to_dict']

_self = 'obspy.core.stream.Stream'

_stream_operations = {
    'decimate': {'method': _self, 'inject': []},
    'detrend': {'method': _self, 'inject': []},
    'filter': {'method': _self, 'inject': []},
    'interpolate': {'method': _self, 'inject': []},
    'merge': {'method': _self, 'inject': []},
    'normalize': {'method': _self, 'inject': []},
    'remove_response': {'method': _self, 'inject': ['inventory']},
    'remove_sensitivity': {'method': _self, 'inject': ['inventory']},
    'resample': {'method': _self, 'inject': []},
    'rotate': {'method': _self, 'inject': ['inventory']},
    'select': {'method': _self, 'inject': []},
    'taper': {'method': _self, 'inject': []},
    'trim': {'method': _self, 'inject': ['starttime', 'endtime']},
    'running_rms': {'method': running_rms, 'inject': []},
}


def stream_operations():
    r"""
    Returns a list of implemented stream operations.
    """
    return list(_stream_operations.keys())


def help(operation: str = None):
    r"""
    Print a more extensive help for a given operation.
    """
    if operation is None:
        operations = list(_stream_operations.keys())
    elif operation in _stream_operations:
        operations = [operation]
    else:
        msg = (
            "Operation '{}' not available as stream operation."
            .format(operation)
        )
        raise ValueError(msg)

    msg = []
    for operation in operations:
        method = _stream_operations[operation]['method']
        inject = _stream_operations[operation]['inject']

        msg.append("Operation '{}'".format(operation))
        msg.append("          method : {}".format(
            method + '.' + operation if method == _self else method.__module__
        ))
        msg.append("   injected args : {}".format(
            ', '.join(inject)
        ))
        msg.append('')

    print("\n".join(msg))


def is_stream_operation(operation: str):
    r"""Verify if the operation is an implemented stream operation.

    Parameters
    ----------
    operation : str
        Operation name. See :func:`stream_operations` for a list of
        implemented operations.

    Returns
    -------
        Returns `True` of the ``operation`` is part of the implemented
        stream operations, `False` otherwise.

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
    params = parameters.copy()
    if (
        'inventory' in
        _stream_operations[operation]['inject']
    ):
        params['inventory'] = inventory
    if (
        'starttime' in
        _stream_operations[operation]['inject']
    ):
        params['starttime'] = to_UTCDateTime(starttime)
    if (
        'endtime' in
        _stream_operations[operation]['inject']
    ):
        params['endtime'] = to_UTCDateTime(endtime)
    return params


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
    if method == _self:
        return eval(f'stream.{operation}(**parameters)')
    else:
        return method(stream, **parameters)
        # return eval(f'{method}(stream, **parameters)')


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


_channel_band_codes = 'FGDCESHBMLVURPTQ'


def filter_operations(
    operations: dict
):
    """
    Only keep keys with 3 character channel codes starting with the known
    SEED channel band codes 'FGDCESHBMLVURPTQ'.
    """  
    channels = [chan for chan in operations.keys() if (len(chan) == 3 and chan[0] in _channel_band_codes)]
    return { chan: operations[chan] for chan in channels }


def _generate_sha256_hash(
    var
):
    r"""Generate the sha256 hash on a :func:`json.dumps`.

    Parameters
    ----------
    var : str, list, tuple or dict
        Input variable to compute the sha256 hash.

    Returns
    -------
    hash : str
        Hexdigested hash object. `var` is dumped to json before hashing.
 
    """
    hash_obj = sha256(str(json.dumps(var, indent=4)).encode('ascii'))
    return hash_obj.hexdigest()


def hash_operations(
    operations: dict
):
    r"""
    Add a sha256 hash to the operations `dict`. ``operations`` are filtered
    before hashing using :func:`filter_operations`.
    """
    operations = filter_operations(operations)
    operations['sha256'] = _generate_sha256_hash(operations) 
    return operations


def check_operations_hash(
    operations: dict, raise_error: bool = False
):
    r"""
    Returns `True` if the operations hash is valid and `False` otherwise
    if ``raise_error`` is `False` (default). Otherwise an error is raised.
    """
    if 'sha256' not in operations:
        raise ValueError(
            'Preprocess operations does not contain a hash!'
        )
    sha256 = _generate_sha256_hash(filter_operations(operations))
    if raise_error and operations['sha256'] != sha256:
        raise ValueError(
            'Preprocess operations `str` contains an invalid hash!'
        )
    return operations['sha256'] == sha256


def operations_to_dict(operations: str):
    r"""Load preprocess operations `dict` from a JSON-encoded attribute `str`.
    The sha256 hash is validated and ``operations`` keys are filtered for valid
    SEED channel codes.
    """
    operations = json.loads(operations)
    if 'sha256' not in operations:
        raise ValueError(
            'Preprocess operations does not contain a hash!'
        )
    sha256 = operations['sha256']
    operations = hash_operations(operations)
    if operations['sha256'] != sha256:
        raise ValueError(
            'Preprocess operations `str` contains an invalid hash!'
        )
    return operations


def operations_to_json(operations: dict):
    r"""Convert preprocess operations from `dict` to a JSON `str`.
    ``operations`` keys are filtered for valid SEED channel codes and a
    sha256 hash is added or updated.
    """
    return json.dumps(hash_operations(operations))


def preprocess_operations_to_dict(pair: xr.DataArray, attribute: str = None):
    r"""Convert ``pair`` preprocess operations attribute inplace from a
    JSON `str` to a `dict`. The operations hash is verified after loading
    the json SEED channel codes and hashed.

    Parameters
    ----------
    pair : :class:`xarray.DataArray`
        Receiver pair couple separated by `separator`.
        Each receiver is specified by a SEED-id string:
        '{network}.{station}.{location}.{channel}'.

    attribute : str, optional
        Specify the operations attribute name. If None, ``attribute`` is
        'preprocess' (default).

    Returns
    -------
    None

    """
    attribute = attribute or 'preprocess'
    if isinstance(pair.attrs[attribute], str):
        pair.attrs[attribute] = operations_to_dict(pair.attrs[attribute])


def preprocess_operations_to_json(pair: xr.DataArray, attribute: str = None):
    r"""Convert ``pair`` preprocess operations attribute inplace from a `dict`
    to a netCDF4 safe JSON `str`. Operations channels are first filtered for
    valid SEED channel codes and hashed.

    Parameters
    ----------
    pair : :class:`xarray.DataArray`
        Receiver pair couple separated by `separator`.
        Each receiver is specified by a SEED-id string:
        '{network}.{station}.{location}.{channel}'.

    attribute : str, optional
        Specify the operations attribute name. If None, ``attribute`` is
        'preprocess' (default).

    Returns
    -------
    None

    """
    attribute = attribute or 'preprocess'
    if isinstance(pair.attrs[attribute], dict):
        pair.attrs[attribute] = operations_to_json(pair.attrs[attribute])
