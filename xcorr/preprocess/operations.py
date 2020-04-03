r"""

:mod:`preprocess.operations` -- Preprocess operations
=====================================================

Preprocess a :class:`obspy.Stream` given a list of operations and parameters.

"""

# Mandatory imports
import warnings
import xarray as xr
import json
from obspy import Inventory, Stream


# Relative imports
from ..preprocess.running_rms import running_rms
from ..util import to_UTCDateTime, hash_obj


__all__ = ['help', 'stream_operations', 'is_stream_operation',
           'preprocess', 'example_operations_dict',
           'hash_operations', 'check_operations_hash',
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
    operation : `str`
        Operation name. See :func:`help` or :func:`stream_operations` for a
        list of implemented operations.

    Returns
    -------
    is_operation : `bool`
        Returns `True` if the ``operation`` is part of the implemented
        stream operations, `False` otherwise.

    """
    return operation in _stream_operations


def inject_dynamic_parameters(
    operation: str, parameters: dict, inventory: Inventory = None,
    starttime=None, endtime=None
):
    r"""Inject dynamic parameters to the static parameter dictionary
    when needed by the operation.

    Parameters
    ----------
    operation : `str`
        Operation name. See :func:`help` or :func:`stream_operations` for a
        list of implemented operations.

    parameters : `dict`
        Dictionary with static arguments for the operation.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    starttime : various, optional
        Start time of the stream, used for trimming and selecting the correct
        instrument response.

    endtime : various, optional
        End time of the stream, used for trimming and selecting the correct
        instrument response.

    Returns
    -------
    parameters : `dict`
        Dictionary with static and dynamic arguments for the operations.

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
    stream, operation: str, parameters: dict, verb: int = 0, prefix: str = ''
):
    r"""Apply an in-place operation with the provided parameters.

    Parameters
    ----------
    stream : :class:`obspy.Stream`
        Waveforms on which to apply the list of operations.

    operations : `list`
        List of operations. Each item is a tuple ('operation', {parameters}).
        Use :func:`help` to list all valid operations and its documentation.

    parameters : `dict`
        Dictionary with all arguments for the operation. If the ``operation``
        requires dynamic parameters you should inject them first using
        :func:`inject_dynamic_parameters`

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    stream : :class:`obspy.Stream`
        Waveforms after applying the list of operations.

    """
    if not is_stream_operation(operation):
        return
    method = _stream_operations[operation]['method']
    if verb > 0:
        print(prefix, operation, ': ', parameters)
    if method == _self:
        return eval(f'stream.{operation}(**parameters)')
    else:
        return method(stream, **parameters)


def preprocess(
    stream: Stream, operations: list, inventory: Inventory = None,
    starttime=None, endtime=None, verb: int = 0
):
    r"""Preprocess waveforms given a list of operations.

    Parameters
    ----------
    stream : :class:`obspy.Stream`
        Waveforms on which to apply the list of operations.

    operations : `list`
        List of operations. Each item is a tuple ('operation', {parameters}).
        Use :func:`help` to list all valid operations and their documentation.

    inventory : :class:`obspy.Inventory`, optional
        Inventory object, including the instrument response.

    starttime : various, optional
        Start time of the stream, used for trimming and selecting the correct
        instrument response.

    endtime : various, optional
        End time of the stream, used for trimming and selecting the correct
        instrument response.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    stream : :class:`obspy.Stream`
        Waveforms after applying the list of operations.

    """
    if verb > 0:
        print('Apply preprocessing operations:')
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
                parameters=inject_dynamic_parameters(
                    operation, parameters, inventory, starttime, endtime
                ),
                verb=verb,
                prefix=' *',
            )
        except Exception as e:
            warnings.warn(
                'Failed to execute operation "{}".'.format(operation),
                RuntimeWarning
            )
            if verb > 0:
                print(e)
        if verb > 2:
            print(st)
    return st


def example_operations_dict(to_json: bool = False):
    r"""Returns and example preprocessing operations dictionary, containing a
    list of operations per SEED channel as key.
    """
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
    r"""Only keep keys with 3 character channel codes starting with the known
    SEED channel band codes 'FGDCESHBMLVURPTQ'.
    """
    channels = [
        chan for chan in operations.keys() if (
            len(chan) == 3 and chan[0] in _channel_band_codes
        )
    ]
    return {chan: operations[chan] for chan in channels}


def hash_operations(
    operations: dict
):
    r"""
    Add a sha256 hash to the operations `dict`. ``operations`` are filtered
    before hashing using :func:`filter_operations`.
    """
    operations = filter_operations(operations)
    operations['sha256_hash'] = hash_obj(operations)
    return operations


def check_operations_hash(
    operations: dict, raise_error: bool = False
):
    r"""
    Returns `True` if the operations hash is valid and `False` otherwise
    if ``raise_error`` is `False` (default). Otherwise an error is raised.
    """
    if 'sha256_hash' not in operations:
        raise ValueError(
            'Preprocess operations does not contain a hash!'
        )
    sha256 = hash_obj(filter_operations(operations))
    if raise_error and operations['sha256_hash'] != sha256:
        raise ValueError(
            "Preprocess operations hash '{}' does not match the computed "
            "hash '{}'!".format(sha256, operations['sha256_hash'])
        )
    return operations['sha256_hash'] == sha256


def operations_to_dict(operations: str):
    r"""Load preprocess operations `dict` from a JSON-encoded attribute `str`.
    The sha256 hash is validated and ``operations`` keys are filtered for valid
    SEED channel codes.
    """
    operations = json.loads(operations)
    if 'sha256_hash' not in operations:
        raise ValueError(
            'Preprocess operations does not contain a hash!'
        )
    sha256 = operations['sha256_hash']
    operations = hash_operations(operations)
    if operations['sha256_hash'] != sha256:
        raise ValueError(
            "Preprocess operations hash '{}' does not match the computed "
            "hash '{}'!".format(sha256, operations['sha256_hash'])
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

    """
    attribute = attribute or 'preprocess'
    if isinstance(pair.attrs[attribute], dict):
        pair.attrs[attribute] = operations_to_json(pair.attrs[attribute])
