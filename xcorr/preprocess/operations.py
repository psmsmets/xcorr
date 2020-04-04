r"""

:mod:`preprocess.operations` -- Preprocess operations
=====================================================

Preprocess a :class:`obspy.Stream` given a list of operations and parameters.

"""

# Mandatory imports
import warnings
import xarray as xr
import json
from obspy import Inventory, Trace, Stream


# Relative imports
from ..preprocess.running_rms import running_rms
from ..util import to_UTCDateTime, hash_obj


__all__ = ['help', 'list_operations', 'is_operation', 'preprocess',
           'example_preprocess_dict', 'hash_operations',
           'check_operations_hash', 'operations_to_dict', 'operations_to_json',
           'preprocess_operations_to_json', 'preprocess_operations_to_dict']

_self = 'obspy.core.stream.Trace'

_operations = {
    'attach_response': {
        'method': _self,
        'inject': ['inventory'],
    },
    'decimate': {
        'method': _self,
        'inject': [],
    },
    'detrend': {
        'method': _self,
        'inject': [],
    },
    'filter': {
        'method': _self,
        'inject': [],
    },
    'interpolate': {
        'method': _self,
        'inject': [],
    },
    'merge': {
        'method': _self,
        'inject': [],
    },
    'normalize': {
        'method': _self,
        'inject': [],
    },
    'remove_response': {
        'method': _self,
        'inject': ['inventory'],
    },
    'remove_sensitivity': {
        'method': _self,
        'inject': ['inventory'],
    },
    'resample': {
        'method': _self,
        'inject': [],
    },
    'rotate': {
        'method': _self,
        'inject': ['inventory'],
    },
    'select': {
        'method': _self,
        'inject': [],
    },
    'simulate': {
        'method': _self,
        'inject': [],
    },
    'taper': {
        'method': _self,
        'inject': [],
    },
    'trim': {
        'method': _self,
        'inject': ['starttime', 'endtime'],
    },
    'running_rms': {
        'method': running_rms,
        'inject': [],
    },
}


def list_operations():
    r"""
    Returns a list of implemented stream operations.
    """
    return list(_operations.keys())


def help(operation: str = None):
    r"""
    Print a more extensive help for a given operation.
    """
    if operation is None:
        operations = list(_operations.keys())
    elif operation in _operations:
        operations = [operation]
    else:
        msg = ('Operation "{}" not available as stream operation.'
               .format(operation))
        raise ValueError(msg)

    msg = []
    for operation in operations:
        method = _operations[operation]['method']
        inject = _operations[operation]['inject']

        msg.append("Operation '{}'".format(operation))
        msg.append("          method : {}".format(
            method + '.' + operation if method == _self else method.__module__
        ))
        msg.append("   injected args : {}".format(
            ', '.join(inject)
        ))
        msg.append('')

    print("\n".join(msg))


def is_operation(operation: str):
    r"""Verify if the operation is an implemented stream operation.

    Parameters
    ----------
    operation : `str`
        Operation name. See :func:`help` or :func:`list_operations` for a
        list of implemented operations.

    Returns
    -------
    is_operation : `bool`
        Returns `True` if the ``operation`` is part of the implemented
        stream operations, `False` otherwise.

    """
    return operation in _operations


def inject_dynamic_parameters(
    operation: str, parameters: dict, inventory: Inventory = None,
    starttime=None, endtime=None, verb: int = 0
):
    r"""Inject dynamic parameters to the static parameter dictionary
    when needed by the operation.

    Parameters
    ----------
    operation : `str`
        Operation name. See :func:`help` or :func:`list_operations` for a
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

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    parameters : `dict`
        Dictionary with static and dynamic arguments for the operations.

    """
    params = parameters.copy()  # do not touch the static parameters!
    if 'inventory' in _operations[operation]['inject']:
        params['inventory'] = inventory
    if 'starttime' in _operations[operation]['inject']:
        params['starttime'] = to_UTCDateTime(starttime)
    if 'endtime' in _operations[operation]['inject']:
        params['endtime'] = to_UTCDateTime(endtime)
    return params


def apply_operation(
    waveforms, operation: str, parameters: dict,
    dynamic_parameters: dict = None, stdout_prefix: str = '', verb: int = 0
):
    r"""Apply an in-place operation with the provided parameters.

    Parameters
    ----------
    waveforms : :class:`obspy.Stream` or :class:`obspy.Trace`
        Waveforms on which to apply the list of operations.

    operations : `list`
        List of operations. Each item is a tuple ('operation', {parameters}).
        Use :func:`help` to list all valid operations and its documentation.

    parameters : `dict`
        Dictionary with all arguments for the operation. If the ``operation``
        requires dynamic parameters you should provide them using
        ``dynamic_parameters`` or inject them first manually using
        :func:`inject_dynamic_parameters`.

    dynamic_parameters : `dict`
        Dictionary with all dynamic parameters that will be injected into
        ``parameters`` if required by ``operation``.

    stdout_prefix : `str`, optional
        Add a prefix before printing to stdout.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Returns
    -------
    operated_waveforms : :class:`obspy.Stream` or :class:`obspy.Trace`
        Waveforms after applying the operation.

    """
    if not (isinstance(waveforms, Trace) or
            isinstance(waveforms, Stream)):
        error = ('``waveforms`` is not of type '
                 ':class:`obspy.Stream` or :class:`obspy.Trace`')
        raise TypeError(error)

    if not is_operation(operation):
        error = '"{}" is not an operation.'.format(operation)
        raise NotImplementedError(error)

    method = _operations[operation]['method']
    params = inject_dynamic_parameters(
        operation, parameters, **dynamic_parameters
    ) if dynamic_parameters else parameters

    if verb > 0:
        print('{}{} :'.format(stdout_prefix, operation), params)

    operated_waveforms = (
        eval(f'waveforms.{operation}(**params)') if method == _self
        else method(waveforms, **params)
    )

    if verb > 3:
        print(operated_waveforms)

    return operated_waveforms


def preprocess(
    waveforms, operations: list, inventory: Inventory = None,
    starttime=None, endtime=None, verb: int = 0, **kwargs
):
    r"""Preprocess waveforms given a list of operations.

    Parameters
    ----------
    waveforms : :class:`obspy.Stream` or :class:`obspy.Trace`
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

    **kwargs :
        Additional parameters passed to :func:`apply_operation`.

    Returns
    -------
    operated_waveforms : :class:`obspy.Stream` or :class:`obspy.Trace`
        Waveforms after applying the list of operations.

    """
    if verb > 0:
        print('Apply preprocessing operations:')

    if not (isinstance(waveforms, Trace) or
            isinstance(waveforms, Stream)):
        error = ('``waveforms`` is not of type '
                 ':class:`obspy.Stream` or :class:`obspy.Trace`')
        raise TypeError(error)

    # make sure not to apply the operations in place
    operated_waveforms = waveforms.copy()

    # bag dynamic parameters
    dyn_params = {'inventory': inventory,
                  'starttime': starttime,
                  'endtime': endtime}

    if verb > 1:
        print('Dynamic parameters to be injected:', dyn_params)

    # evaluate all listed operations
    for operation_params in operations:
        if (
            not(isinstance(operation_params, tuple) or
                isinstance(operation_params, list)) or
            len(operation_params) != 2
        ):
            msg = ('Provided operation should be a tuple or list with '
                   'length 2 (method:str,params:dict).')
            warnings.warn(msg, UserWarning)
            continue
        operation, parameters = operation_params
        if not is_operation(operation):
            msg = ('Provided operation "{}" is invalid thus ignored.'
                   .format(operation))
            warnings.warn(msg, UserWarning)
            continue
        try:
            operated_waveforms = apply_operation(
                waveforms=operated_waveforms,
                operation=operation,
                parameters=parameters,
                dynamic_parameters=dyn_params,
                verb=verb,
                stdout_prefix=' * ',
            )
        except Exception as e:
            msg = ('Failed to execute operation "{}". Returned error: {}'
                   .format(operation, e))
            warnings.warn(msg, UserWarning)

    if verb > 2:
        print(operated_waveforms)

    return operated_waveforms


def example_preprocess_dict(to_json: bool = False):
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
