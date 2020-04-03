r"""

:mod:`util.hasher` -- Hash Utilities
====================================

Utilities for ``xcorr`` to generate hashes, preferrably sha256.

"""


# Mandatory imports
from obspy import Stream, Trace
import xarray as xr
import numpy as np
import json
import hashlib


__all__ = ['hash', 'hash_obj', 'hash_Trace', 'hash_Stream',
           'hash_DataArray', 'hash_Dataset']

_ignore_keys = ['sha256_hash', 'sha256_hash_metadata',
                'add_offset', 'scale_factor']

_trace_keys = ['network', 'station', 'location', 'channel', 'starttime',
               'endtime', 'sampling_rate', 'delta', 'npts']

_enc = 'utf-8'


def _filter_obj(obj):
    r"""Exclude some keys from a dictionary.
    """
    keys = [key for key in obj.keys() if key not in _ignore_keys]
    return {key: obj[key] for key in keys}


def to_json(obj):
    r"""Wrapper for :func:`json.dumps`.
    """
    return json.dumps(
        _filter_obj(obj),
        separators=(',', ':'),
        sort_keys=True,
        indent=4,
        skipkeys=False,
        default=_to_serializable
    )


def _to_serializable(obj):
    """Used by :func:`to_json` to serialize non-standard dtypes.
    """
    if (
        isinstance(obj, np.int8) or
        isinstance(obj, np.int16) or
        isinstance(obj, np.int32) or
        isinstance(obj, np.int64)
    ):
        return int(obj)
    elif (
        isinstance(obj, np.float32) or
        isinstance(obj, np.float64)
    ):
        return float(obj)
    else:
        return repr(obj)


def hash(var, **kwargs):
    r"""Hash ``var`` using serialization.

    Parameters
    ----------
    var : various
        Variable to compute or update the hash for.

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``it`` (default). If a ``hashlib_obj``
        is a provided ``None`` is returned.

    """
    if isinstance(var, xr.DataArray):
        return hash_DataArray(var, **kwargs)
    elif isinstance(var, xr.Dataset):
        return hash_Dataset(var, **kwargs)
    elif isinstance(var, Stream):
        return hash_Stream(var, **kwargs)
    elif isinstance(var, Trace):
        return hash_Trace(var, **kwargs)
    else:
        return hash_obj(var, **kwargs)


def hash_obj(
    obj, hashlib_obj=None, debug: bool = False
):
    r"""Hash a variable.

    Hash variable ``obj`` using :func:`json.dumps` to solve
    formatting, string representation and key order issues.

    Parameters
    ----------
    obj : `str`, `list`, `tuple` or `dict`
        Variable to compute or update the hash for.

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``it`` (default). If a ``hashlib_obj``
        is provided `None` is returned.

    """
    h = hashlib_obj or hashlib.sha256()
    h.update(to_json(obj).encode(_enc))
    if debug:
        print('Obj {} hash'.format(type(obj)), h.hexdigest())
    return None if hashlib_obj else h.hexdigest()


def hash_Trace(
    trace: Trace, hashlib_obj=None, debug: bool = False
):
    r"""Hash a :class:`obspy.Trace`.

    The hash is calculated on the following `obspy.Trace.stats` keys:
    ['network', 'station', 'location', 'channel', 'starttime', 'endtime',
     'sampling_rate', 'delta', 'npts'], sorted and dumped to json with
    4 character space indentation and separators ',' and ':', followed by
    the hash of each sample byte representation.

    Parameters
    ----------
    trace : :class:`obspy.Trace`
        Trace to compute or update the hash for.

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``trace`` (default). If a ``hashlib_obj``
        is provided `None` is returned.

    """
    h = hashlib_obj or hashlib.sha256()
    stats = dict(zip(_trace_keys, [trace.stats[key] for key in _trace_keys]))
    h.update(to_json(stats).encode(_enc))
    for d in trace.data:
        h.update(d.tobytes())
    if debug:
        print('Trace {} hash'.format(trace.id), h.hexdigest())
    return None if hashlib_obj else h.hexdigest()


def hash_Stream(
    stream: Stream, hashlib_obj=None, debug: bool = False
):
    r"""Hash a :class:`obspy.Stream`.

    Parameters
    ----------
    stream : :class:`obspy.Stream`
        Stream to compute or update the hash for.

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``stream`` (default). If a ``hashlib_obj``
        is provided `None` is returned.

    """
    h = hashlib_obj or hashlib.sha256()
    for trace in stream:
        hash_Trace(trace, hashlib_obj=h, debug=debug)
    if debug:
        print('Stream hash', h.hexdigest())
    return None if hashlib_obj else h.hexdigest()


def hash_Dataset(
    dataset: xr.Dataset, metadata_only: bool = True,
    hashlib_obj=None, debug: bool = False
):
    r"""Hash a :class:`xarray.Dataset`.

    Parameters
    ----------
    dataset : :class:`xarray.Dataset`
        Variable to compute or update the hash for.

    metadata_only : `bool`, optional
        If `True` (default), only hash the variable name, dimensions and
        attributes are hashed. When `False` the data is also hashed (which
        can take some time!). For multidimensional arrays only available
        metrics are hashed (min(), max(), sum(), cumsum(dim[-1]).sum(),
        diff(dim[-1]).sum(), std(), and median()).

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``stream`` (default). If a ``hashlib_obj``
        is provided `None` is returned.

    """
    h = hashlib_obj or hashlib.sha256()
    if not metadata_only:
        h.update(to_json(dataset.attrs).encode(_enc))
    for coord in dataset.coords:
        hash_DataArray(
            dataset[coord],
            hashlib_obj=h,
            metadata_only=metadata_only,
            debug=debug
        )
    for var in dataset:
        hash_DataArray(
            dataset[var],
            hashlib_obj=h,
            metadata_only=metadata_only,
            debug=debug
        )
    if debug:
        print('Dataset hash', h.hexdigest())
    return None if hashlib_obj else h.hexdigest()


def hash_DataArray(
    darray: xr.DataArray, metadata_only: bool = True,
    hashlib_obj=None, debug: bool = False
):
    r"""Hash a :class:`xarray.DataArray`.

    Parameters
    ----------
    darray : :class:`xarray.DataArray`
        Variable to compute or update the hash for.

    metadata_only : bool, optional
        If `True` (default), only hash the variable name, dimensions and
        attributes are hashed. When `False` the data is also hashed (which
        can take some time!).

    hashlib_obj : :class:`hashlib._hashlib.HASH`, optional
        Hashlib algorithm class. If `None` (default),
        :func:`hashlib.sha256()` is used for hashing and the
        hexdigested hash is returned.

    debug : bool, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``stream`` (default). If a ``hashlib_obj``
        is provided `None` is returned.

    """
    h = hashlib_obj or hashlib.sha256()
    h.update(darray.name.encode(_enc))
    h.update(repr(darray.dims).encode(_enc))
    h.update(to_json(darray.attrs).encode(_enc))
    if not metadata_only:
        if darray.dtype == np.dtype(object):
            for d in np.nditer(darray, flags=['refs_ok']):
                h.update(str(d).encode(_enc))
        else:
            for d in np.nditer(darray, flags=['refs_ok']):
                h.update(d.tobytes())
    if debug:
        print(darray.name, h.hexdigest())
    return None if hashlib_obj else h.hexdigest()
