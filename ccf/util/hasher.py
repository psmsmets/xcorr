r"""

:mod:`util.hasher` -- Hash Utilities
====================================

Utilities for ``ccf`` to generate hashes, preferrably sha256.

"""


# Mandatory imports
import warnings
import xarray as xr
import numpy as np
import json
import hashlib


__all__ = ['hash_it', 'hash_obj', 'hash_DataArray', 'hash_Dataset']

_ignore_keys = ['sha256_hash', 'sha256_hash_metadata',
                'add_offset', 'scale_factor']
_enc = 'utf-8'


def _filter_obj(obj):
    r"""Exclude some keys from a dictionary.
    """
    keys = [key for key in obj.keys() if key not in _ignore_keys]
    return { key: obj[key] for key in keys }


def _to_json(obj):
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
    """Used by :func:`_to_json` to serialize non-standard dtypes.
    """
    if (
        isinstance(obj,np.int8) or
        isinstance(obj,np.int16) or
        isinstance(obj,np.int32) or
        isinstance(obj,np.int64)
    ):
        return int(obj)
    elif (
        isinstance(obj,np.float32) or
        isinstance(obj,np.float64)
    ):
        return float(obj)
    else:
        return repr(obj)


def hash_it(it, **kwargs):
    r"""Hash ``it`` using serialization.

    Parameters
    ----------
    it : various
        Variable to compute or update the hash for.

    hashlib_obj : :class:`hashlib`, optional
        Hashlib algorithm class. If `None` (default),
        :class:`hashlib.sha256()` is used for hashing and the hexdigested
        hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``it`` (default). If a ``hashlib_obj``
        is a provided ``None`` is returned.
 
    """
    if isinstance(it, xarray.DataArray):
        return hash_DataArray(it, **kwargs)
    elif isinstance(it, xarray.Dataset):
        return hash_Dataset(it, **kwargs)
    else:
        return hash_obj(it, **kwargs)


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

    hashlib_obj : :class:`hashlib`, optional
        Hashlib algorithm class. If `None` (default),
        :class:`hashlib.sha256()` is used for hashing and the hexdigested
        hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str` or `None`
        Hexdigested hash of ``it`` (default). If a ``hashlib_obj``
        is a provided `None` is returned.
 
    """
    h = hashlib_obj or hashlib.sha256()
    h.update(_to_json(obj).encode(_enc))
    if debug:
        print('Obj {} hash'.format(type(obj)), h.hexdigest())
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

    hashlib_obj : :class:`hashlib`, optional
        Hashlib algorithm class. If `None` (default),
        :class:`hashlib.sha256()` is used for hashing and the hexdigested
        hash is returned.

    debug : `bool`, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : `str`
        Hexdigested sha256 hash of ``obj``.
 
    """
    h = hashlib_obj or hashlib.sha256()
    if not metadata_only:
        h.update(_to_json(dataset.attrs).encode(_enc))
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
        can take some time!). For multidimensional arrays only available
        metrics are hashed (min(), max(), sum(), cumsum(dim[-1]).sum(),
        diff(dim[-1]).sum(), std(), and median()).

    hashlib_obj : :class:`hashlib`, optional
        Hashlib algorithm class. If `None` (default),
        :class:`hashlib.sha256()` is used for hashing and the hexdigested
        hash is returned.

    debug : bool, optional
        Defaults `False`. When `True` the updated hexdigested hash
        of ``obj`` is printed.

    Returns
    -------
    hash : str
        Hexdigested sha256 hash of ``obj``.
 
    """
    h = hashlib_obj or hashlib.sha256()
    h.update(darray.name.encode(_enc))
    h.update(repr(darray.dims).encode(_enc))
    h.update(_to_json(darray.attrs).encode(_enc))
    if not metadata_only:
        if len(darray.dims) == 1:
            for val in darray.values:
                h.update((
                    val.data.tobytes() if getattr(val, "data", False)
                    else val.encode(_enc)
                ))
        else:
            try:
                h.update(darray.min().data.tobytes())
                h.update(darray.max().data.tobytes())
                h.update(darray.sum().data.tobytes())
                h.update(
                    darray.cumsum(dim=darray.dims[-1]).sum().data.tobytes()
                )
                h.update(
                    darray.diff(dim=darray.dims[-1]).sum().data.tobytes()
                )
                if 'float' in repr(darray.dtype):
                    h.update(darray.std().data.tobytes())
                    h.update(darray.median().data.tobytes())
            except NotImplementedError:
                if debug:
                    print('Fallback')
                h.update((
                    darray.data.tobytes() if getattr(darray, "data", False)
                    else darray.encode(_enc)
                ))
    if debug:
        print(darray.name, h.hexdigest())
    return None if hashlib_obj else h.hexdigest()
