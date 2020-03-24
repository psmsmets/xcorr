r"""

:mod:`util.hasher` -- Hash Utilities
====================================

Utilities for ``ccf`` to generate sha256 hashes.

"""


# Mandatory imports
import warnings
import xarray as xr
import numpy as np
import json
import hashlib


__all__ = ['sha256_hash_obj', 'sha256_hash_DataArray', 'sha256_hash_Dataset',
           'sha256_hash_Dataset_metadata']

_ignore_keys = ['sha256_hash', 'sha256_hash_metadata',
                'add_offset', 'scale_factor']
_enc = 'utf-8'


def _filter_obj(obj):
    r"""
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


def sha256_hash_obj(
    obj 
):
    r"""Generate the sha256 hash on :func:`json.dumps` of ``obje`` to solve
    formatting and sorting issues.

    Parameters
    ----------
    obj : str, list, tuple or dict
        Input variable to compute the sha256 hash.

    Returns
    -------
    hash : str
        Hexdigested sha256 hash of ``obj``.
 
    """
    h = hashlib.sha256()
    h.update(_to_json(obj).encode(_enc))
    return h.hexdigest()


def sha256_hash_Dataset_metadata(
    dataset: xr.Dataset, debug: bool = False
):
    r"""
    """
    h = hashlib.sha256()
    for coord in dataset.coords:
        h.update(coord.encode(_enc))
        h.update(_to_json(dataset[coord].attrs).encode(_enc))
        if debug:
            print(coord,h.hexdigest())
    for var in dataset:
        h.update(var.encode(_enc))
        h.update(_to_json(dataset[var].attrs).encode(_enc))
        if debug:
            print(var,h.hexdigest())
    return h.hexdigest() 


def sha256_hash_Dataset(
    dataset: xr.Dataset, debug: bool = False
):
    r"""
    """
    h = hashlib.sha256()
    h.update(_to_json(dataset.attrs).encode(_enc))
    for coord in dataset.coords:
        sha256_hash_DataArray(
            dataset[coord],
            update_hashlib_obj=h,
            debug=debug
        )
    for var in dataset:
        sha256_hash_DataArray(
            dataset[var],
            update_hashlib_obj=h,
            debug=debug
        )
    return h.hexdigest()


def sha256_hash_DataArray(
    darray: xr.DataArray, update_hashlib_obj=None, debug: bool = False
):
    r"""
    """
    h = update_hashlib_obj or hashlib.sha256()
    h.update(darray.name.encode(_enc))
    h.update(repr(darray.dims).encode(_enc))
    h.update(_to_json(darray.attrs).encode(_enc))
    if len(darray.dims) == 1:
        for val in darray.values:
            h.update(val.data.tobytes() if getattr(val, "data", False) else val.encode(_enc))
    else:
        try:
            h.update(darray.sum().data.tobytes())
            h.update(darray.cumsum().sum().data.tobytes())
            h.update(darray.min().data.tobytes())
            h.update(darray.max().data.tobytes())
            if 'float' in repr(darray.dtype):
                h.update(darray.std().data.tobytes())
                h.update(darray.median().data.tobytes())
        except NotImplementedError:
            if debug:
                print('Fallback')
            h.update(darray.data.tobytes() if getattr(darray, "data", False) else darray.encode(_enc))
    if debug:
        print(darray.name,h.hexdigest())
    return None if update_hashlib_obj else h.hexdigest()
