# Mandatory imports
import xarray as xr

# Relative imports
from ..stream.process import operations_to_dict, operations_to_json


__all__ = ['preprocess_operations_to_dict', 'preprocess_operations_to_json']


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
    if attribute in pair.attrs and isinstance(pair.attrs[attribute], str):
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
    if attribute in pair.attrs and isinstance(pair.attrs[attribute], dict):
        pair.attrs[attribute] = operations_to_json(pair.attrs[attribute])
