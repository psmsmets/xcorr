r"""

:mod:`util.metadata` -- Metadata Utilities
==========================================

Utilities for ``xcorr`` for metadata conventions.

"""


# Mandatory imports
import xarray as xr
import pandas as pd
import numpy as np
import scipy
import json
import obspy

# Relative imports
from ..version import version


__all__ = ['global_attrs', 'list_versions', 'version']


def global_attrs(kwargs: dict):
    r"""Set the global attribute dictionary for a dataset following
    COARDS/CF-1.9 standards

    Parameters
    ----------

    kwargs : `dict`
        Dictionary with keyword arguments. COARDS/CF-1.9 attributes are
        extracted to set the global metadata attributes.

    Returns
    -------
    attrs : `dict`
        COARDS/CF-1.9 attribute dictionary.

    """
    attrs = {
        'title': kwargs.pop('title', ''),
        'institution': kwargs.pop('institution', 'n/a'),
        'author': kwargs.pop('author', 'n/a'),
        'source': kwargs.pop('source', 'n/a'),
        'history': 'Created @ {}'.format(pd.to_datetime('now')),
        'references': kwargs.pop('references', 'n/a'),
        'comment': kwargs.pop('comment', 'n/a'),
        'Conventions': 'CF-1.9',
        'xcorr_version': version,
        'dependencies_version': list_versions(as_str=True),
    }

    return attrs


def list_versions(as_str: bool = True):
    """Returns the core dependencies and their version as a `str` (default)
    or as a dictionary.
    """
    versions = {
        json.__name__: json.__version__,
        np.__name__: np.__version__,
        obspy.__name__: obspy.__version__,
        pd.__name__: pd.__version__,
        scipy.__name__: scipy.__version__,
        xr.__name__: xr.__version__,
    }
    if as_str:
        return ', '.join(['-'.join(item) for item in versions.items()])
    else:
        return versions
