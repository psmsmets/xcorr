r"""

:mod:`core.accessors` -- Accessors
==================================

Xarray Dataset accessors

"""


# Mandatory imports
import warnings
import xarray as xr
from functools import wraps


# Relative imports
from .merge import merge
from .plot import plot_ccf, plot_ccfs, plot_ccfs_colored
from .postprocess import postprocess
from .process import process
from ..io import write
from .. import util, version


__all__ = []


def register_xcorr_dataset_accessor():
    """Register the xcorr Dataset accessor
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr.register_dataset_accessor('xcorr')(XcorrAccessor)


@xr.register_dataset_accessor('xcorr')
class XcorrAccessor():
    """Dataset xcorr accessor
    """
    def __init__(self, obj: xr.Dataset):
        self._obj = obj

    @property
    def version(self):
        """Return the current xcorr version
        """
        return version.version

    @property
    def dependencies_version(self):
        """Return current the dependencies version
        """
        return util.metadata.list_versions(as_str=True)

    @wraps(plot_ccf)
    def plot_ccf(self, *args, **kwargs):
        """
        """
        return plot_ccf(self._obj.cc, self._obj.distance, *args, **kwargs)

    @wraps(plot_ccfs)
    def plot_ccfs(self, *args, **kwargs):
        """
        """
        return plot_ccfs(self._obj.cc, self._obj.distance, *args, **kwargs)

    @wraps(plot_ccfs_colored)
    def plot_ccfs_colored(self, *args, **kwargs):
        """
        """
        return plot_ccfs_colored(self._obj.cc, *args, **kwargs)

    @wraps(postprocess)
    def postprocess(self, *args, **kwargs):
        """
        """
        return postprocess(self._obj, *args, **kwargs)

    @wraps(process)
    def process(self, *args, **kwargs):
        """
        """
        return process(self._obj, *args, **kwargs)

    @wraps(merge)
    def merge(self, *args, **kwargs):
        """
        """
        return merge(self._obj, *args, **kwargs)

    @wraps(write)
    def write(self, *args, **kwargs):
        """
        """
        return write(self._obj, *args, **kwargs)

    def hash(self, debug: bool = False):
        """
        Hash a :class:`xarray.Dataset`

        Both the metadata (name, dimensions and attributes) and the data are
        hashed (which can take some time!). For multidimensional arrays only
        available metrics are hashed (min(), max(), sum(),
        cumsum(dim[-1]).sum(), diff(dim[-1]).sum(), std(), and median()).

        Parameters
        ----------
        debug : `bool`, optional
            Defaults `False`. When `True` the updated hexdigested hash of
            ``obj`` is printed.

        Returns
        -------
        hash : `str`
            Hexdigested hash of ``dataset``.
        """
        return util.hasher.hash_Dataset(self._obj, metadata_only=False,
                                        debug=debug)

    def hash_metadata(self, debug: bool = False):
        """
        Hash a :class:`xarray.Dataset`

        Only hash the variable name, dimensions and attributes are hashed.

        Parameters
        ----------
        debug : `bool`, optional
            Defaults `False`. When `True` the updated hexdigested hash of
            ``obj`` is printed.

        Returns
        -------
        hash : `str`
            Hexdigested hash of ``dataset``.
        """
        return util.hasher.hash_Dataset(self._obj, metadata_only=True,
                                        debug=debug)
