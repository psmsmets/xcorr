r"""

:mod:`signal.peak_local_max` -- Peak local max
==============================================

Finding local maxima of an N-D labelled array of data.

"""


# Mandatory imports
import xarray as xr
import numpy as np
from skimage import feature
try:
    import dask
except ModuleNotFoundError:
    dask = False


# Relative imports
from ..util.history import historicize


__all__ = ['peak_local_max']


def peak_local_max(
    x: xr.DataArray, dims: tuple = None, as_index: bool = True,
    as_dataframe: bool = False, extend: bool = False, **kwargs
):
    """
    Finding local maxima of an N-D labelled array of data.

    The maxima are searched along the last two dims by default.

    Parameters
    ----------
    x : :class:`xarray.DataArray`
        The array of data to extract the local maxima.

    dims : `tuple`, optional
        A tuple pair with the coordinates names. Defaults to the last two
        dimensions of ``x``.

    as_index : `bool`, optional
        If `True` (default) return the peak local maxima index (descending)
        with 0 corresponding to the absolute maximum.
        If `False` the

    as_dataframe : `bool`, optional
        Returns ``y`` as a :class:`pandas.DataFrame`. Defaults to `False`.

    extend : `bool`, optional
        Return all parameters such as the peak local maxima, the index and the
        relative threshold. Defaults to `False`.

    **kwargs :
        Any additional keyword arguments will be passed to
        :func:`skimage.feature.peak_local_max`.

    Returns
    -------
    y : various
        Defaults to a :class:`xr.DataArray` with the local maxima of ``x``
        either as the descending integer index or as the true value.
        In case of ``extend`` both are returned as a :class:`xr.Dataset`,
        including the relative threshold.
        When ``as_dataframe`` is `True` the return type is a
        :class:`pd.DataFrame`.

    """
    # dim
    dims = dims or x.dims[-2:]
    if not isinstance(dims, tuple) or len(dims) != 2:
        raise TypeError('dims should be a tuple of length 2')

    for d in dims:
        if d not in x.dims:
            raise ValueError(f'x has no dimensions "{dims}"')

    # defaults
    as_index = True if extend else as_index

    # unfunc wrapper
    def _peak_local_max(x, **kwargs):
        c = feature.peak_local_max(x, **kwargs)
        m = -np.ones(x.shape, dtype=np.int64)
        m[c[:, 0], c[:, 1]] = np.arange(len(c))
        return m

    # dask collection?
    dargs = {}
    if dask and dask.is_dask_collection(x):
        dargs = dict(dask='allowed', output_dtypes=[np.int64])

    # apply ufunc (and optional dask distributed)
    y = xr.apply_ufunc(_peak_local_max, x,
                       input_core_dims=[dims],
                       output_core_dims=[dims],
                       keep_attrs=True,
                       vectorize=True,
                       **dargs,
                       kwargs=kwargs)

    # mask
    y = y.where(y > -1) if as_index else x.where(y >= 0)

    # attributes
    y.attrs = {**x.attrs, **dict(**kwargs)}
    y.name = f"plmax"
    y.attrs["long_name"] = f"Local Maximum {x.long_name}"
    y.attrs["standard_name"] = f"peak_local_max_{x.standard_name}"
    y.attrs["units"] = "-" if as_index else x.attrs["units"]

    # log workflow
    historicize(y, f='peak_local_max', a={
        'x': x.name,
        'dims': dims,
        '**kwargs': kwargs,
    })

    if extend:
        ds = xr.Dataset()
        ds[y.name] = x.where(y >= 0)
        ds[y.name].attrs["units"] = x.attrs["units"]

        ds[f"{y.name}_index"] = y
        ds[f"{y.name}_index"].attrs["long_name"] = f"Local Maximum {x.long_name} Index"
        ds[f"{y.name}_index"].attrs["standard_name"] = f"peak_local_max_{x.standard_name}_index"
        ds[f"{y.name}_index"].attrs["units"] = "-"

        ds[f"{y.name}_threshold"] = ds[y.name] / ds[y.name].max(dims, skipna=True)
        ds[f"{y.name}_threshold"].attrs["long_name"] = f"Local Maximum {x.long_name} Threshold"
        ds[f"{y.name}_threshold"].attrs["standard_name"] = f"peak_local_max_{x.standard_name}_threshold"
        ds[f"{y.name}_threshold"].attrs["units"] = "-"

        y = ds

    return y.to_dataframe().dropna() if as_dataframe else y
