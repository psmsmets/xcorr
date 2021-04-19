r"""

:mod:`io.merge` -- Merge
========================

Merge xcorr N-D labelled set of data arrays.

"""

# Mandatory imports
import xarray as xr
import pandas as pd

# Relative imports
from ..io.validate import validate_list


__all__ = ['merge']


def merge(
    *datasets, extract: bool = False, verb: int = 0, **kwargs
):
    """
    Merge a list of xcorr N-D labelled set of data arrays.

    Parameters
    ----------
    *datasets : :class:`xarray.Dataset` or `str`
        Either a `str` specifying the path (can be a glob string) or a
        :class:`xarray.Dataset` containing the `xcorr` N-D labelled data array.

    extract : `bool`, optional
        Mask cross-correlation estimates with ``status != 1`` with `Nan` if
        `True`. Defaults to `False`.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity. Defaults to 0.

    Any additional keyword arguments will be passed to :func:`validate_list`.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        The merged N-D labelled set of data arrays.

    """
    # check
    dsets = validate_list(datasets, verb=verb, keep_opened=True,
                          parallel=False, **kwargs)

    # merge
    ds = xr.combine_by_coords(
        dsets,
        data_vars='minimal',
        join='outer',
        combine_attrs='override',
    )

    # update global attrs
    ds.attrs = dsets[0].attrs
    if 'sha256_hash' in ds.attrs:
        ds.attrs['sha256_hash']

    strt0 = pd.to_datetime(ds.time.values[0]).strftime('%Y.%j')
    strt1 = pd.to_datetime(ds.time.values[-1]).strftime('%Y.%j')
    ds.attrs['title'] = (
        (ds.attrs['title']).split(' - ')[0] +
        ' - ' +
        strt0 +
        ' to {}'.format(strt1) if strt0 != strt1 else ''
    ).strip()
    ds.attrs['history'] = 'Merged @ {}'.format(pd.to_datetime('now'))

    # extract valid data
    if extract and 'cc' in ds.data_vars:
        ds['cc'] = ds.cc.where(ds.status == 1)

    # cleanup
    for dset in dsets:
        dset.close()
    del dsets

    return ds
