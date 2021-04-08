r"""

:mod:`core.postprocess` -- xcorr postprocess
============================================

Predefined cc postproccessing

"""


# Mandatory imports
import xarray as xr


__all__ = ['postprocess']


def postprocess(
    ds: xr.Dataset, cmin: float = None, cmax: float = None,
    filter_kwargs: dict = None
):
    """Postprocess an xcorr CCF dataset

    Parameters
    ----------
    ds : :class:`xr.Dataset`
        Xcorr cc dataset.

    cmin : `float`, optional
        Siginal window minimal celerity, in m/s. Defaults to 1460 m/s.

    cmax : `float`, optional
        Signam window maximal celerity, in m/s. Defaults to 1500 m/s.

    filter_kwargs : `dict`, optional
        Dictionary of keyword arguments to pass to the filter.
        Defaults to a 2nd order highpass filter with a 3Hz corner frequency.

    Returns
    -------
    ds2 : `xr.Dataset`
        Dataset with ``ds.cc`` extracted for the valid signal window,
        postprocessed and replaced.

    """
    # set celerities
    cmin = cmin or 1460
    cmax = cmax or 1500
    if ds.distance.units == 'km':
        cmin /= 1000
        cmax /= 1000

    # set filter arguments
    filter_kwargs = {
        'frequency': 3.,
        'btype': 'highpass',
        'order': 2,
        **(filter_kwargs or dict()),
    }

    # extract time_offset and pair_offset
    delay = -(ds.pair_offset + ds.time_offset)

    # extract and postprocess cc
    cc = (
        ds.cc.where(
            (ds.status == 1) &
            (ds.lag >= ds.distance.min()/cmax) &
            (ds.lag <= ds.distance.max()/cmin),
            drop=True
        )
        .signal.unbias()
        .signal.demean()
        .signal.taper(max_length=5.)  # timeshift phase wrapping
        .signal.timeshift(delay=delay, dim='lag', fast=True)
        .signal.filter(**filter_kwargs)
        .signal.taper(max_length=3/2)  # filter artefacts
    )

    # replace raw with processed cc
    ds = ds.drop_vars(('cc', 'lag'))
    ds['cc'] = cc

    return ds
