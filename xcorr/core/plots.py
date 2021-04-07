r"""

:mod:`io.plot` -- xcorr plot
============================

Some predefined plotting routines

"""


# Mandatory imports
import numpy as np
import xarray as xr
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator


__all__ = []


# set fontsize
plt.rcParams.update({'font.size': 9})


def postprocess_ccf(
    ds: xr.Dataset, cmin: float = None, cmax: float = None
):
    """Postprocess an xcorr CCF dataset
    """
    # set celerities
    cmin = cmin or 1460
    cmax = cmax or 1500
    if ds.distance.units == 'km':
        cmin /= 1000
        cmax /= 1000

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
        .signal.filter(frequency=3., btype='highpass', order=2)
        .signal.taper(max_length=3/2)  # filter artefacts
    )
    ds = ds.drop_vars(('cc', 'lag'))
    ds['cc'] = cc
    return ds


def plot_ccf(
    cc: xr.DataArray, distance=None, time: int = 0,
    db: bool = True, contour: bool = True,
    cmin: float = None, cmax: float = None, cmajor: float = None,
    cminor: float = None, lag_lim: tuple = None,
    freq_lim: tuple = None, spectrogram_kwargs: dict = None,
    cc_plot_kwargs: dict = None, spectrogram_plot_kwargs: dict = None,
    cbar_kwargs: dict = None,
):
    """Plot a CCF DataArray with spectrogram.
    """

    # check CCF
    if not isinstance(cc, xr.DataArray):
        raise TypeError("cc should be a :class:`xarray.DataArray`")
    if sorted(cc.dims) != ['lag', 'time']:
        raise ValueError("cc can only have dimensions 'time' and 'lag'")

    ccf_lim = (-1.05, 1.05)
    freq_lim = freq_lim or ()
    factor = cc.signal.abs().max()

    cmin = cmin or 1460
    cmax = cmax or 1500
    if isinstance(distance, xr.DataArray):
        distance.load()  # no Dask arrays
        d = distance.values.item()*(1000 if distance.units == 'km' else 1)
        c_major = np.arange(cmin, cmax, cmajor or (cmax-cmin)/5)
        c_minor = np.arange(cmin, cmax, cminor or (cmax-cmin)/25)
        lag_lim = lag_lim or (d/cmax, d/cmin)
    else:
        d = None
        lag_lim = lag_lim or (cc.lag.min().item(), cc.lag.max().item())

    fig = plt.figure(constrained_layout=True, figsize=(7, 4))

    gs = GridSpec(2, 2, figure=fig, width_ratios=(40, 1))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[1, 1])

    # CCF
    cc_plot_kwargs = {
        'x': 'lag',
        'ax': ax1,
        'color': 'k',
        'alpha': .25,
        'add_legend': False,
        **(cc_plot_kwargs or dict()),
    }
    (cc/factor).plot.line(**cc_plot_kwargs)
    ax1.set_title(None)
    ax1.set_xlabel(None)
    ax1.set_ylim(*ccf_lim)
    ax1.set_ylabel('CCF [-]')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xlim(*lag_lim)

    if d:
        ax1_t = ax1.secondary_xaxis('top')
        ax1_t.set_xticks(d/c_major)
        ax1_t.set_xticklabels([str(c) + ' m/s' for c in c_major],
                              fontsize=7, fontstyle='italic')
        ax1_t.set_xticks(d/c_minor, minor=True)
        ax1_t.set_xlabel(' ')

    ax1.text(0.02, 0.96, str(cc.pair.values),
             transform=ax1.transAxes, ha='left', va='top')
    ax1.text(0.98, 0.96, str(cc.time[0].values)[:19],
             transform=ax1.transAxes, ha='right', va='top')

    # compute spectrogram and normalize
    spectrogram_kwargs = {
        'duration': 2.5,
        'padding_factor': 4,
        **(spectrogram_kwargs or dict())
    }
    p = (cc.isel(time=time)).signal.spectrogram(**spectrogram_kwargs)
    p = p/p.max()

    # plot spectrogram
    spectrogram_plot_kwargs = {
        'cmap': 'afmhot_r',
        'vmin': -70 if db else 0,
        'vmax': -10 if db else .5,
        'ax': ax2,
        **(spectrogram_plot_kwargs or dict()),
        'add_colorbar': False,
    }
    p = (xr.ufuncs.log10(p.where(p > 0)) * 20) if db else p
    p = (p.plot.contourf(**spectrogram_plot_kwargs) if contour
         else p.plot.imshow(**spectrogram_plot_kwargs))

    ax2.set_title(None)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_ticks_position('both')
    ax2.tick_params(labelbottom=True)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_ticks_position('both')
    ax2.set_xlim(*lag_lim)
    ax2.set_ylim(*freq_lim)

    # colorbar
    cbar_kwargs = {
        'cax': ax3,
        'use_gridspec': True,
        'extend': 'both',
        **(cbar_kwargs or dict()),
    }
    if contour:
        cbar_kwargs.pop('extend')
    cb = plt.colorbar(p, **cbar_kwargs)
    cb.set_label(r'PSD [dB]' if db else r'PSD [-]')

    return gs


def plot_ccfs(
    cc: xr.DataArray, distance: xr.DataArray = None,
    pairs: xr.DataArray = None, cmin: float = None, cmax: float = None,
    cmajor: float = None, cminor: float = None, lag_lim: tuple = None,
    cc_plot_kwargs: dict = None,
):
    """
    """

    ccf_lim = (-1.05, 1.05)
    factor = cc.signal.abs().max()

    pairs = pairs or cc.pair

    cmin = cmin or 1460
    cmax = cmax or 1500
    if isinstance(distance, xr.DataArray):
        distance.load()  # no Dask arrays
        d_factor = 1000 if distance.units == 'km' else 1
        c_major = np.arange(cmin, cmax, cmajor or (cmax-cmin)/5)
        c_minor = np.arange(cmin, cmax, cminor or (cmax-cmin)/25)
        lag_lim = lag_lim or (distance.min().item()*d_factor/cmax,
                              distance.max().item()*d_factor/cmin)
    else:
        lag_lim = lag_lim or (cc.lag.min().item(), cc.lag.max().item())

    fig = plt.figure(constrained_layout=True, figsize=(7, len(pairs) + 1))
    gs = GridSpec(pairs.size, 1, figure=fig)

    cc_plot_kwargs = {
        'color': 'k',
        'alpha': .25,
        'add_legend': False,
        'x': 'lag',
        **(cc_plot_kwargs or dict()),
    }

    axis = []
    for i, pair in enumerate(pairs):
        ax = fig.add_subplot(gs[i, 0])
        (cc.sel(pair=pair)/factor).plot.line(**cc_plot_kwargs, ax=ax)
        ax.set_title(None)
        if i != len(pairs)-1:
            ax.set_xlabel(None)
            ax.tick_params(labelbottom=False)

        if isinstance(distance, xr.DataArray):
            d = distance.sel(pair=pair).item()*d_factor
            ax_t = ax.secondary_xaxis('top')
            ax_t.set_xticks(d/c_major)
            ax_t.set_xticks(d/c_minor, minor=True)
            ax_t.set_xticklabels([str(c) + ' m/s' for c in c_major],
                                 fontsize=7, fontstyle='italic')

        ax.set_ylabel('CCF [-]' if i == 0 else None)
        ax.tick_params(labelleft=i == 0)
        ax.set_xlim(*lag_lim)
        ax.set_ylim(*ccf_lim)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_ticks_position('both')
        ax.text(0.02, 0.96, str(pair.values),
                transform=ax.transAxes, ha='left', va='top')
        if i == 0:
            ax.text(0.98, 0.96, str(cc.time[0].values)[:19],
                    transform=ax.transAxes, ha='right', va='top')

        axis.append(ax)

    return gs
