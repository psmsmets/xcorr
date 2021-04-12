r"""

:mod:`core.plot` -- xcorr plot
==============================

Some predefined plotting routines

"""


# Mandatory imports
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

# Relative imports
from ..signal.trigger import plot_trigs


__all__ = ['plot_ccf', 'plot_ccfs', 'plot_snr_ct']


# set fontsize
plt.rcParams.update({'font.size': 9})


def plot_ccf(
    cc: xr.DataArray, distance=None, pair: int = 0, time: int = 0,
    normalize: bool = False, cmin: float = None, cmax: float = None,
    cmajor: float = None, cminor: float = None, lag_lim: tuple = None,
    freq_lim: tuple = None, spectrogram_db: bool = True,
    spectrogram_contourf: bool = True, spectrogram_kwargs: dict = None,
    spectrogram_plot_kwargs: dict = None,
    cc_plot_kwargs: dict = None, cbar_kwargs: dict = None,
    figure: mpl.figure.Figure = None
) -> GridSpec:
    """Plot a single-pair xcorr CCFs and it's spectrogram

    Parameters
    ----------
    cc : :class:`xr.DataArray`
        Cross-correlation data with dims 'lag' and 'time'.
        If depending on 'pair', the dimension is reduced with ``pair``.

    distance : :class:`xr.DataArray`, optional
        The distance used to set the celerity ticks and range. Can be the
        geodetic distance between the cross-correlation pair or the relative
        distance towards a POI.
        If depending on 'pair', the dimension is reduced with ``pair``.

    time : `int`, optional
        Set the time coordinate index. Defaults to `0`.

    pair : `int`, optional
        Set the pair coordinate index. Defaults to `0`.

    normalize : `bool`, optional
        Normalize the CCF and spectogram plots.

    cmin : `float`, optional
        Siginal window minimal celerity, in m/s. Defaults to 1460 m/s.

    cmax : `float`, optional
        Signal window maximal celerity, in m/s. Defaults to 1500 m/s.

    cmajor : `float`, optional
        Celerity major thick spacing, in m/s. Defaults to 5 m/s.

    cminor : `float`, optional
        Celerity minor thick spacing, in m/s. Defaults to 1 m/s.

    lag_lim : `tuple`, optional
        Set the lag lower and upper limit.

    freq_lim : `tuple`, optional
        Set the frequency lower and upper limit.

    spectrogram_db : `bool`, optional
        Plot the spectrogram in dB. Defaults to `True`.

    spectrogram_contourf : `bool`, optional
        Plot the spectrogram using contourf. Defaults to `True`.

    spectrogram_kwargs : `dict`
        Dictionary of keyword arguments to pass to the spectrogram computation
        function.

    spectrogram_plot_kwargs : `dict`
        Dictionary of keyword arguments to pass to the spectrogram plot
        function.

    cc_plot_kwargs : `dict`, optional
        Dictionary of keyword arguments to pass to the cc line plot function.

    cbar_kwargs : `dict`, optional
        Dictionary of keyword arguments to pass to the colorbar.

    figure : :class:`matplotlib.figure.Figure`, optional
        User-defined figure object. If empty (default) a constrained layout
        with figsize (7, 4) is created.

    Returns
    -------
    gs : :class:`matplotlib.gridspec.GridSpec`
        Gridset with the figure and axes.

    """

    # check CCF
    if not isinstance(cc, xr.DataArray):
        raise TypeError("cc should be a :class:`xarray.DataArray`")
    if 'pair' in cc.dims:
        cc = cc.isel(pair=pair)
    if sorted(cc.dims) != ['lag', 'time']:
        raise ValueError("cc can only have dimensions 'time' and 'lag'")

    ccf_max = cc.signal.abs().max()
    ccf_lim = (-1.05, 1.05) if normalize else (-1.05 * ccf_max, 1.05 * ccf_max)
    freq_lim = freq_lim or ()

    cmin = cmin or 1460
    cmax = cmax or 1500
    if isinstance(distance, xr.DataArray):
        distance.load()  # no Dask arrays
        if 'pair' in distance.dims:
            distance = distance.isel(pair=pair)
        d = distance.values.item()*(1000 if distance.units == 'km' else 1)
        c_major = np.arange(cmin, cmax, cmajor or 5)
        c_minor = np.arange(cmin, cmax, cminor or 1)
        lag_lim = lag_lim or (d/cmax, d/cmin)
    else:
        d = None
        lag_lim = lag_lim or (cc.lag.min().item(), cc.lag.max().item())

    fig = figure or plt.figure(constrained_layout=True, figsize=(7, 4))
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
    (cc/ccf_max if normalize else cc).plot.line(**cc_plot_kwargs)
    ax1.set_title(None)
    ax1.set_xlim(*lag_lim)
    ax1.set_xlabel(None)
    ax1.set_ylim(*ccf_lim)
    ax1.ticklabel_format(axis='y', useOffset=False, style='plain')
    ax1.set_ylabel('CCF [-]')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_ticks_position('both')

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
    p = p/p.max() if (normalize or spectrogram_db) else p
    p = (10 * xr.ufuncs.log10(p.where(p > 0))) if spectrogram_db else p

    # plot spectrogram
    spectrogram_plot_kwargs = {
        'cmap': 'afmhot_r',
        'vmin': -36 if spectrogram_db else (0 if normalize else None),
        'vmax': -0 if spectrogram_db else (.5 if normalize else None),
        'ax': ax2,
        **(spectrogram_plot_kwargs or dict()),
        'add_colorbar': False,
    }
    p = (p.plot.contourf(**spectrogram_plot_kwargs) if spectrogram_contourf
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
    if spectrogram_contourf:
        cbar_kwargs.pop('extend')
    cb = plt.colorbar(p, **cbar_kwargs)
    cb.set_label(r'PSD [dB]' if spectrogram_db else r'PSD [-]')

    return gs


def plot_ccfs(
    cc: xr.DataArray, distance: xr.DataArray = None,
    pairs: xr.DataArray = None, cmin: float = None, cmax: float = None,
    cmajor: float = None, cminor: float = None, lag_lim: tuple = None,
    cc_plot_kwargs: dict = None, figure: mpl.figure.Figure = None
) -> GridSpec:
    """Plot multi-pair xcorr CCFs in separate axis

    Parameters
    ----------
    cc : :class:`xr.DataArray`
        Cross-correlation data with dims 'lag' and 'time'.
        If depending on 'pair', the dimension is reduced with ``pair``.

    distance : :class:`xr.DataArray`, optional
        The distance used to set the celerity ticks and range. Can be the
        geodetic distance between the cross-correlation pair or the relative
        distance towards a POI.

    cmin : `float`, optional
        Siginal window minimal celerity, in m/s. Defaults to 1460 m/s.

    cmax : `float`, optional
        Signal window maximal celerity, in m/s. Defaults to 1500 m/s.

    cmajor : `float`, optional
        Celerity major thick spacing, in m/s. Defaults to 5 m/s.

    cminor : `float`, optional
        Celerity minor thick spacing, in m/s. Defaults to 1 m/s.

    lag_lim : `tuple`, optional
        Set the time lag lower and upper limit.

    cc_plot_kwargs : `dict`, optional
        Dictionary of keyword arguments to pass to the cc line plot function.

    figure : :class:`matplotlib.figure.Figure`, optional
        User-defined figure object. If empty (default) a constrained layout
        with figsize (7, len(pairs)+1) is created.

    Returns
    -------
    gs : :class:`matplotlib.gridspec.GridSpec`
        Gridset with the figure and axes.

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

    fig = figure or plt.figure(constrained_layout=True,
                               figsize=(7, len(pairs) + 1))
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

        ax = (fig.add_subplot(gs[i, 0]) if i == 0 else
              fig.add_subplot(gs[i, 0], sharex=axis[0], sharey=axis[0]))

        (cc.sel(pair=pair)/factor).plot.line(**cc_plot_kwargs, ax=ax)
        ax.set_title(None)

        if i != len(pairs)-1:
            ax.set_xlabel(None)
            ax.tick_params(labelbottom=False)
            ax.ticklabel_format(axis='y', useOffset=False, style='plain')

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


def plot_snr_ct(
    snr: xr.DataArray, ct: xr.DataArray = None, ax: mpl.axes.Axes = None,
    *args, **kwargs
):
    """Plot the signal-to-noise ratio and coincidence triggers.

    Parameters
    ----------
    snr : :class:`xr.DataArray`
        Signal-to-noise ratio data.

    ct : :class:`xr.DataArray`, optional
        Coincidence trigger data.

    ax : :class:`mpl.axes.Axes`, optional
        Axis on which to plot this figure. By default, use the current axis.

    *args, **kwargs : optional
        Additional arguments to snr.plot.line
    """
    snr.plot.line(x='time', hue='pair', ax=ax, **kwargs)
    plot_trigs(snr, ct, ax=ax)


def plot_ccfs_colored(
    cc: xr.DataArray, sn: xr.DataArray = None, sn_threshold: float = None,
    alpha: float = None, pairs: xr.DataArray = None, ax: mpl.axes.Axes = None,
    **kwargs
) -> mpl.axes.Axes:
    """Plot multi-pair CCFs color-coded in a single axes.

    Parameters
    ----------
    cc : :class:`xr.DataArray`
        Cross-correlation data with dims 'lag' and 'time'.
        If depending on 'pair', the dimension is reduced with ``pair``.

    sn : :class:`xr.DataArray`, optional
        Signal-to-noise ratio to filter CCFs with the minimal threshold.

    sn_threshold : `float`, optional
        Set the signal-to-noise ratio threshold to filter CCFs. Defaults to 1.

    ax : :class:`mpl.axes.Axes`, optional
        Axis on which to plot this figure. By default, use the current axis.

    **kwargs : optional
        Additional keyword arguments to cc.plot.line

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The plot axes.

    """
    ax = ax or plt.gca()
    alpha = alpha or .25

    # filter CCFs by the signal-to-noise ratio?
    sn_filt = isinstance(sn, xr.DataArray)

    lines = []
    for p, c in zip(cc.pair,  mpl.rcParams['axes.prop_cycle']()):
        sn_pass = sn.loc[{'pair': p}] >= sn_threshold if sn_filt else None
        if sn_filt and not any(sn_pass):
            continue
        for t in cc.time[sn_pass] if sn_filt else cc.time:
            line, = (cc
                     .sel(pair=p, time=t)
                     .plot.line(x='lag', alpha=alpha, ax=ax, **c, **kwargs)
                     )
        lines.append((line, str(p.values)))
    plt.legend(list(zip(*lines))[0], list(zip(*lines))[1])

    if sn_filt:
        ax.set_title(f'{sn.long_name} > {sn_threshold}')

    return ax
