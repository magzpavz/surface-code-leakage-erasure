from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import ler_ansatz


_AXIS_LABELS = {
    "p": "Physical error rate $p$",
    "d": "Code distance $d$",
    "ec_sched": "Erasure check schedule",
}


def _resolve_sweep(sweep, dat, default_values=None):
    """ Normalize a sweep argument to a (level_name, values) pair.

    `sweep` may be a level name (string), a (level_name, values) tuple, or None.
    """
    if sweep is None:
        return None, default_values if default_values is not None else [None]
    if isinstance(sweep, str):
        return sweep, list(dat.index.get_level_values(sweep).unique())
    level, values = sweep
    return level, list(values)


def _color_map(values):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        return {}
    return {v: default_colors[i % len(default_colors)] for i, v in enumerate(values)}


def _normalize_ls_kwarg(plot_kwargs):
    if "ls" in plot_kwargs:
        plot_kwargs["linestyle"] = plot_kwargs.pop("ls")

## Logical error rate plotting functions

def plot_LER(
    ax : plt.Axes, dat : pd.DataFrame, *,
    ler_name = "ler", vs = "p", label=None,
    errorbands=True, errorbars=False,
    suppress_warnings=False, **plot_kwargs):
    """ Plot LER vs `vs` (e.g. "p" or "d"). `dat` should already be filtered to a single set of other parameters. """

    if vs in (dat.index.names or []):
        x_list = dat.index.get_level_values(vs)
    elif vs in dat.columns:
        x_list = dat[vs]
    else:
        raise ValueError(f"Could not find x-axis values in index or columns with name {vs}")

    ler = dat.filter(regex=ler_name+"$", axis=1).squeeze(axis=1)
    if ler.empty:
        if not suppress_warnings:
            warn(f"No columns found matching regex {ler_name}")
        return

    if errorbands or errorbars:
        ler_lo = dat.filter(regex=ler_name+"_lo", axis=1).squeeze(axis=1)
        ler_hi = dat.filter(regex=ler_name+"_hi", axis=1).squeeze(axis=1)
        if ler_lo.empty or ler_hi.empty:
            if not suppress_warnings:
                warn(f"Could not find confidence interval columns matching regex {ler_name}_lo and {ler_name}_hi. Not plotting confidence intervals.")
            ler_lo = ler_hi = None
    else:
        ler_lo = ler_hi = None

    x_arr = np.asarray(x_list)
    ler_arr = np.asarray(ler, dtype=float)
    ler_lo_arr = None if ler_lo is None else np.asarray(ler_lo, dtype=float)
    ler_hi_arr = None if ler_hi is None else np.asarray(ler_hi, dtype=float)

    if errorbands and ler_lo_arr is not None:
        plot_min = ax.get_ylim()[0]
        ler_lo_limited = ler_lo_arr.copy()
        ler_lo_limited[ler_lo_arr <= 0] = plot_min**2  # avoid plotting error bands that extend to zero or negative values on a log scale
        color_dict = {}
        if 'color' in plot_kwargs:
            color_dict['color'] = plot_kwargs['color']
        ax.fill_between(
            x_arr, ler_lo_limited, ler_hi_arr,
            alpha=0.3, **color_dict)

    if errorbars and ler_lo_arr is not None:
        ax.errorbar(
            x_arr, ler_arr,
            yerr=[ler_arr - ler_lo_arr, ler_hi_arr - ler_arr],
            label=label, **plot_kwargs)
    else:
        plot_y = np.where(ler_arr == 0, np.nan, ler_arr)
        ax.plot(
            x_arr,
            plot_y,
            label=label, **plot_kwargs)


def plot_sweep_LER(
    dat, col_sweep, row_sweep, ls_sweep=None, *,
    vs="p", inner_sweep="",
    fig = None, axs=None,
    ymin=0.8e-6, ymax=1, ylabel="Logical error rate",
    suptitle=None, figsize=(10,8),
    **plot_kwargs):
    """ Plot LER vs `vs` (e.g. "p" or "d"), with subplots arranged according to `col_sweep` and `row_sweep`.

    Arguments:
    dat: DataFrame with a MultiIndex containing the sweep parameters and a column containing the LER values.
    col_sweep: The parameter to sweep across columns. Can be a string (the name of the index level to use) or a tuple of (index level name, list of values to plot).
    row_sweep: The parameter to sweep across rows.
    ls_sweep: The parameter to sweep across line styles.
    vs: The index level used as the x-axis (default "p").
    inner_sweep: The index level used to draw multiple curves per axis (one color each). Defaults to "d" when vs="p" and to "p" when vs="d".
    Returns:
    fig, axs: The figure and axes objects containing the plot.

    """
    if inner_sweep == "":
        inner_sweep = "d" if vs == "p" else "p"

    inner_level, inner_values = _resolve_sweep(inner_sweep, dat)
    inner_colors = _color_map(inner_values)
    col_level, col_values = _resolve_sweep(col_sweep, dat)
    row_level, row_values = _resolve_sweep(row_sweep, dat)

    _normalize_ls_kwarg(plot_kwargs)
    ls_level, ls_values = _resolve_sweep(
        ls_sweep, dat, default_values=[plot_kwargs.get("linestyle", "-")])

    if axs is not None:
        if axs.shape != (len(row_values), len(col_values)):
            raise ValueError(f"Provided axes array has shape {axs.shape}. Expected {(len(row_values), len(col_values))} based on the number of row and column values.")
        fig = axs[0,0].get_figure()
    elif fig is not None:
        try:
            axs = np.array(fig.get_axes()).reshape(len(row_values), len(col_values))
        except ValueError:
            raise ValueError(f"Provided figure has {len(fig.get_axes())} axes. Expected {len(row_values) * len(col_values)} based on the number of row and column values.")
    else:
        fig, axs = plt.subplots(
            len(row_values), len(col_values),
            figsize=figsize, sharex=True, sharey=True,
            squeeze=False,
        )
        if vs == "p":
            axs[0, 0].set_xscale("log")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_ylim(ymin, ymax)
        if vs == "d":
            axs[0, 0].set_xticks(sorted(dat.index.get_level_values("d").unique()))

    for ax_col, col_value in zip(axs.T, col_values):
        ax_col[0].set_title(f"{col_level} = {col_value}")
        for ax, row_value in zip(ax_col, row_values):
            for ls, ls_value in zip(["-", "--", "-.", ":"], ls_values):
                if ls_level is not None:
                    plot_kwargs.update({"linestyle": ls})
                for inner in inner_values:
                    xs_args = [
                        (xs_name, xs_value) for xs_name, xs_value in
                        [(col_level, col_value), (row_level, row_value), (ls_level, ls_value), (inner_level, inner)]
                        if xs_name is not None]
                    xs_names, xs_values = zip(*xs_args)
                    subset = dat.xs(xs_values, level=xs_names)
                    if subset.empty:
                        continue

                    inner_kwargs = dict(plot_kwargs)
                    if "color" not in inner_kwargs and inner in inner_colors:
                        inner_kwargs["color"] = inner_colors[inner]
                    plot_LER(
                        ax,
                        subset,
                        vs=vs,
                        suppress_warnings=True,
                        label=f"{inner}" + (f", {ls_value}" if ls_sweep is not None else ""),
                        **inner_kwargs)

            if vs == "p":
                text_pos = (0.05, 0.85)
            else:
                text_pos = (0.6, 0.9)
            ax.text(
                *text_pos, f"{row_level} = {row_value}",
                transform=ax.transAxes,
                backgroundcolor=("w", 0.5),
            )
            ax.grid(which="major", alpha=0.7)
            ax.grid(which="minor", alpha=0.3)

    xlabel = _AXIS_LABELS.get(vs, vs)
    for ax in axs[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.subplots_adjust(right=0.9, hspace=0.05, wspace=0.05, top=0.9)
    if inner_sweep is not None or ls_sweep is not None:
        handles, labels = axs[0, 0].get_legend_handles_labels()   # might need to edit this in future in case first plot doesn't have all lines
        legend_title = (
            inner_level if inner_sweep is not None else ""
            ) + (", " if inner_sweep is not None and ls_sweep is not None else ""
            ) + (ls_level if ls_sweep is not None else "")
        fig.legend(handles, labels, loc="right", title=legend_title)
    return fig, axs


def plot_sweep_LER_fits(
    fits, col_sweep, row_sweep, ls_sweep=None, *,
    vs="p", p_min=1e-4, p_max=None, p_values=None, d_values=None,
    restrict_to=None, ler_name="ler",
    **kwargs):
    """ Plot fitted `ler_ansatz` curves on a grid of subplots arranged by
    `col_sweep` and `row_sweep`. Builds a synthetic (d, p) DataFrame from the
    fitted parameters and delegates to `plot_sweep_LER`.

    Arguments:
    fits: DataFrame indexed by grouping levels (as returned by `fit_ler_by_group`),
        with columns `log_a`, `alpha`, `log_p_th`.
    col_sweep, row_sweep, ls_sweep: As in `plot_sweep_LER`. Must refer to
        levels in the `fits` index.
    vs: x-axis variable, "p" or "d".
    p_values: Explicit p values to evaluate the fit at. If given, overrides
        `p_min`/`p_max` and the default geomspace grid. Per group, values
        above the fitted `p_th` are dropped.
    p_min, p_max: Bounds for the default p grid (used when `p_values` is None).
        `p_max` defaults to each group's fitted `p_th` (vs="p") or 3e-2 (vs="d").
    d_values: Iterable of code distances to plot one curve per (when vs="p") or
        to evaluate the curve at (when vs="d"). Defaults to [3,5,7,9].
    Remaining kwargs are forwarded to `plot_sweep_LER`.
    """
    finite = fits[np.isfinite(fits[["log_a", "alpha", "log_p_th"]]).all(axis=1)]

    if d_values is None:
        d_values = [3,5,7,9]

    p_values = None if p_values is None else np.asarray(sorted(p_values))

    if vs not in ("p", "d"):
        raise ValueError(f"Unsupported vs={vs!r}; expected 'p' or 'd'")

    def _attach_key(df, key):
        key_tuple = key if isinstance(key, tuple) else (key,)
        for name, value in zip(fits.index.names, key_tuple):
            df[name] = value
        return df

    pieces = []
    for key, params in finite.iterrows():
        group_p_th = float(np.exp(params["log_p_th"]))
        if vs == "p":
            if p_values is not None:
                p_grid = p_values
            else:
                this_p_max = p_max if p_max is not None else group_p_th
                p_grid = np.geomspace(p_min, this_p_max, 3)
            log_p_grid = np.log(p_grid)
            mask_above = p_grid > group_p_th
            for d in d_values:
                log_ler = ler_ansatz(
                    (d, log_p_grid),
                    params["log_a"], params["alpha"], params["log_p_th"])
                ler = np.where(mask_above, np.nan, np.exp(log_ler))
                df = pd.DataFrame({ler_name: ler, "p": p_grid})
                df["d"] = d
                pieces.append(_attach_key(df, key))
        else:  # vs == "d"
            if p_values is not None:
                p_iter = p_values
            else:
                this_p_max = p_max if p_max is not None else 3e-2
                p_iter = np.geomspace(p_min, this_p_max, 7)
            d_grid = np.array(d_values)
            for p in np.asarray(p_iter)[::-1]:  # reverse so higher-p curves draw below lower-p
                log_p = np.log(p)
                log_ler = ler_ansatz(
                    (d_grid, np.full_like(d_grid, log_p, dtype=float)),
                    params["log_a"], params["alpha"], params["log_p_th"])
                ler = np.exp(log_ler)
                if p > group_p_th:
                    ler = np.full_like(ler, np.nan)
                df = pd.DataFrame({ler_name: ler, "d": d_grid})
                df["p"] = p
                pieces.append(_attach_key(df, key))

    synthetic = pd.concat(pieces, ignore_index=True).set_index(
        list(fits.index.names) + ["d", "p"])

    if restrict_to is not None:
        synthetic = synthetic.reorder_levels(restrict_to.index.names)
        synthetic = synthetic.loc[synthetic.index.intersection(restrict_to.index)]

    return plot_sweep_LER(
        synthetic, col_sweep, row_sweep, ls_sweep,
        vs=vs, ler_name=ler_name, **kwargs)


## Fit parameter plotting

def plot_sweep_fit_params(
    fits, col_sweep=None, ls_sweep=None, *,
    vs="ec_sched", inner_sweep=None,
    params=("alpha", "log_p_th"),
    err_suffix="_err", value_suffix="",
    fig=None, figsize=(5, 8), suptitle=None,
    x_offset_range=0.1,
    **plot_kwargs):
    """ Plot fitted parameters (e.g. ``alpha``, ``log_p_th``) vs an index level
    `vs`, with one row of subplots per parameter and an optional column sweep.

    Mirrors the call style of `plot_sweep_LER`: each sweep argument is either
    the name of an index level (string) or a (level_name, values) tuple.

    Arguments:
    fits: DataFrame indexed by the sweep levels, with one column per fitted
        parameter (named ``f"{param}{value_suffix}"``) and matching error columns
        (named ``f"{param}{err_suffix}"``).
    col_sweep: Index level swept across columns. None for a single column.
    ls_sweep: Index level swept across line styles.
    vs: Index level used as the x-axis. Treated as categorical with evenly
        spaced ticks.
    inner_sweep: Index level drawn as multiple curves (one color each) per axis.
    params: Iterable of parameter base-names to plot, one row each.
    err_suffix, value_suffix: Suffixes appended to each param name to find the
        value and error columns in `fits`.
    x_offset_range: Total horizontal spread (in x-tick units) used to dodge
        overlapping curves at each x value.
    Returns:
    fig, axs: The figure and the (n_params, n_cols) axes array.
    """
    _normalize_ls_kwarg(plot_kwargs)

    vs_level, vs_values = _resolve_sweep(vs, fits)
    if vs_level is None:
        raise ValueError("`vs` must name an index level")
    col_level, col_values = _resolve_sweep(col_sweep, fits)
    ls_level, ls_values = _resolve_sweep(
        ls_sweep, fits, default_values=[plot_kwargs.get("linestyle", "-")])
    inner_level, inner_values = _resolve_sweep(inner_sweep, fits)

    inner_colors = _color_map(inner_values)

    params = list(params)
    n_rows, n_cols = len(params), len(col_values)

    n_offsets = max(len(ls_values) * len(inner_values), 1)
    offsets = np.linspace(-x_offset_range, x_offset_range, n_offsets) if n_offsets > 1 else np.zeros(1)
    xs = np.arange(len(vs_values))

    if fig is not None:
        axs = np.array(fig.get_axes()).reshape(n_rows, n_cols)
    else:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            sharex=True, sharey="row", squeeze=False)

    ls_cycle = ["-", "--", "-.", ":"]
    for col_idx, col_value in enumerate(col_values):
        if col_level is not None:
            axs[0, col_idx].set_title(f"{col_level} = {col_value}")
        for row_idx, param in enumerate(params):
            ax = axs[row_idx, col_idx]
            offset_idx = 0
            for ls_i, ls_value in enumerate(ls_values):
                if ls_level is not None:
                    plot_kwargs["linestyle"] = ls_cycle[ls_i % len(ls_cycle)]
                for inner in inner_values:
                    xs_args = [
                        (name, value) for name, value in
                        [(col_level, col_value), (ls_level, ls_value), (inner_level, inner)]
                        if name is not None]
                    if xs_args:
                        xs_names, xs_vals = zip(*xs_args)
                        try:
                            subset = fits.xs(xs_vals, level=xs_names)
                        except KeyError:
                            offset_idx += 1
                            continue
                    else:
                        subset = fits
                    subset = subset.reset_index().set_index(vs_level)
                    subset = subset.reindex(vs_values)

                    inner_kwargs = dict(plot_kwargs)
                    if "color" not in inner_kwargs and inner in inner_colors:
                        inner_kwargs["color"] = inner_colors[inner]

                    label_parts = []
                    if inner_level is not None:
                        label_parts.append(f"{inner}")
                    if ls_level is not None:
                        label_parts.append(f"{ls_value}")
                    label = ", ".join(label_parts) if label_parts else None

                    val_col = f"{param}{value_suffix}"
                    err_col = f"{param}{err_suffix}"
                    y = np.asarray(subset[val_col], dtype=float)
                    yerr = np.asarray(subset[err_col], dtype=float) if err_col in subset.columns else None
                    if param.startswith("log_"):
                        y_lin = np.exp(y)
                        if yerr is not None:
                            yerr = [y_lin - np.exp(y - yerr), np.exp(y + yerr) - y_lin]
                        y = y_lin
                    ax.errorbar(
                        xs + offsets[offset_idx],
                        y, yerr=yerr,
                        label=label, **inner_kwargs)
                    offset_idx += 1

            if param.startswith("log_"):
                ax.set_yscale("log")
            ylabel = "Fitted " + param.lstrip("log_")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.grid(which="major", alpha=0.5)

    for ax in axs[-1, :]:
        ax.set_xticks(xs, labels=vs_values)
        ax.set_xlabel(_AXIS_LABELS.get(vs_level, vs_level))

    fig.suptitle(suptitle)
    fig.tight_layout()
    if inner_level is not None or ls_level is not None:
        handles, labels = axs[0, 0].get_legend_handles_labels()
        legend_title = ", ".join(s for s in [inner_level, ls_level] if s is not None)
        fig.subplots_adjust(right=0.78)
        fig.legend(handles, labels, loc="center right", title=legend_title)
    return fig, axs


## Joint plotting of data and fits

def plot_sweep_LER_data_and_fits(
    dat, fits, col_sweep, row_sweep, ls_sweep=None, *,
    vs="p", ler_name="ler",
    ler_fit_max=None,
    data_plot_kwargs=None, fit_plot_kwargs=None, **kwargs):
    """ Plot data points and the corresponding fitted `ler_ansatz` curves on
    the same axes. Delegates to `plot_sweep_LER` for the data, then overlays
    the fits via `plot_sweep_LER_fits` on the same figure. `p_min` and
    `d_values` for the fit curves are inferred from `dat`.
    """
    if data_plot_kwargs is None:
        data_plot_kwargs = {}
    if fit_plot_kwargs is None:
        fit_plot_kwargs = {}

    inner_sweep = kwargs.pop("inner_sweep", None)
    if inner_sweep is None:
        inner_level = "d" if vs == "p" else "p"
        inner_values = sorted(dat.index.get_level_values(inner_level).unique())
    elif isinstance(inner_sweep, str):
        inner_level = inner_sweep
        inner_values = sorted(dat.index.get_level_values(inner_level).unique())
    else:
        inner_level, inner_values = inner_sweep
        inner_values = sorted(inner_values)
    if inner_level == "p":
        inner_values = inner_values[::-1]  # largest p first so smaller-p curves draw on top
    inner_sweep = (inner_level, inner_values)

    p_values = sorted(dat.index.get_level_values("p").unique())
    d_values = sorted(dat.index.get_level_values("d").unique())
    if inner_level == "p":
        p_values = inner_values
    elif inner_level == "d":
        d_values = inner_values

    data_plot_defaults = {'errorbars': True, 'errorbands': False, 'ls': 'none', 'marker': '.'}
    data_plot_kwargs = data_plot_defaults | kwargs | data_plot_kwargs
    fig, axs = plot_sweep_LER(
        dat, col_sweep, row_sweep, ls_sweep,
        vs=vs, inner_sweep=inner_sweep, ler_name=ler_name,
        **data_plot_kwargs)

    for legend in list(fig.legends):
        legend.remove()  # remove legend from data plot; we'll add a combined legend at the end

    fit_plot_kwargs = kwargs | fit_plot_kwargs
    dat_restricted = dat[dat[ler_name] <= ler_fit_max] if ler_fit_max is not None else dat
    plot_sweep_LER_fits(
        fits, col_sweep, row_sweep, ls_sweep,
        vs=vs, inner_sweep=inner_sweep,
        p_values=p_values, d_values=d_values,
        restrict_to=dat_restricted,
        ler_name=ler_name,
        **( {'ls': '-', 'lw': 0.8, 'fig': fig, 'axs': axs} | fit_plot_kwargs))

    return fig, axs
