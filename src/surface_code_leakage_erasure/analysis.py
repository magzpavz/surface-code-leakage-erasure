from warnings import warn
import re
import numpy as np
import pandas as pd
from sinter import fit_binomial
from scipy.optimize import curve_fit


def recalculate_LERs(dat : pd.DataFrame, max_likelihood_factor = 9.0, fail_prefix = "fails", pfail_prefix = "pfail", ler_prefix = "ler", samples_name="num_samples", rounds_name="r"):
    """ Recalculate LERs using a binomial fit to the failure counts. Modifies `dat` in place.

    Iterates over columns of `dat` matching `^{fail_prefix}`, calculates the
    corresponding failure probabilities with bounds, and writes results into
    `{pfail_prefix}{suffix}`, `{pfail_prefix}{suffix}_lo`, and
    `{pfail_prefix}{suffix}_hi`, where `suffix` is whatever follows `fail_prefix`
    in the matched column name.

    If the `rounds_name` index or column is present, the LERs are calculated as
    `pfail / rounds`. Otherwise, no LER is calculated and a warning is issued.
    """
    pattern = re.compile(rf"^{fail_prefix}")
    fail_name_list = [c for c in dat.columns if pattern.match(c)]
    if len(fail_name_list) == 0:
        raise ValueError(f"No columns found matching prefix {fail_prefix!r}")

    for fail_col in fail_name_list:
        suffix = fail_col[len(fail_prefix):]
        pfail_col = pfail_prefix + suffix
        pfail_lo_col = pfail_col + "_lo"
        pfail_hi_col = pfail_col + "_hi"
        samples_col = samples_name + suffix if samples_name + suffix in dat.columns else samples_name
        for col in (pfail_col, pfail_lo_col, pfail_hi_col):
            if col not in dat.columns:  # initialize pfail columns if they don't exist
                dat[col] = np.nan
        for idx, row in dat.iterrows():
            if int(row[fail_col]) >= 0:
                fit = fit_binomial(
                    num_shots=int(row[samples_col]),
                    num_hits=int(row[fail_col]),
                    max_likelihood_factor=max_likelihood_factor,
                )
                dat.at[idx, pfail_col] = fit.best
                dat.at[idx, pfail_lo_col] = fit.low
                dat.at[idx, pfail_hi_col] = fit.high
            else:
                dat.at[idx, pfail_col] = np.nan
                dat.at[idx, pfail_lo_col] = np.nan
                dat.at[idx, pfail_hi_col] = np.nan

    try:
        r = dat[rounds_name] if rounds_name in dat.columns else dat.index.get_level_values(rounds_name)
        for fail_col in fail_name_list:
            suffix = fail_col[len(fail_prefix):]
            for bound in ("", "_lo", "_hi"):
                pfail_col = pfail_prefix + suffix + bound
                ler_col = ler_prefix + suffix + bound
                dat[ler_col] = dat[pfail_col] / r
    except KeyError:
        warn(f"Index or column {rounds_name!r} not found; LERs not calculated")


def pivot_by_suffix(dat : pd.DataFrame, new_index : str, pivot_suffixes : list[str]):
    """ Pivot a DataFrame with columns of the form `{prefix}_{suffix}` to a MultiIndex DataFrame with index 'new_index' whose levels are the suffixes and columns `{prefix}`. Columns that don't match the pattern are kept in both rows.

    Arguments:
        dat: DataFrame with columns of the form `{prefix}_{suffix}`
        new_index: Name of the new index level to create from the suffixes
        pivot_suffixes: List of suffixes to pivot on. Only columns with these suffixes will be pivoted.
    """

    new_dfs = []

    common_cols = [col for col in dat.columns if not(any(
        suffix in col for suffix in pivot_suffixes))]

    for suffix in pivot_suffixes:
        # Separate columns by the suffixes
        col_names = [col for col in dat.columns if suffix in col] + common_cols
        df = dat[col_names].copy()

        # Extract clean column names by removing the suffix
        new_col_names = [col.replace('_' + suffix, '') for col in col_names]
        df.columns = new_col_names
        df[new_index] = suffix

        new_dfs.append(df)

    # Concatenate and add 'decoding' as a new index level
    dat_pivoted = pd.concat(new_dfs).set_index(new_index, append=True)

    return dat_pivoted


def unpivot_by_suffix(dat : pd.DataFrame, pivot_index : str):
    """ Inverse of `pivot_by_suffix`. Takes a DataFrame whose index includes
    `pivot_index` (the level holding suffix values) and re-appends `_{suffix}`
    to the columns that vary across suffixes, collapsing the suffix level.

    Columns whose values are identical across all suffix values for a given
    outer-index row are treated as common columns and left un-suffixed.
    """
    if pivot_index not in dat.index.names:
        raise ValueError(f"{pivot_index!r} is not a level of the index")

    suffixes = list(dat.index.get_level_values(pivot_index).unique())
    other_levels = [n for n in dat.index.names if n != pivot_index]

    groups = {s: dat.xs(s, level=pivot_index) for s in suffixes}

    common_cols = []
    varying_cols = []
    for col in dat.columns:
        ref = groups[suffixes[0]][col]
        is_common = True
        for s in suffixes[1:]:
            other = groups[s][col]
            aligned_ref, aligned_other = ref.align(other)
            if not aligned_ref.equals(aligned_other):
                is_common = False
                break
        (common_cols if is_common else varying_cols).append(col)

    pieces = []
    if common_cols:
        pieces.append(groups[suffixes[0]][common_cols])
    for s in suffixes:
        piece = groups[s][varying_cols].copy()
        piece.columns = [f"{c}_{s}" for c in varying_cols]
        pieces.append(piece)

    result = pd.concat(pieces, axis=1)
    return result


def ler_ansatz(x, log_a, alpha, log_pth):
    d, log_p = x
    return log_a + alpha * d * (log_p - log_pth)


_FIT_OUT_COLS = ["log_a", "alpha", "log_p_th", "p_th", "log_p_th_err", "alpha_err"]


def fit_ler_ansatz(
    sub: pd.DataFrame,
    d_level: str = "d",
    p_level: str = "p",
    ler_col: str = "ler",
    ler_lo_col: str = "ler_lo",
    ler_hi_col: str = "ler_hi",
    ler_max: float = 0.01,
    p0: tuple = (-1.0, 1.0, -2.0),
):
    """ Fit `ler_ansatz` to (d, log p) -> log(LER) for a single DataFrame.

    Only points with 0 < LER < `ler_max` are included. Sigma is computed in
    log-space as (log(ler_hi) - log(ler_lo)) / 2.

    Returns a list [log_a, alpha, log_p_th, p_th, log_p_th_err, alpha_err],
    or NaNs if the fit fails or there are too few points.
    """
    sub = sub[(sub[ler_col] > 0) & (sub[ler_col] < ler_max)]
    if len(sub) < 3:
        return [np.nan] * len(_FIT_OUT_COLS)
    d_vals = sub.index.get_level_values(d_level).to_numpy(dtype=float)
    p_vals = sub.index.get_level_values(p_level).to_numpy(dtype=float)
    log_p = np.log(p_vals)
    log_ler = np.log(sub[ler_col].to_numpy(dtype=float))
    sigma = (np.log(sub[ler_hi_col].to_numpy(dtype=float))
             - np.log(sub[ler_lo_col].to_numpy(dtype=float))) / 2
    try:
        popt, pcov = curve_fit(
            ler_ansatz, (d_vals, log_p), log_ler,
            sigma=sigma, absolute_sigma=True, p0=p0,
        )
        perr = np.sqrt(np.diag(pcov))
        return [popt[0], popt[1], popt[2], np.exp(popt[2]), perr[2], perr[1]]
    except (RuntimeError, ValueError) as e:
        warn(f"Fit failed: {e}")
        return [np.nan] * len(_FIT_OUT_COLS)


def fit_ler_by_group(
    dat: pd.DataFrame,
    d_level: str = "d",
    p_level: str = "p",
    ler_col: str = "ler",
    ler_lo_col: str = "ler_lo",
    ler_hi_col: str = "ler_hi",
    ler_max: float = 0.01,
    p0: tuple = (-1.0, 1.0, -2.0),
):
    """ Fit `ler_ansatz` to (d, log p) -> log(LER) for each combination of
    index levels other than `d_level` and `p_level`.

    Returns a DataFrame indexed by the grouping levels with columns
    log_a, alpha, log_p_th, p_th, log_p_th_err, alpha_err.
    """
    kwargs = dict(
        d_level=d_level, p_level=p_level,
        ler_col=ler_col, ler_lo_col=ler_lo_col, ler_hi_col=ler_hi_col,
        ler_max=ler_max, p0=p0,
    )
    group_levels = [n for n in dat.index.names if n not in (d_level, p_level)]

    if not group_levels:
        return pd.DataFrame([fit_ler_ansatz(dat, **kwargs)], columns=_FIT_OUT_COLS)

    results = {key: fit_ler_ansatz(sub, **kwargs)
               for key, sub in dat.groupby(level=group_levels)}

    index = pd.MultiIndex.from_tuples(
        [k if isinstance(k, tuple) else (k,) for k in results.keys()],
        names=group_levels,
    )
    return pd.DataFrame(list(results.values()), index=index, columns=_FIT_OUT_COLS)
