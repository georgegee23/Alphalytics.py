
# ============== RISK METRICS ============== #

import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Optional

from alphalytics.utils import _infer_periods_per_year


# ==========================================
# ANNUAL VOLATILITY
# ==========================================

def annual_std(returns: Union[pd.Series, pd.DataFrame], periods_per_year: int = None, ddof: int = 1):
    """
    Calculates the annualized standard deviation of returns.

    Parameters:
    returns: A pandas Series or DataFrame of periodic returns.
    periods_per_year (int, optional): Trading periods in a year. Inferred if None.
    ddof (int): Delta Degrees of Freedom. Defaults to 1 for sample standard deviation.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)
        
    # Explicitly pass ddof to the pandas .std() method
    return returns.std(ddof=ddof) * np.sqrt(periods_per_year)

def rolling_volatility(returns: Union[pd.Series, pd.DataFrame], window: int,
    periods_per_year: int = None, ddof: int = 1) -> Union[pd.Series, pd.DataFrame]:
    """
    Annualized rolling volatility.

        sigma_t = std(R_{t-w+1}, ..., R_t) * sqrt(P)

    Rolling-window counterpart to `annual_std`. Same shape as the input
    (Series in / Series out, DataFrame in / DataFrame out), with the first
    `window - 1` rows NaN by construction and dropped from the result.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        window (int): Rolling window size in periods.
        periods_per_year (int, optional): Annualization factor P. Inferred if None.
        ddof (int): Delta Degrees of Freedom. Defaults to 1 (sample std).

    Returns:
        pd.Series (Series input) or pd.DataFrame (DataFrame input) of rolling
        annualized volatility.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    result = returns.rolling(window=window).std(ddof=ddof) * np.sqrt(periods_per_year)
    return result.dropna(how="all")

def vol_of_vol(returns: Union[pd.Series, pd.DataFrame], window: int,
    periods_per_year: int = None, ddof: int = 1, normalized: bool = True
    ) -> Union[float, pd.Series]:
    """
    Volatility of volatility — dispersion of the rolling volatility series.

        VolOfVol Ratio = std(sigma_t) / mean(sigma_t)   if normalized=True (default)
        VolOfVol       = std(sigma_t)                   if normalized=False

    sigma_t is the annualized rolling volatility from `rolling_volatility`.
    The normalized form is the coefficient of variation of rolling vol —
    unitless and comparable across strategies with different absolute risk
    levels — and is the default since cross-strategy comparison is the
    typical use case.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        window (int): Rolling window size used to compute sigma_t.
        periods_per_year (int, optional): Annualization factor. Inferred if None.
        ddof (int): Delta Degrees of Freedom for both the rolling std and the
            outer std. Defaults to 1.
        normalized (bool): If True, return std(sigma_t) / mean(sigma_t).
            Defaults to True.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when the rolling-vol series is too short for the outer std, or
        (normalized=True) when mean(sigma_t) is zero.
    """
    rol_vol = rolling_volatility(returns, window=window,
        periods_per_year=periods_per_year, ddof=ddof)

    sigma_std = rol_vol.std(ddof=ddof)

    if not normalized:
        return sigma_std

    sigma_mean = rol_vol.mean()

    if isinstance(sigma_mean, pd.Series):
        return sigma_std / sigma_mean.where(sigma_mean != 0)

    if pd.isna(sigma_mean) or sigma_mean == 0:
        return np.nan
    return float(sigma_std / sigma_mean)

def _semi_variance(returns: Union[pd.Series, pd.DataFrame], mar: float, ddof: int,
    side: str) -> Union[float, pd.Series]:
    """Second moment of returns on one side of ``mar``.

    The denominator is the count of finite returns in each column, minus
    ``ddof``. Missing rows are not observations. ``side="down"`` zeros out
    returns above ``mar``; ``side="up"`` zeros out returns below it.
    """
    if isinstance(returns, pd.Series):
        frame = returns.to_frame()
    elif isinstance(returns, pd.DataFrame):
        frame = returns
    else:
        raise TypeError(f"returns must be a pd.Series or pd.DataFrame, got {type(returns).__name__}")

    valid = frame.notna().sum()
    centered = frame - mar
    if side == "down":
        deviations = centered.clip(upper=0)
    else:
        deviations = centered.clip(lower=0)

    denom = (valid - ddof).where(valid > ddof)
    result = (deviations ** 2).sum() / denom
    return result.squeeze()


def downside_variance(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0, ddof: int = 0) -> Union[float, pd.Series]:
    """
    Downside variance — volatility of returns below the MAR.

    Only penalizes returns that fall below the Minimum Acceptable Return (MAR).
    The denominator is the count of finite observations (not just downside
    periods), which is the Sortino convention. The default ``ddof`` is 0,
    so that denominator is ``n``.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 0.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (finite n <= ddof or n == 0).
    """
    return _semi_variance(returns, mar=mar, ddof=ddof, side="down")

def ann_downside_deviation(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    ddof: int = 0, periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Annualized downside deviation (square root of annualized downside variance).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 0.
        periods_per_year (int, optional): Trading periods in a year. Inferred if None.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (finite n <= ddof or n == 0).
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    return np.sqrt(downside_variance(returns, mar=mar, ddof=ddof) * periods_per_year)

def upside_variance(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0, ddof: int = 0) -> Union[float, pd.Series]:
    """
    Upside variance — volatility of returns above the MAR.

    Mirror of `downside_variance`. Only counts returns that exceed the
    Minimum Acceptable Return (MAR). The denominator is the count of finite
    observations (not just upside periods), the same denominator as
    `downside_variance`.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 0.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (finite n <= ddof or n == 0).
    """
    return _semi_variance(returns, mar=mar, ddof=ddof, side="up")

def ann_upside_deviation(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    ddof: int = 0, periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Annualized upside deviation (square root of annualized upside variance).

    Mirror of `ann_downside_deviation`. Useful as the numerator in
    upside-vs-downside dispersion comparisons that share the Sortino-
    convention denominator (total N).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 0.
        periods_per_year (int, optional): Trading periods in a year. Inferred if None.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (finite n <= ddof or n == 0).
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    return np.sqrt(upside_variance(returns, mar=mar, ddof=ddof) * periods_per_year)

def variance_ratio(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0, ddof: int = 1) -> Union[float, pd.Series]:
    """
    Variance ratio — semi-variance divided by total variance.

        VR = downside_variance / var(returns)

    The share of total return dispersion attributable to moves below the MAR.
    A complement to skewness for assessing asymmetry of the risk profile:
    higher values indicate downside moves dominate total volatility.

    Both numerator and denominator use the same ddof. The numerator follows
    the Sortino convention (denominator = n - ddof, total observations), so
    with mar = mean(returns) the ratio is bounded in [0, 1]; with mar = 0
    (default) it can exceed 1 when the mean is sufficiently negative.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom. Defaults to 1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when total variance is zero or observations are insufficient.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(f"returns must be a pd.Series or pd.DataFrame, got {type(returns).__name__}")

    down_var = downside_variance(returns, mar=mar, ddof=ddof)
    total_var = returns.var(ddof=ddof)

    if isinstance(total_var, pd.Series):
        return down_var / total_var.where(total_var != 0)

    if pd.isna(total_var) or total_var == 0:
        return np.nan
    return float(down_var / total_var)

# ==========================================
# DRAWDOWNS
# ==========================================

def _series_drawdowns(returns: pd.Series) -> pd.Series:
    """Drawdown path for one series, anchored at starting wealth of 1.

    A gap in this series is skipped in its own wealth path and comes back
    as NaN. It does not affect any other series.
    """
    out = pd.Series(np.nan, index=returns.index, dtype=float, name=returns.name)
    clean = returns.dropna()
    if clean.empty:
        return out

    wealth = (1 + clean).cumprod()
    peaks = np.maximum(wealth.cummax().to_numpy(), 1.0)
    drawdowns = pd.Series(wealth.to_numpy() / peaks - 1.0, index=clean.index)
    # A new high is exactly zero. Tiny positives are floating-point noise.
    drawdowns = drawdowns.mask(drawdowns > 0, 0.0)
    out.loc[drawdowns.index] = drawdowns
    return out


def to_drawdowns(returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates the historical drawdowns from a series of financial returns.

    Wealth starts at 1 before the first return, so a loss in the opening
    period is a drawdown from invested capital. Each column is computed
    on its own: a missing return in one column does not remove that date
    from the others.

    Args:
        returns: A pandas Series (single asset) or DataFrame (multiple assets) of returns.
                 (e.g., 0.05 for 5%, -0.02 for -2%).

    Returns:
        A pandas Series or DataFrame of drawdowns (expressed as negative decimals).
    """
    if isinstance(returns, pd.Series):
        return _series_drawdowns(returns)
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame({col: _series_drawdowns(returns[col]) for col in returns.columns})
    raise TypeError("Input must be a pandas Series or DataFrame.")


def _index_pos(index: pd.Index, label) -> int:
    loc = index.get_loc(label)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        return int(np.flatnonzero(loc)[0])
    return int(loc)


def _episode_bounds(drawdowns: pd.Series) -> list:
    """Underwater episodes on an anchored drawdown path.

    ``inception`` is True when the episode begins before the first return,
    so there is no in-sample date with drawdown zero ahead of it. The
    peak date is then the first underwater timestamp, and that period's
    return belongs to the episode.

    Missing observations are dropped before the runs are formed. A gap
    inside a decline is not a recovery and does not start a second episode.
    """
    observed = drawdowns.dropna()
    under = observed.lt(0)
    if not bool(under.any()):
        return []

    run_id = under.ne(under.shift(fill_value=False)).cumsum()
    episodes = []
    for _, group in observed[under].groupby(run_id[under]):
        start = group.index[0]
        end = group.index[-1]
        start_pos = _index_pos(observed.index, start)
        prior = observed.iloc[:start_pos].eq(0)
        if bool(prior.any()):
            peak_date = prior[prior].index[-1]
            inception = False
        else:
            peak_date = start
            inception = True

        end_pos = _index_pos(observed.index, end)
        after = observed.iloc[end_pos + 1:]
        recovered = after[after.eq(0)]
        if len(recovered):
            recovery_date = recovered.index[0]
            status = "Recovered"
        else:
            recovery_date = pd.NaT
            status = "Ongoing"

        episodes.append({
            "peak_date": peak_date,
            "trough_date": group.idxmin(),
            "depth": float(group.min()),
            "inception": inception,
            "recovery_date": recovery_date,
            "status": status,
        })
    return episodes


def _window_returns(returns: pd.Series, peak_date, end_date, inception: bool) -> pd.Series:
    """Returns that belong to an episode, through ``end_date`` inclusive.

    When the peak is an in-sample high, the return on the peak date is
    excluded. When the peak is starting capital, the first return is included.
    """
    if len(returns.dropna()) == 0:
        return returns.dropna()
    if pd.isna(end_date):
        end_date = returns.dropna().index[-1]
    if inception:
        window = returns.loc[:end_date]
    else:
        window = returns.loc[(returns.index > peak_date) & (returns.index <= end_date)]
    return window.dropna()

# ==========================================
# MAX DRAWDOWN
# ==========================================

def max_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """Maximum drawdown from a series of returns."""
    return to_drawdowns(returns).min()

# ==========================================
# TOP DRAWDOWN
# ==========================================

def top_drawdowns(returns: pd.Series, n: int = 5, periods_per_year: int = None) -> pd.DataFrame:
    """
    Identifies the top N worst drawdowns for a single asset, including dates, duration,
    and the annualized volatility experienced during the drawdown.

    Args:
        returns: A single pandas Series of returns.
        n: The number of top drawdowns to return.
        periods_per_year: Used to annualize the volatility. Inferred from the
            index when omitted.

    Returns:
        A pandas DataFrame detailing the worst drawdown periods.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Please pass a single pandas Series (e.g., returns['SPY']).")

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    drawdowns = to_drawdowns(returns)
    episodes = _episode_bounds(drawdowns)
    if not episodes:
        return pd.DataFrame()

    last_valid = drawdowns.dropna().index[-1]
    results = []
    for episode in episodes:
        peak_date = episode["peak_date"]
        trough_date = episode["trough_date"]
        recovery_date = episode["recovery_date"]
        duration_end = recovery_date if pd.notna(recovery_date) else last_valid
        duration = len(_window_returns(returns, peak_date, duration_end, episode["inception"]))
        vol_returns = _window_returns(returns, peak_date, trough_date, episode["inception"])
        if len(vol_returns) < 2:
            period_volatility = np.nan
        else:
            period_volatility = vol_returns.std(ddof=1) * np.sqrt(periods_per_year)

        results.append({
            "Peak": peak_date.strftime('%Y-%m-%d'),
            "Trough": trough_date.strftime('%Y-%m-%d'),
            "Recovery": recovery_date.strftime('%Y-%m-%d') if pd.notna(recovery_date) else "N/A",
            "Depth": episode["depth"],
            "Duration": duration,
            "Volatility (Ann)": period_volatility,
            "Status": episode["status"],
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Depth", ascending=True).head(n)
    df.reset_index(drop=True, inplace=True)
    return df

# ==========================================
# COMPARE DRAWDOWNS
# ==========================================

def compare_drawdowns(strategies: Union[pd.Series, pd.DataFrame], benchmark: pd.Series, 
    n: int = 5, periods_per_year: int = None) -> dict:
    """
    Compares strategy behaviour during the benchmark's worst drawdown periods.

    Args:
        strategies: A pandas Series (single) or DataFrame (multiple) of strategy returns.
        benchmark:  A pandas Series of benchmark returns.
        n:          Number of top benchmark drawdowns to analyse.
        periods_per_year: Annualisation factor (inferred from index if None).

    Returns:
        dict: Keyed by benchmark peak date string (e.g. '2020-02-19').
              Each value is a DataFrame with strategies as rows and columns:
                - Peak / Trough:     Benchmark period boundary dates.
                - Depth:             Worst drawdown experienced during the window.
                - Volatility (Ann):  Annualised std of returns during the window.
                - Time to Trough:    Periods from the peak until the strategy's own
                                     wealth low inside the benchmark's peak-to-trough
                                     window (need not align with the benchmark trough).
                - Time to Recovery:  Periods from the peak until the strategy's
                                     wealth returns to its pre-episode peak, after
                                     first falling below it inside the benchmark's
                                     peak-to-trough window. The recovery may come
                                     after the benchmark trough. ``0`` if the
                                     strategy never trades below that peak inside
                                     the window. NaN if it went below and has not
                                     recovered by the end of the sample.
                - Periods Underwater: Count of periods from the peak through the
                                     benchmark trough where strategy wealth is
                                     below the pre-episode peak.

              The window covers only dates from the benchmark episode: when the
              episode starts at the benchmark's first return, strategy returns
              before that date are ignored. A strategy with no returns inside
              the window is NaN in every numeric column.
    """
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pd.Series.")

    if isinstance(strategies, pd.Series):
        strategies = strategies.to_frame()

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(benchmark.index)

    episodes = _episode_bounds(to_drawdowns(benchmark))
    if not episodes:
        return {}

    episodes = sorted(episodes, key=lambda episode: episode["depth"])[:n]

    result = {}
    for episode in episodes:
        peak_date = episode["peak_date"]
        trough_date = episode["trough_date"]
        inception = episode["inception"]

        rows = {}
        for col in strategies.columns:
            strat = strategies[col]
            post = strat.dropna()
            if inception:
                # The benchmark's first return opens the episode. Anything the
                # strategy earned before the benchmark began is not part of it.
                post = post[post.index >= peak_date]
            else:
                post = post[post.index > peak_date]
            episode_returns = post.loc[:trough_date] if len(post) else post

            if len(episode_returns) == 0:
                strat_depth = np.nan
                vol = np.nan
            else:
                strat_depth = float(to_drawdowns(episode_returns).min())
                if len(episode_returns) > 1:
                    vol = float(episode_returns.std(ddof=1) * np.sqrt(periods_per_year))
                else:
                    vol = np.nan

            # A strategy with no returns inside the window has nothing to report.
            time_to_trough = np.nan
            time_to_recovery = np.nan
            periods_underwater = np.nan

            if len(episode_returns) > 0:
                wealth = (1 + post).cumprod()
                wealth_episode = wealth.loc[:trough_date]
                strat_trough_date = wealth_episode.idxmin()
                time_to_trough = _index_pos(wealth.index, strat_trough_date) + 1
                underwater = wealth_episode < 1.0
                periods_underwater = int(underwater.sum())

                # Only a dip inside the benchmark's peak-to-trough window needs
                # recovering. A later, unrelated drawdown is not this episode's.
                # The recovery itself may come after the benchmark trough.
                if not bool(underwater.any()):
                    time_to_recovery = 0.0
                else:
                    breach = underwater[underwater].index[0]
                    recovered = wealth.loc[breach:]
                    recovered = recovered[recovered >= 1.0]
                    if len(recovered) > 0:
                        time_to_recovery = _index_pos(wealth.index, recovered.index[0]) + 1

            rows[col] = {
                "Peak": peak_date.strftime('%Y-%m-%d'),
                "Trough": trough_date.strftime('%Y-%m-%d'),
                "Depth": strat_depth,
                "Volatility (Ann)": vol,
                "Time to Trough": time_to_trough,
                "Time to Recovery": time_to_recovery,
                "Periods Underwater": periods_underwater,
            }

        key = peak_date.strftime('%Y-%m-%d')
        result[key] = pd.DataFrame(rows).T

    return result

# ==========================================
# DRAWDOWNS TABLE
# ==========================================

def _pivot_comparison(comparison: dict, field: str, columns: pd.Index) -> pd.DataFrame:
    """
    Reshapes a `compare_drawdowns` result into a flat table for one field.

    Args:
        comparison: The dict returned by `compare_drawdowns`.
        field:      Column of each sub-frame to extract (e.g. "Depth").
        columns:    Strategy names, used to fix the column order and to give
                    the table its shape when `comparison` is empty.

    Returns:
        A DataFrame with index "Date Range" ("peak to trough"), strategy
        tickers as columns, and float values.
    """
    rows = {}
    for peak_str, sub_df in comparison.items():
        # Empty when `strategies` has no columns — nothing to pull out.
        if sub_df.empty:
            continue
        trough_str = sub_df["Trough"].iloc[0]
        rows[f"{peak_str} to {trough_str}"] = sub_df[field]

    # `compare_drawdowns` builds its sub-frames from dicts mixing date strings
    # with numbers, so the extracted column is object dtype — cast it back or
    # .round()/.mean()/.style break downstream.
    # Keys are ISO dates, so a lexicographic sort_index() is chronological.
    table = pd.DataFrame(rows, index=columns).T.sort_index().astype(float)
    table.index.name = "Date Range"
    table.columns.name = "Ticker"
    return table

def drawdowns_table(strategies: Union[pd.Series, pd.DataFrame], benchmark: pd.Series,
    n: int = 5) -> pd.DataFrame:
    """
    Flat table of strategy drawdown depths during the benchmark's worst N drawdowns.

    Compact reshape of `compare_drawdowns` — one row per benchmark drawdown
    window (peak-to-trough), one column per strategy, values are the
    strategy's worst drawdown depth observed inside that window.

    Args:
        strategies: A pandas Series (single) or DataFrame (multiple) of strategy returns.
        benchmark:  A pandas Series of benchmark returns.
        n:          Number of top benchmark drawdowns to include.

    Returns:
        A DataFrame with index "Date Range" formatted ``"YYYY-MM-DD to YYYY-MM-DD"``
        (peak to trough), strategy tickers as columns, and depth (negative
        decimal, e.g. ``-0.36`` = 36% drawdown) as float values. Sorted
        chronologically by peak date. If the benchmark never went underwater,
        a row-less DataFrame with the same columns and axis names.
    """
    comparison = compare_drawdowns(strategies, benchmark, n=n)

    columns = (strategies.to_frame() if isinstance(strategies, pd.Series) else strategies).columns
    return _pivot_comparison(comparison, "Depth", columns)

# ==========================================
# TIME TO RECOVERY TABLE
# ==========================================

def time_to_recovery_table(strategies: Union[pd.Series, pd.DataFrame], benchmark: pd.Series,
    n: int = 5) -> pd.DataFrame:
    """
    Flat table of strategy time-to-recovery during the benchmark's worst N drawdowns.

    Compact reshape of `compare_drawdowns` — one row per benchmark drawdown
    window (peak-to-trough), one column per strategy, values are the number
    of periods the strategy took to return to its pre-episode peak (its
    wealth level at the benchmark peak date). The recovery may extend
    beyond the benchmark's trough date.

    Args:
        strategies: A pandas Series (single) or DataFrame (multiple) of strategy returns.
        benchmark:  A pandas Series of benchmark returns.
        n:          Number of top benchmark drawdowns to include.

    Returns:
        A DataFrame with index "Date Range" formatted ``"YYYY-MM-DD to YYYY-MM-DD"``
        (peak to trough), strategy tickers as columns, and time to recovery
        (in periods) as float values. ``0`` means the strategy never went below
        its pre-episode peak inside the benchmark's peak-to-trough window;
        ``NaN`` means it went underwater and has not yet recovered, or has no
        returns inside the window. Sorted chronologically by peak date. If the benchmark
        never went underwater, a row-less DataFrame with the same columns and
        axis names.
    """
    comparison = compare_drawdowns(strategies, benchmark, n=n)

    columns = (strategies.to_frame() if isinstance(strategies, pd.Series) else strategies).columns
    return _pivot_comparison(comparison, "Time to Recovery", columns)

# ==========================================
# AVERAGE DRAWDOWNS
# ==========================================

def average_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculates the average drawdown (the mean of all underwater periods).
    
    Args:
        returns: A pandas Series (single asset) or DataFrame (multiple assets) of returns.
        
    Returns:
        A float (if input is a Series) or a pandas Series (if input is a DataFrame).
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")
    
    # 1. Generate the continuous drawdown timeseries
    drawdowns = to_drawdowns(returns)
    
    # 2. Mask the all-time highs
    # By filtering for < 0, all the 0 values become NaN. 
    # This is crucial because pandas .mean() automatically ignores NaNs.
    underwater_only = drawdowns[drawdowns < 0]
    
    # 3. Calculate the mean of the underwater periods
    avg_drawdown = underwater_only.mean()
    
    # 4. Handle the edge case where an asset never went underwater
    if isinstance(avg_drawdown, pd.Series):
        return avg_drawdown.fillna(0.0)
    else:
        return float(avg_drawdown) if pd.notna(avg_drawdown) else 0.0

# ==========================================
# AVERAGE DRAWDOWN DURATION
# ==========================================

def average_drawdown_duration(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Average drawdown duration in periods across all underwater episodes.

    For each drawdown episode (peak to recovery), counts the number of periods
    spent underwater. Returns the mean across all episodes.

    Args:
        returns: A pandas Series (single asset) or DataFrame (multiple assets) of returns.

    Returns:
        A float (if input is a Series) or a pandas Series (if input is a DataFrame).
        Returns 0.0 if the asset was never underwater.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")

    if isinstance(returns, pd.DataFrame):
        return pd.Series(
            {col: average_drawdown_duration(returns[col]) for col in returns.columns}
        )

    dd = to_drawdowns(returns)
    is_zero = dd == 0
    period_id = is_zero.cumsum()
    underwater = dd[dd < 0]

    if underwater.empty:
        return 0.0

    durations = underwater.groupby(period_id[underwater.index]).size()

    return durations.mean()

# ==========================================
# ULCER INDEX
# ==========================================

def ulcer_index(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Ulcer Index — root-mean-square drawdown (Peter Martin).

        UI = sqrt( mean(D_t^2) )

    Where D_t is the drawdown from the running peak at time t. Unlike
    `max_drawdown`, the Ulcer Index penalises both depth *and* duration of
    drawdowns: long, shallow underwater stretches accumulate as much as
    short, deep ones. Peak periods (D_t = 0) are included in the mean.

    Output is on the same decimal scale as `to_drawdowns` (e.g. 0.08 = 8%).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
    """
    drawdowns = to_drawdowns(returns)
    return np.sqrt((drawdowns ** 2).mean())

# ==========================================
# CONDITIONAL VAR (EXPECTED SHORTFALL)
# ==========================================

def cvar(returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.95) -> Union[float, pd.Series]:
    """
    Historical Conditional Value-at-Risk (Expected Shortfall).

        CVaR_α = E[ r | r <= q_(1-α) ]

    The mean return in the worst (1 - α) tail of the empirical distribution.
    Returned in the same sign convention as the input — losses come out
    negative (e.g. a CVaR of -0.04 means the expected loss in the worst-tail
    scenario is 4%).

    No distributional assumption: this is the historical/empirical estimator,
    appropriate for fat-tailed return series where Gaussian VaR understates
    risk.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        alpha: Confidence level. Defaults to 0.95 (5% tail).

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN where the series is empty or the tail contains no observations.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: cvar(col, alpha))

    r = returns.dropna()
    if r.empty: return np.nan

    threshold = r.quantile(1 - alpha)
    tail = r[r <= threshold]

    if tail.empty: return np.nan
    return float(tail.mean())


# ==========================================
# ANDERSON-DARLING NORMALITY P-VALUE
# ==========================================

def anderson_darling_pvalue(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Anderson-Darling test for normality — returns the p-value.

    Tests the null hypothesis that returns are drawn from a normal
    distribution. Small p-values (< 0.05) indicate rejection — the empirical
    distribution deviates from normal more than chance would explain. The
    AD statistic puts extra weight on the tails, so it is particularly
    sensitive to fat-tailed deviations.

    The p-value is computed from the modified statistic via the
    D'Agostino & Stephens (1986) piecewise approximation, which is the
    standard closed-form mapping used in most statistical packages.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.

    Returns:
        A float (Series input) or pd.Series (DataFrame input) of p-values
        in [0, 1]. NaN where the series has fewer than 8 observations
        (the regime where the approximation is unreliable).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(anderson_darling_pvalue)

    r = returns.dropna()
    n = len(r)
    if n < 8: return np.nan

    a2 = stats.anderson(r.values, dist="norm").statistic
    a2_mod = a2 * (1 + 0.75 / n + 2.25 / n ** 2)

    if a2_mod >= 0.6:
        p = np.exp(1.2937 - 5.709 * a2_mod + 0.0186 * a2_mod ** 2)
    elif a2_mod >= 0.34:
        p = np.exp(0.9177 - 4.279 * a2_mod - 1.38 * a2_mod ** 2)
    elif a2_mod >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a2_mod - 59.938 * a2_mod ** 2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a2_mod - 223.73 * a2_mod ** 2)

    return float(np.clip(p, 0.0, 1.0))


# ==========================================
# BOWLEY SKEWNESS
# ==========================================

def bowley_skewness(returns: Union[pd.Series, pd.DataFrame],
    q: float = 0.25) -> Union[float, pd.Series]:
    """
    Bowley (quartile-based) skewness coefficient.

        Bowley = (Q_{1-q} + Q_q - 2 * median) / (Q_{1-q} - Q_q)

    Robust, quantile-based skewness measure bounded in [-1, 1]. Negative
    values indicate a left-skewed (downside-heavy) distribution; positive
    values indicate a right-skewed distribution. Less sensitive to outliers
    than the third-moment skewness, since it depends only on the inter-quartile
    spread and the median.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        q: Lower-tail quantile defining the spread. Defaults to 0.25
            (classic Bowley using Q1, Q2, Q3). Pass a smaller value
            (e.g. 0.10) for the Hinkley generalization that puts more
            weight on the tails.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when the inter-quantile range is zero or observations are
        insufficient.
    """
    if not 0 < q < 0.5:
        raise ValueError(f"q must be in (0, 0.5), got {q}")

    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: bowley_skewness(col, q=q))

    r = returns.dropna()
    if len(r) < 3: return np.nan

    q_lo, median, q_hi = r.quantile([q, 0.5, 1 - q]).values
    iqr = q_hi - q_lo
    if iqr <= 0: return np.nan

    return float((q_hi + q_lo - 2 * median) / iqr)


# ==========================================
# TAIL COUNT
# ==========================================

def tail_count(returns: Union[pd.Series, pd.DataFrame],
    tail: str = "left", quantile: float = 0.05) -> Union[int, pd.Series]:
    """
    Count of observations beyond an empirical tail quantile.

        N_left  = #{ r_t : r_t < Q(quantile) }
        N_right = #{ r_t : r_t > Q(1 - quantile) }

    Distribution-free tail-mass diagnostic. NaNs are dropped before the
    quantile is estimated, so the count reflects the observed sample only.
    The strict inequality means observations exactly on the boundary are
    excluded, so for continuous data the count is approximately
    ``quantile * n``; ties at the cutoff (common with discretized or
    rounded data) can produce a smaller count.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        tail: ``"left"`` (default) for the loss tail, ``"right"`` for the
            gain tail.
        quantile: Tail cutoff in (0, 0.5). Defaults to 0.05 (5% / 95%).

    Returns:
        An int (Series input) or pd.Series of ints (DataFrame input).
        0 when the series is empty.
    """
    if tail not in ("left", "right"):
        raise ValueError(f"tail must be 'left' or 'right', got {tail!r}")
    if not 0 < quantile < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")

    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: tail_count(col, tail=tail, quantile=quantile))

    r = returns.dropna()
    if r.empty: return 0

    if tail == "left":
        threshold = r.quantile(quantile)
        return int((r < threshold).sum())

    threshold = r.quantile(1 - quantile)
    return int((r > threshold).sum())


# ==========================================
# TAIL INDEX (HILL ESTIMATOR)
# ==========================================

def tail_index(returns: Union[pd.Series, pd.DataFrame],
    tail: str = "left", k: Optional[int] = None,
    frac: float = 0.1) -> Union[float, pd.Series]:
    """
    Hill estimator of the tail index ``α`` for a return distribution.

    For a power-law tail ``P(|R| > x) ~ x^{-α}``, the tail index α
    quantifies how fast the tail decays — *smaller α means a heavier
    tail* (more mass in extreme outcomes). Typical equity returns sit
    around α ≈ 3-5; values below ~3 imply infinite kurtosis, below ~2
    imply infinite variance.

    The Hill estimator is computed on the ``k`` largest tail magnitudes:

        ξ̂_H = (1/k) Σ_{i=1}^{k} ln(X_{(i)} / X_{(k+1)}),    α̂ = 1 / ξ̂_H

    where X_{(1)} ≥ X_{(2)} ≥ … are the sorted absolute tail values
    (losses for ``tail="left"``, gains for ``tail="right"``).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        tail: ``"left"`` (default) for the loss tail, ``"right"`` for
            the gain tail.
        k: Number of order statistics used in the estimator. If ``None``,
            defaults to ``max(10, int(frac * n_tail))``, where ``n_tail``
            is the count of strictly negative (or positive) observations.
        frac: Fraction of tail observations to use when ``k`` is None.
            Defaults to ``0.1`` (top 10% of the chosen tail).

    Returns:
        A float (Series input) or pd.Series (DataFrame input). NaN when
        the chosen tail has too few observations or the estimator
        degenerates (non-positive ξ̂).
    """
    if tail not in ("left", "right"):
        raise ValueError(f"tail must be 'left' or 'right', got {tail!r}")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")

    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: tail_index(col, tail=tail, k=k, frac=frac))

    r = returns.dropna().values
    if len(r) < 20: return np.nan

    if tail == "left":
        tail_vals = -r[r < 0]
    else:
        tail_vals = r[r > 0]

    n_tail = len(tail_vals)
    if n_tail < 11: return np.nan

    x = np.sort(tail_vals)[::-1]

    k_use = int(k) if k is not None else max(10, int(frac * n_tail))
    k_use = min(k_use, n_tail - 1)
    if k_use < 5: return np.nan

    threshold = x[k_use]
    if threshold <= 0: return np.nan

    xi = float(np.mean(np.log(x[:k_use] / threshold)))
    if xi <= 0: return np.nan

    return 1.0 / xi
