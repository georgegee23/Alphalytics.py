
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

def downside_variance(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0, ddof: int = 1) -> Union[float, pd.Series]:
    """
    Downside variance — volatility of returns below the MAR.

    Only penalizes returns that fall below the Minimum Acceptable Return (MAR).
    The denominator is total observations (not just downside periods),
    consistent with the Sortino ratio convention.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (n <= ddof or n == 0).
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.Series or pd.DataFrame, got {type(returns).__name__}")

    n_obs = len(returns)
    insufficient_data = n_obs == 0 or n_obs <= ddof
    if insufficient_data:
        return pd.Series(np.nan, index=returns.columns)

    # Clip positive deviations to 0 — only penalize returns below MAR
    downside_deviations = (returns - mar).clip(upper=0)

    result = (downside_deviations ** 2).sum() / (n_obs - ddof)
    return result.squeeze()

def ann_downside_deviation(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    ddof: int = 1, periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Annualized downside deviation (square root of annualized downside variance).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 1.
        periods_per_year (int, optional): Trading periods in a year. Inferred if None.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (n <= ddof or n == 0).
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    return np.sqrt(downside_variance(returns, mar=mar, ddof=ddof) * periods_per_year)

def upside_variance(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0, ddof: int = 1) -> Union[float, pd.Series]:
    """
    Upside variance — volatility of returns above the MAR.

    Mirror of `downside_variance`. Only counts returns that exceed the
    Minimum Acceptable Return (MAR). The denominator is total observations
    (not just upside periods), matching the Sortino convention so the upside
    and downside variances are directly comparable and sum components are
    on the same scale as `var(returns, ddof=ddof)`.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (n <= ddof or n == 0).
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.Series or pd.DataFrame, got {type(returns).__name__}")

    n_obs = len(returns)
    insufficient_data = n_obs == 0 or n_obs <= ddof
    if insufficient_data:
        return pd.Series(np.nan, index=returns.columns)

    # Clip negative deviations to 0 — only count returns above MAR
    upside_deviations = (returns - mar).clip(lower=0)

    result = (upside_deviations ** 2).sum() / (n_obs - ddof)
    return result.squeeze()

def ann_upside_deviation(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    ddof: int = 1, periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Annualized upside deviation (square root of annualized upside variance).

    Mirror of `ann_downside_deviation`. Useful as the numerator in
    upside-vs-downside dispersion comparisons that share the Sortino-
    convention denominator (total N).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 1.
        periods_per_year (int, optional): Trading periods in a year. Inferred if None.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when observations are insufficient (n <= ddof or n == 0).
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

def to_drawdowns(returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates the historical drawdowns from a series of financial returns.
    
    Args:
        returns: A pandas Series (single asset) or DataFrame (multiple assets) of returns.
                 (e.g., 0.05 for 5%, -0.02 for -2%).
        
    Returns:
        A pandas Series or DataFrame of drawdowns (expressed as negative decimals).
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("Input must be a pandas Series or DataFrame.")
    
    # 1. Convert returns to a wealth index (cumulative growth)
    # We drop NAs first so the cumprod() doesn't propagate NaNs incorrectly
    clean_returns = returns.dropna()
    wealth_index = (1 + clean_returns).cumprod()
    
    # 2. Track the historical maximum (running peak)
    previous_peaks = wealth_index.cummax()
    
    # 3. Calculate the drawdown percentage
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Realign with the original index in case there were leading NaNs
    return drawdowns.reindex(returns.index)

# ==========================================
# MAX DRAWDOWN
# ==========================================

def max_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """Maximum drawdown from a series of returns."""
    return to_drawdowns(returns).min()

# ==========================================
# TOP DRAWDOWN
# ==========================================

def top_drawdowns(returns: pd.Series, n: int = 5, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Identifies the top N worst drawdowns for a single asset, including dates, duration, 
    and the annualized volatility experienced during the drawdown.
    
    Args:
        returns: A single pandas Series of returns.
        n: The number of top drawdowns to return.
        periods_per_year: Used to annualize the volatility (252 for daily, 12 for monthly).
        
    Returns:
        A pandas DataFrame detailing the worst drawdown periods.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Please pass a single pandas Series (e.g., returns['SPY']).")

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    # 1. Call the core function to get the drawdown timeseries
    drawdowns = to_drawdowns(returns)
    is_zero = drawdowns == 0
    period_id = is_zero.cumsum()
    underwater = drawdowns[drawdowns < 0]

    if underwater.empty:
        return pd.DataFrame()

    groups = underwater.groupby(period_id[underwater.index])

    results = []
    for _, group in groups:
        trough_date = group.idxmin()
        depth = group.min()

        prior_zeros = is_zero[:group.index[0]]
        peak_date = prior_zeros[prior_zeros].index[-1]

        post_group_zeros = is_zero[group.index[-1]:]
        recovery_dates = post_group_zeros[post_group_zeros].index

        if len(recovery_dates) > 0:
            recovery_date = recovery_dates[0]
            duration = len(returns.loc[peak_date:recovery_date]) - 1
            status = "Recovered"
        else:
            recovery_date = pd.NaT
            last_valid_date = drawdowns.dropna().index[-1]
            duration = len(returns.loc[peak_date:last_valid_date]) - 1
            status = "Ongoing"

        # Volatility during the drawdown (underwater periods only)
        period_returns = returns.loc[group.index[0]:group.index[-1]]

        if len(period_returns) < 2:
            period_volatility = np.nan
        else:
            period_volatility = period_returns.std() * np.sqrt(periods_per_year)
            
        results.append({
            "Peak": peak_date.strftime('%Y-%m-%d'),
            "Trough": trough_date.strftime('%Y-%m-%d'),
            "Recovery": recovery_date.strftime('%Y-%m-%d') if pd.notna(recovery_date) else "N/A",
            "Depth": depth,
            "Duration": duration,
            "Volatility (Ann)": period_volatility, # New Column
            "Status": status
        })
        
    # Compile, sort by the worst depth, and grab the top N
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
                - Time to Trough:    Periods from peak_date to the strategy's own
                                     wealth low within [peak_date, trough_date]
                                     (need not align with the benchmark trough).
                - Time to Recovery:  Periods from peak_date until the strategy's
                                     wealth returns to its pre-episode peak
                                     (level at peak_date), searched strictly
                                     after the benchmark's trough so all
                                     strategies are compared on the same
                                     reference point. NaN if it never recovers.
                - Periods Underwater:   Count of periods within [peak_date, trough_date]
                                     where strategy wealth is below the pre-episode
                                     peak.
    """
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pd.Series.")

    if isinstance(strategies, pd.Series):
        strategies = strategies.to_frame()

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(benchmark.index)

    # --- Identify benchmark drawdown periods ---
    bench_dd = to_drawdowns(benchmark)
    is_zero = bench_dd == 0
    period_id = is_zero.cumsum()
    underwater = bench_dd[bench_dd < 0]

    if underwater.empty:
        return {}

    groups = underwater.groupby(period_id[underwater.index])

    periods = []
    for _, group in groups:
        prior_zeros = is_zero[:group.index[0]]
        peak_date = prior_zeros[prior_zeros].index[-1]
        trough_date = group.idxmin()
        depth = group.min()
        periods.append((depth, peak_date, trough_date))

    periods.sort(key=lambda x: x[0])
    periods = periods[:n]

    result = {}
    for _, peak_date, trough_date in periods:

        rows = {}
        for col in strategies.columns:
            strat = strategies[col]

            # --- Depth: worst drawdown during the window ---
            period_ret = strat.loc[peak_date:trough_date].dropna()
            period_wealth = (1 + period_ret).cumprod()
            period_peak = period_wealth.cummax()
            period_dd = (period_wealth - period_peak) / period_peak
            strat_depth = period_dd.min() if len(period_dd) > 0 else np.nan

            # --- Volatility (annualised) ---
            if len(period_ret) > 1:
                vol = period_ret.iloc[1:].std() * np.sqrt(periods_per_year)
            else:
                vol = np.nan

            # --- Time to trough / recovery / days underwater ---
            # Reference = strategy wealth at peak_date (pre-episode peak = 1.0).
            # Returns AFTER peak_date drive the wealth path; the return ON
            # peak_date is excluded since it's what brought us to the peak.
            strat_clean = strat.dropna()
            post_peak = strat_clean[strat_clean.index > peak_date]

            time_to_trough = np.nan
            time_to_recovery = np.nan
            periods_underwater = 0

            if len(post_peak) > 0:
                wealth_after = (1 + post_peak).cumprod()

                # Episode portion (strictly after peak, through trough_date)
                wealth_episode = wealth_after.loc[:trough_date]
                if len(wealth_episode) > 0:
                    strat_trough_date = wealth_episode.idxmin()
                    time_to_trough = wealth_after.index.get_loc(strat_trough_date) + 1
                    periods_underwater = int((wealth_episode < 1.0).sum())

                # Recovery: search strictly AFTER the benchmark's trough so all
                # strategies are compared against the same reference point.
                post_bench_trough = wealth_after[wealth_after.index > trough_date]
                if len(post_bench_trough) > 0:
                    recovered = post_bench_trough[post_bench_trough >= 1.0]
                    if len(recovered) > 0:
                        recovery_date = recovered.index[0]
                        time_to_recovery = wealth_after.index.get_loc(recovery_date) + 1

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
        decimal, e.g. ``-0.36`` = 36% drawdown) as values. Sorted chronologically
        by peak date. Empty DataFrame if the benchmark never went underwater.
    """
    comparison = compare_drawdowns(strategies, benchmark, n=n)

    if not comparison:
        return pd.DataFrame()

    rows = {}
    for peak_str, sub_df in comparison.items():
        trough_str = sub_df["Trough"].iloc[0]
        rows[f"{peak_str} to {trough_str}"] = sub_df["Depth"]

    table = pd.DataFrame(rows).T.sort_index()
    table.index.name = "Date Range"
    table.columns.name = "Ticker"
    return table

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
        (in periods) as values. ``0`` means the strategy never went below
        its pre-episode peak; ``NaN`` means it went underwater and has not
        yet recovered. Sorted chronologically by peak date. Empty DataFrame
        if the benchmark never went underwater.
    """
    comparison = compare_drawdowns(strategies, benchmark, n=n)

    if not comparison:
        return pd.DataFrame()

    rows = {}
    for peak_str, sub_df in comparison.items():
        trough_str = sub_df["Trough"].iloc[0]
        rows[f"{peak_str} to {trough_str}"] = sub_df["Time to Recovery"]

    table = pd.DataFrame(rows).T.sort_index()
    table.index.name = "Date Range"
    table.columns.name = "Ticker"
    return table

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
