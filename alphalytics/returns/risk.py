
# ============== RISK METRICS ============== #

import pandas as pd
import numpy as np
from typing import Union

from .utils import _infer_periods_per_year


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
                - Peak / Trough:    Period boundary dates.
                - Depth:            Worst drawdown experienced during the window.
                - Volatility (Ann): Annualised std of returns during the window.
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

            rows[col] = {
                "Peak": peak_date.strftime('%Y-%m-%d'),
                "Trough": trough_date.strftime('%Y-%m-%d'),
                "Depth": strat_depth,
                "Volatility (Ann)": vol,
            }

        key = peak_date.strftime('%Y-%m-%d')
        result[key] = pd.DataFrame(rows).T

    return result

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
