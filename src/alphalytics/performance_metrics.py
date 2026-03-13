
import pandas as pd
import numpy as np
from typing import Union
import quantstats as qs
import warnings

from .utils import fill_first_nan  # Import from utils module



# ============== RETURNS FUNCTIONS ============== #

def return_n(rets: pd.DataFrame, n: int) -> pd.Series:
    """
    Calculates the cumulative trailing return over the last 'n' periods.
    """
    if len(rets) < n:
        return pd.Series(np.nan, index=rets.columns)
        
    ret_n = (rets.iloc[-n:] + 1).prod() - 1
    ret_n.name = f"{n} Period"
    return ret_n

def return_ytd(rets: pd.DataFrame) -> pd.Series:
    """
    Calculates YTD return using the calendar year of the last available date.
    Assumes rets has a DatetimeIndex.
    """
    if rets.empty: 
        return pd.Series(0.0, index=rets.columns)
    
    # Identify the current year based on the very last data point
    current_year = rets.index[-1].year
    
    # Slice only the returns belonging to that year
    ytd_data = rets[rets.index.year == current_year]
    
    # Calculate geometric return for that slice
    ytd_rets = (ytd_data + 1).prod() - 1
    ytd_rets.name = "YTD"
    return ytd_rets

def ann_return(rets: pd.DataFrame, years: int = 3, periods_per_year: int = 12) -> pd.Series:
    """
    Calculates the annualized trailing return over a specific number of years.
    """
    if years < 1:
        raise ValueError(f"years must be >= 1 to annualize, got {years}")

    n = years * periods_per_year
    
    # Safety check: If history is shorter than requested periods, return NaN
    if len(rets) < n:
        return pd.Series(np.nan, index=rets.columns)

    # 1. Calculate cumulative return over the sliced last 'n' periods
    n_ret = (rets.iloc[-n:] + 1).prod() - 1
    
    # 2. Annualize the cumulative return
    ann_ret = (n_ret + 1) ** (1 / years) - 1
    ann_ret.name = f"{years} Year"
    
    return ann_ret

def ann_return_common_si(rets: pd.DataFrame, periods_per_year: int = 12) -> pd.Series:
    """
    Calculates the annualized Since Inception return over the COMMON period.
    Forces an apples-to-apples comparison by only evaluating dates where 
    all strategies in the DataFrame have overlapping data.
    """
    if rets.empty:
        return pd.Series(np.nan, index=rets.columns)
        
    # 1. Align data: Keep only dates where ALL columns have data
    aligned_rets = rets.dropna()
    
    total_periods = len(aligned_rets)
    
    # 2. Safety Check: Ensure the overlapping period is at least 1 year
    if total_periods < periods_per_year:
        raise ValueError("The overlapping 'apples-to-apples' history is less than 1 year. Cannot annualize.")
        
    years = total_periods / periods_per_year
    
    # 3. Calculate total cumulative return over the perfectly aligned period
    cumret = (aligned_rets + 1).prod() - 1
    
    # 4. Annualize based on the exact fractional years of the common period
    si_ret = (cumret + 1) ** (1 / years) - 1
    si_ret.name = "SI"
    
    return si_ret

def performance_table(rets: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """
    Builds a standard institutional performance table (1M, 3M, YTD, 1Y, 3Y, 5Y, 10Y, SI).
    Forces an apples-to-apples comparison by aligning all dates first.
    
    Parameters
    ----------
    rets : pd.DataFrame
        Periodic returns of the strategies.
    periods_per_year : int, default 12
        Frequency of the data (12 for monthly, 252 for daily).
        
    Returns
    -------
    pd.DataFrame
        A table where rows are strategies and columns are trailing timeframes.
    """
    # 1. Align all data to guarantee apples-to-apples comparison
    aligned_rets = rets.dropna()
    
    if aligned_rets.empty:
        raise ValueError("No common dates found across all strategies.")

    # 2. Calculate individual metrics using your toolkit
    
    # 1-Month Return (Safely compounds daily/weekly data if needed)
    m1_n = int(periods_per_year / 12)
    m1 = return_n(aligned_rets, m1_n)
    m1.name = "1M"

    # 3-Month Return
    m3_n = int(periods_per_year / 4)
    m3 = return_n(aligned_rets, m3_n) 
    m3.name = "3M"

    ytd = return_ytd(aligned_rets)
    yr1 = ann_return(aligned_rets, years=1, periods_per_year=periods_per_year)
    yr3 = ann_return(aligned_rets, years=3, periods_per_year=periods_per_year)
    yr5 = ann_return(aligned_rets, years=5, periods_per_year=periods_per_year)
    yr10 = ann_return(aligned_rets, years=10, periods_per_year=periods_per_year)
    
    # 3. Calculate Common SI safely (in case they share less than 1 year)
    try:
        si = ann_return_common_si(aligned_rets, periods_per_year=periods_per_year)
    except ValueError:
        si = pd.Series(np.nan, index=aligned_rets.columns, name="Common SI")
        
    # 4. Concatenate into a single presentation DataFrame
    perf_table = pd.concat([m1, m3, ytd, yr1, yr3, yr5, yr10, si], axis=1)
    
    return perf_table


# ============== RISK & RETURN METRICS ============== #

def downside_variance(returns: pd.DataFrame, mar: float = 0.0, ddof: int = 1,) -> pd.Series:
    """
    Downside variance for each strategy — volatility of returns below the MAR.

    Only penalizes returns that fall below the Minimum Acceptable Return (MAR).
    The denominator is total observations (not just downside periods),
    consistent with the Sortino ratio convention.

    Args:
        returns (pd.DataFrame): Periodic returns, one column per strategy.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom for the variance denominator. Defaults to 1.

    Returns:
        pd.Series: Downside variance per strategy, indexed by strategy name.
        NaN when observations are insufficient (n <= ddof or n == 0).

    Raises:
        TypeError: If returns is not a pd.DataFrame.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")

    n_obs = len(returns)
    insufficient_data = n_obs == 0 or n_obs <= ddof
    if insufficient_data:
        return pd.Series(np.nan, index=returns.columns)

    # Clip positive deviations to 0 — only penalize returns below MAR
    downside_deviations = (returns - mar).clip(upper=0)

    return (downside_deviations ** 2).sum() / (n_obs - ddof)

def sortino_ratio(returns: pd.DataFrame, mar: float = 0.0, periods_per_year: int = 12, ddof: int = 1) -> pd.Series:
    """
    Annualized Sortino Ratio for each strategy.

    Improves on the Sharpe Ratio by penalizing only downside volatility,
    making it more appropriate for strategies with asymmetric return distributions.

    Annualization: mean excess return scaled by periods_per_year,
    downside deviation scaled by sqrt(periods_per_year).

    Args:
        returns (pd.DataFrame): Periodic returns, one column per strategy.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        periods_per_year (int): Annualization factor (12 = monthly, 252 = daily).
        ddof (int): Degrees of freedom for downside variance. Defaults to 1.

    Returns:
        pd.Series: Annualized Sortino Ratio per strategy.
        NaN when downside deviation is zero or undefined.

    Raises:
        TypeError: If returns is not a pd.DataFrame.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")

    down_dev = np.sqrt(downside_variance(returns, mar=mar, ddof=ddof))
    down_dev = down_dev.replace(0, np.nan)  # guard: strategy never fell below MAR

    mean_excess_return = returns.mean() - mar

    # Calculate the ratio and annualize
    # Note: We scale the final ratio by the square root of periods_per_year
    sortino = (mean_excess_return / down_dev) * np.sqrt(periods_per_year)

    # Annualize: numerator scales linearly, denominator scales by √periods_per_year
    return sortino

def beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """
    Calculates beta for multiple strategies against a benchmark.

    Beta = Cov(Strategy, Benchmark) / Var(Benchmark) using sample statistics (ddof=1).
    Requires at least 2 observations; otherwise returns NaN for each strategy.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies (columns = strategy names).
        benchmark (pd.Series): Periodic returns of the benchmark.

    Returns:
        pd.Series: Beta for each strategy.

    Raises:
        ValueError: If returns and benchmark do not share the same index.
    """
    if not returns.index.equals(benchmark.index):
        raise ValueError(
            "returns and benchmark must share the same index. "
            "Align them before calling calculate_beta()."
        )

    if len(benchmark) < 2:
        return pd.Series(np.nan, index=returns.columns)

    bm_variance = benchmark.var(ddof=1)

    if bm_variance == 0:
        return pd.Series(np.nan, index=returns.columns)

    ret_centered = returns - returns.mean()
    bm_centered = benchmark - benchmark.mean()

    covariances = ret_centered.mul(bm_centered, axis=0).sum() / (len(benchmark) - 1)

    return covariances / bm_variance

def bull_bear_beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Bull Market Beta (benchmark > 0) and Bear Market Beta (benchmark < 0)
    for each strategy.

    Computed on the common overlapping period only, so all strategies are
    comparable. Zero-return benchmark periods are excluded from both regimes.
    NaN is returned for a regime with fewer than 2 observations.

    Args:
        returns (pd.DataFrame): Periodic returns, one column per strategy.
        benchmark (pd.Series): Periodic benchmark returns.

    Returns:
        pd.DataFrame: Bull Beta and Bear Beta per strategy (index = strategy names).

    Raises:
        TypeError: If inputs are not the expected pandas types.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()

    if combined.empty:
        return pd.DataFrame(
            {"Bull Beta": np.nan, "Bear Beta": np.nan},
            index=returns.columns,
        )

    aligned_returns = combined[returns.columns]
    aligned_bench   = combined["__benchmark__"]

    bull_periods = aligned_bench > 0
    bear_periods = aligned_bench < 0

    bull_beta = beta(aligned_returns[bull_periods], aligned_bench[bull_periods])
    bear_beta = beta(aligned_returns[bear_periods], aligned_bench[bear_periods])

    return pd.DataFrame({"Bull Beta": bull_beta, "Bear Beta": bear_beta})

def compute_capm(returns: pd.DataFrame, benchmark: pd.Series = None, periods_per_year: int = 12) -> pd.DataFrame:
    """
    Calculates CAPM Beta and Annualized Alpha for multiple strategies simultaneously.
    
    Args:
        returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark (pd.Series, optional): Periodic returns of the benchmark. 
                                         Defaults to an equal-weight average of all strategies.
        annualization_factor (int): Periods per year (e.g., 252 for daily, 12 for monthly). 
                                    Defaults to 252 (daily trading days).
        
    Returns:
        pd.DataFrame: A summary table with Beta, Periodic Alpha, and Annualized Alpha for each strategy.
    """
    # 1. Validate inputs
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
        
    if returns.empty:
        raise ValueError("returns cannot be empty")

    if benchmark is None:
        benchmark = returns.mean(axis=1)
        benchmark.name = "Benchmark"
    elif not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    # 2. Align data and handle missing values naturally
    data = returns.copy()
    data['bench_'] = benchmark
    data = data.dropna()
    
    aligned_returns = data.drop(columns=['bench_'])
    aligned_bench = data['bench_']
    
    # 3. Vectorized CAPM Math
    returns_centered = aligned_returns - aligned_returns.mean()
    bench_centered = aligned_bench - aligned_bench.mean()
    
    dof = len(data) - 1
    covariances = (returns_centered.mul(bench_centered, axis=0)).sum() / dof
    bench_variance = aligned_bench.var(ddof=1) 
    
    # Calculate Beta
    betas = covariances / bench_variance
    
    # Calculate Periodic Alpha, then Annualize it
    # Annualized Alpha = Periodic Alpha * Periods per Year
    periodic_alphas = aligned_returns.mean() - (betas * aligned_bench.mean())
    annualized_alphas = periodic_alphas * periods_per_year
    
    # 4. Format Output
    capm_df = pd.DataFrame({
        "Beta": betas,
        "Periodic Alpha": periodic_alphas,
        "Annualized Alpha": annualized_alphas
    })
    
    return capm_df


# ==========================================
# BATTING AVERAGES (Hit Rates)
# ==========================================

def batting_average(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall percentage of periods the strategy beats the benchmark."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: batting_average(col, benchmark_returns))

    excess = (strategy_returns - benchmark_returns).dropna()
    if excess.empty: return np.nan
        
    return float((excess > 0).mean())

def bull_batting_average(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Batting average only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_batting_average(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = strat[bench > 0] - bench[bench > 0]
    
    if bull_excess.empty: return np.nan
        
    return float((bull_excess > 0).mean())

def bear_batting_average(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Batting average only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_batting_average(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = strat[bench <= 0] - bench[bench <= 0]
    
    if bear_excess.empty: return np.nan
        
    return float((bear_excess > 0).mean())

# ==========================================
# WIN/LOSS RATIOS (Payoff Ratios)
# ==========================================

def win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall ratio of average outperformance to absolute average underperformance."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: win_loss_ratio(col, benchmark_returns))

    excess = (strategy_returns - benchmark_returns).dropna()
    wins, losses = excess[excess > 0], excess[excess < 0]
    
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    
    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

def bull_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = strat[bench > 0] - bench[bench > 0]
    
    wins, losses = bull_excess[bull_excess > 0], bull_excess[bull_excess < 0]
    
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    
    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

def bear_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = strat[bench <= 0] - bench[bench <= 0]
    
    wins, losses = bear_excess[bear_excess > 0], bear_excess[bear_excess < 0]
    
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    
    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

# ==========================================
# ACTIVE RETURN
# ==========================================

def active_return(
    strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Computes the annualized arithmetic mean of excess returns.
    """
    # Handle DataFrame recursion
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: active_return(col, benchmark_returns, periods_per_year))

    # Align data to avoid mismatched date errors
    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()
    
    if excess.empty: 
        return np.nan
        
    # Annualize the mean excess return
    return float(excess.mean() * periods_per_year)


# ==========================================
# TRACKING ERROR (Active Risk)
# ==========================================

def tracking_error(
    strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Computes the annualized standard deviation of excess returns.
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: tracking_error(col, benchmark_returns, periods_per_year))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()
    
    # Require at least 2 data points for standard deviation
    if len(excess) < 2: 
        return np.nan
        
    # Calculate sample standard deviation (ddof=1) and annualize it
    return float(excess.std(ddof=1) * np.sqrt(periods_per_year))


# ==========================================
# INFORMATION RATIO
# ==========================================

def information_ratio(
    strategy_returns: Union[pd.Series, pd.DataFrame], 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> Union[float, pd.Series]:
    """
    Computes the annualized Information Ratio (Active Return / Tracking Error).
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: information_ratio(col, benchmark_returns, periods_per_year))

    # We can just call our other two functions directly
    ann_active_return = active_return(strategy_returns, benchmark_returns, periods_per_year)
    ann_tracking_error = tracking_error(strategy_returns, benchmark_returns, periods_per_year)
    
    # Catch division by zero if the strategy perfectly mimics the benchmark
    if pd.isna(ann_tracking_error) or ann_tracking_error == 0:
        return np.nan
        
    return float(ann_active_return / ann_tracking_error)

# ==========================================
# EVALUATE INVESTMENTS CONSISTENCY
# ==========================================

def evaluate_consistency(
    strategy_returns: pd.DataFrame | pd.Series, 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    formatted: bool = False
) -> pd.DataFrame:
    """
    Evaluates the relative consistency and active risk of one or multiple strategies.
    
    Parameters
    ----------
    strategy_returns : pd.Series or pd.DataFrame
        Periodic returns for the strategy/strategies.
    benchmark_returns : pd.Series
        Periodic returns for the benchmark.
    periods_per_year : int
        Annualization factor (252 for daily, 12 for monthly).
    formatted : bool
        If True, returns strings with %, x, and decimals. If False, returns raw floats.
        
    Returns
    -------
    pd.DataFrame
        A tearsheet with strategies as rows and consistency/risk metrics as columns.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns

    results = {}
    
    # Calculate metrics, ordering them exactly how they should appear left-to-right
    for col in strat_df.columns:
        strat = strat_df[col]
        
        results[col] = {
            "Batting Average (Overall)": batting_average(strat, benchmark_returns),
            "Batting Average (Bull)": bull_batting_average(strat, benchmark_returns),
            "Batting Average (Bear)": bear_batting_average(strat, benchmark_returns),
            "Win/Loss Ratio (Overall)": win_loss_ratio(strat, benchmark_returns),
            "Win/Loss Ratio (Bull)": bull_win_loss_ratio(strat, benchmark_returns),
            "Win/Loss Ratio (Bear)": bear_win_loss_ratio(strat, benchmark_returns),
            # Placed at the end per your request
            "Active Return (Ann.)": active_return(strat, benchmark_returns, periods_per_year),
            "Tracking Error (Ann.)": tracking_error(strat, benchmark_returns, periods_per_year),
            "Information Ratio": information_ratio(strat, benchmark_returns, periods_per_year)
        }
        
    # Convert dictionary to DataFrame and transpose (.T) so strategies are rows
    tearsheet = pd.DataFrame(results).T
    
    # Apply string formatting for presentation if requested
    if formatted:
        format_rules = {
            "Batting Average (Overall)": "{:.2%}",
            "Batting Average (Bull)": "{:.2%}",
            "Batting Average (Bear)": "{:.2%}",
            "Win/Loss Ratio (Overall)": "{:.2f}x",
            "Win/Loss Ratio (Bull)": "{:.2f}x",
            "Win/Loss Ratio (Bear)": "{:.2f}x",
            "Active Return (Ann.)": "{:.2%}",
            "Tracking Error (Ann.)": "{:.2%}",
            "Information Ratio": "{:.2f}"
        }
        for metric, fmt in format_rules.items():
            if metric in tearsheet.columns:
                tearsheet[metric] = tearsheet[metric].apply(
                    lambda x: fmt.format(x) if pd.notnull(x) else "N/A"
                )
                
    return tearsheet


def down_capture(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Down Capture Ratio using the Geometric Mean method.
    
    Args:
        returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.
        
    Returns:
        pd.Series: The ratios (e.g., 0.95 = 95%) for each strategy. Returns np.nan if no down markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()
    
    # 2. Filter for periods where Benchmark was strictly DOWN (< 0)
    #    Standard practice is < 0, but some use <= 0.
    down_market = df[df['bench'] < 0]
    
    # 3. Handle edge case: No down markets
    if len(down_market) == 0:
        return pd.Series(np.nan, index=returns.columns)

    n = len(down_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for down periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product down each column simultaneously.
    
    port_geo_avg = (np.prod(1 + down_market[returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + down_market['bench'])) ** (1 / n) - 1
    
    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter < 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=returns.columns)
        
    ratio = port_geo_avg / bench_geo_avg
    
    return ratio

def up_capture(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Up Capture Ratio using the Geometric Mean method.
    
    Args:
        returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.
        
    Returns:
        pd.Series: The ratios (e.g., 1.05 = 105%) for each strategy. Returns np.nan if no up markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()
    
    # 2. Filter for periods where Benchmark was strictly UP (> 0)
    #    Standard practice is > 0, but some use >= 0.
    up_market = df[df['bench'] > 0]
    
    # 3. Handle edge case: No up markets
    if len(up_market) == 0:
        return pd.Series(np.nan, index=returns.columns)

    n = len(up_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for up periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product up each column simultaneously.
    
    port_geo_avg = (np.prod(1 + up_market[returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + up_market['bench'])) ** (1 / n) - 1
    
    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter > 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=returns.columns)
        
    ratio = port_geo_avg / bench_geo_avg
    
    return ratio

def capture_ratios(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Calculates Up Capture, Down Capture, and Capture Spread for multiple strategies.
    
    Args:
        returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark_returns (pd.Series): Periodic returns of the benchmark.
        
    Returns:
        pd.DataFrame: A summary table with strategies as rows and capture metrics as columns.
    """
    
    # 1. Calculate ratios using our vectorized functions
    # (These functions already handle the date alignment and dropna internally!)
    up = up_capture(returns, benchmark_returns)
    down = down_capture(returns, benchmark_returns)
    
    # 2. Combine into a clean summary DataFrame
    summary_df = pd.DataFrame({
        "Up Capture": up,
        "Down Capture": down
    })
    
    # 3. Add Capture Spread (Up Capture - Down Capture)
    # A positive spread indicates the manager adds value across full market cycles.
    summary_df["Capture Spread"] = summary_df["Up Capture"] - summary_df["Down Capture"]
    
    # 4. Add Overall Capture Ratio (Up Capture / Down Capture)
    # A ratio > 1.0 generally implies a favorable asymmetric return profile.
    summary_df["Overall Capture"] = summary_df["Up Capture"] / summary_df["Down Capture"]
    
    return summary_df

def batting_averages(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Calculates Overall, Up Market, and Down Market Batting Averages for multiple strategies.
    
    IMPORTANT: All metrics are computed on the COMMON overlapping period only
    (i.e., dates where EVERY strategy AND the benchmark have valid returns).
    This guarantees perfect comparability across strategies.
    
    Batting Average is the percentage of periods where the strategy outperformed the benchmark,
    expressed as a decimal (e.g., 0.65 for 65%). Up/Down markets exclude periods where benchmark == 0.
    
    Args:
        returns (pd.DataFrame): Periodic returns of the strategies (columns = strategy names).
        benchmark (pd.Series): Periodic returns of the benchmark.
        
    Returns:
        pd.DataFrame: Summary table with the three batting average metrics per strategy.
    """
    # 1. Input validation
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    # 2. Align on index + keep only common valid dates
    data = returns.copy()
    data["bench_"] = benchmark                    # pandas aligns automatically
    data = data.dropna(how="any")                 # drop row if ANY NaN present

    if data.empty:
        # No overlapping data at all
        return pd.DataFrame(
            {
                "Batting Average": np.nan,
                "Up Market Batting Avg": np.nan,
                "Down Market Batting Avg": np.nan,
            },
            index=returns.columns,
        )

    aligned_returns = data[returns.columns]       # or data.drop(columns=["bench_"])
    aligned_bench = data["bench_"]

    # 3. Overall Batting Average: % of time Strategy > Benchmark
    overall_avg = aligned_returns.gt(aligned_bench, axis=0).mean()

    # 4. Up Market Batting Average: % of time Strategy > Benchmark (when Bench > 0)
    up_mask = aligned_bench > 0
    if up_mask.sum() > 0:
        up_avg = aligned_returns.loc[up_mask].gt(aligned_bench.loc[up_mask], axis=0).mean()
    else:
        up_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 5. Down Market Batting Average: % of time Strategy > Benchmark (when Bench < 0)
    down_mask = aligned_bench < 0
    if down_mask.sum() > 0:
        down_avg = aligned_returns.loc[down_mask].gt(aligned_bench.loc[down_mask], axis=0).mean()
    else:
        down_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 6. Final table
    batting_df = pd.DataFrame(
        {
            "Batting Average": overall_avg,
            "Up Market Batting Avg": up_avg,
            "Down Market Batting Avg": down_avg,
        }
    )

    return batting_df


# ============== PERFORMANCE METRICS ============== #

def cumgrowth(returns: pd.DataFrame, init_value: float = 1.0) -> pd.DataFrame:
    """Compute cumulative growth from a DataFrame of periodic returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic returns in decimal form with a DatetimeIndex.
    init_value : float, optional
        Starting value for the growth series. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        Cumulative growth values, prepended with an initial-value row
        if the index frequency can be inferred.
    """
    cumulative_growth = (returns + 1).cumprod() * init_value

    dt_freq = returns.index.inferred_freq

    if dt_freq:
        init_dt = returns.index[0] - pd.tseries.frequencies.to_offset(dt_freq)
        init_row = pd.DataFrame(init_value, index=[init_dt], columns=returns.columns)
        cumulative_growth = pd.concat([init_row, cumulative_growth]).sort_index()
    else:
        
        warnings.warn(
            "Could not infer index frequency — initial value row not prepended.",
            stacklevel=2,
        )

    return cumulative_growth

def compute_forward_returns(returns: pd.DataFrame, forward_periods: int) -> pd.DataFrame:
    """
    Compute cumulative forward returns over a specified horizon for each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of simple returns (not log returns), indexed by date.
    forward_periods : int
        Number of periods ahead to compute the forward return.

    Returns
    -------
    pd.DataFrame
        DataFrame of forward returns, aligned with the original index.
    """
    # Compute cumulative product of (1 + returns)
    cumulative_growth = (returns + 1).cumprod()
    # Compute cumulative forward returns: (future value / current value) - 1
    forward_returns = (cumulative_growth.shift(-forward_periods) / cumulative_growth) - 1

    return forward_returns

# ============== ROLLING METRICS ================== #

def rolling_beta(returns: pd.DataFrame, benchmark: pd.Series, window: int = 12) -> pd.DataFrame:
    """
    Calculates rolling beta using native vectorized pandas methods.
    Mathematically identical to Cov(R, B) / Var(B) with ddof=1.
    """
    # 1. Align data
    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()
    aligned_returns = combined[returns.columns]
    aligned_bench = combined["__benchmark__"]
    
    # 2. Vectorized rolling covariance and variance
    # pandas smartly broadcasts the benchmark Series against every column in the DataFrame
    rolling_cov = aligned_returns.rolling(window=window).cov(aligned_bench)
    rolling_var = aligned_bench.rolling(window=window).var()
    
    # 3. Divide to get rolling beta (div(axis=0) ensures correct date alignment)
    rolling_betas = rolling_cov.div(rolling_var, axis=0)
    rolling_betas[rolling_var == 0] = np.nan  # guard flat-benchmark windows
    
    return rolling_betas




__all__ = ['return_n', "return_ytd", "ann_return", 'ann_return_common_si', 'performance_table',
           'cumgrowth', 'compute_cumulative_growth',
           'compute_forward_returns', 'compute_capm',
           'sortino_ratio', 'beta', 'bull_bear_beta', 'rolling_beta', 'downside_variance',
           'down_capture', 'up_capture', "capture_ratios",
           'batting_averages', 
           'batting_average', 'bull_batting_average', 'bear_batting_average',
           'win_loss_ratio', 'bull_win_loss_ratio', 'bear_win_loss_ratio', 
           'active_return', 'tracking_error', 'information_ratio',
           'evaluate_consistency']