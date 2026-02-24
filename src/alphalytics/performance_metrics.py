
import pandas as pd
import numpy as np
import quantstats as qs

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
    assert years >= 1, "Must be at least 1 year to annualize"

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

def compute_capm(returns: pd.DataFrame, benchmark: pd.Series = None, periods_per_year: int = 252) -> pd.DataFrame:
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
        raise TypeError("returns must be a pandas DataFrame")
        
    if returns.empty:
        raise ValueError("returns cannot be empty")

    if benchmark is None:
        benchmark = returns.mean(axis=1)
        benchmark.name = "Benchmark"
    elif not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series")

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


 # ============== THE END ============== #     

def down_capture(portfolio_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Down Capture Ratio using the Geometric Mean method.
    
    Args:
        portfolio_returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.
        
    Returns:
        pd.Series: The ratios (e.g., 0.95 = 95%) for each strategy. Returns np.nan if no down markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = portfolio_returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()
    
    # 2. Filter for periods where Benchmark was strictly DOWN (< 0)
    #    Standard practice is < 0, but some use <= 0.
    down_market = df[df['bench'] < 0]
    
    # 3. Handle edge case: No down markets
    if len(down_market) == 0:
        return pd.Series(np.nan, index=portfolio_returns.columns)

    n = len(down_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for down periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product down each column simultaneously.
    
    port_geo_avg = (np.prod(1 + down_market[portfolio_returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + down_market['bench'])) ** (1 / n) - 1
    
    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter < 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=portfolio_returns.columns)
        
    ratio = port_geo_avg / bench_geo_avg
    
    return ratio

def up_capture(portfolio_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Up Capture Ratio using the Geometric Mean method.
    
    Args:
        portfolio_returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.
        
    Returns:
        pd.Series: The ratios (e.g., 1.05 = 105%) for each strategy. Returns np.nan if no up markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = portfolio_returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()
    
    # 2. Filter for periods where Benchmark was strictly UP (> 0)
    #    Standard practice is > 0, but some use >= 0.
    up_market = df[df['bench'] > 0]
    
    # 3. Handle edge case: No up markets
    if len(up_market) == 0:
        return pd.Series(np.nan, index=portfolio_returns.columns)

    n = len(up_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for up periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product up each column simultaneously.
    
    port_geo_avg = (np.prod(1 + up_market[portfolio_returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + up_market['bench'])) ** (1 / n) - 1
    
    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter > 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=portfolio_returns.columns)
        
    ratio = port_geo_avg / bench_geo_avg
    
    return ratio

def capture_ratios(portfolio_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Calculates Up Capture, Down Capture, and Capture Spread for multiple strategies.
    
    Args:
        portfolio_returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark_returns (pd.Series): Periodic returns of the benchmark.
        
    Returns:
        pd.DataFrame: A summary table with strategies as rows and capture metrics as columns.
    """
    
    # 1. Calculate ratios using our vectorized functions
    # (These functions already handle the date alignment and dropna internally!)
    up = up_capture(portfolio_returns, benchmark_returns)
    down = down_capture(portfolio_returns, benchmark_returns)
    
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
    summary_df["Overall Ratio"] = summary_df["Up Capture"] / summary_df["Down Capture"]
    
    return summary_df

def batting_averages(returns:pd.Series, benchmark:pd.Series):
    """
    Calculates Overall, Up Market, and Down Market Batting Averages.
    
    Parameters:
    - returns (pd.Series): The strategy returns.
    - benchmark (pd.Series): The benchmark returns.
    
    Returns:
    - pd.Series: Containing the three batting average metrics.
    """
    # Ensure inputs are aligned by index (dates) and drop missing data
    data = pd.concat([returns, benchmark], axis=1).dropna()
    r = data.iloc[:, 0]  # Strategy
    b = data.iloc[:, 1]  # Benchmark

    # 1. Overall Batting Average: % of time Strategy > Benchmark
    batting_avg = (r > b).mean()

    # 2. Up Market Batting Average: % of time Strategy > Benchmark (when Benchmark > 0)
    up_market_mask = b > 0
    if up_market_mask.sum() > 0:
        up_batting_avg = (r[up_market_mask] > b[up_market_mask]).mean()
    else:
        up_batting_avg = None # Handle case with no up markets

    # 3. Down Market Batting Average: % of time Strategy > Benchmark (when Benchmark < 0)
    down_market_mask = b < 0
    if down_market_mask.sum() > 0:
        down_batting_avg = (r[down_market_mask] > b[down_market_mask]).mean()
    else:
        down_batting_avg = None # Handle case with no down markets

    return pd.Series({
        "Average": batting_avg,
        "Up Market": up_batting_avg,
        "Down Market": down_batting_avg
    })


# ============== PERFORMANCE METRICS ============== #

def cumgrowth(returns: pd.DataFrame) -> pd.DataFrame:
  
    """
    Compound the returns to compute the cumulative growth factor, assuming a starting value of 1 for each series.

    This function calculates the cumulative product of (1 + returns), which represents the compounded growth
    over time. Useful for converting periodic returns into an equity curve or price level series in finance.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with dates as index and assets as columns, containing periodic return values (e.g., 0.05 for 5%).
        NaN values are preserved and will propagate in the cumulative product, resulting in NaN from the first
        occurrence onward in each column.

    Returns
    -------
    pd.DataFrame
        DataFrame of the same shape, containing the cumulative compounded values starting from 1, with NaNs propagated.

    Notes
    -----
    - Assumes returns are in decimal form.
    - Cumprod is applied along axis=0 (down the rows, time dimension).
    - For cumulative returns (not growth factor), subtract 1 from the result.
    - If you need to start from a different initial value, multiply the result by that value externally.
    - To treat NaNs as 0% return (no change) instead of propagating, use returns.fillna(0) externally before calling.

    Example
    -------
    >>> returns = pd.DataFrame({'Asset1': [0.1, 0.2, np.nan, -0.1]})
    >>> compound(returns)
           Asset1
    0    1.1000
    1    1.3200
    2         NaN
    3         NaN
    """
    
    return returns.add(1).cumprod()

def compute_cumulative_growth(returns: pd.DataFrame, init_value: float = 1.0) -> pd.DataFrame:
    """
    Compute cumulative growth from returns with an initial value.
    
    Args:
        returns (pd.DataFrame): DataFrame of returns in decimal form
        init_value (float): Initial value to start growth from (default: 1.0)
    
    Returns:
        pd.DataFrame: DataFrame with cumulative growth values
    """
    # Get frequency from index
    dt_freq = returns.index.inferred_freq
    
    # Calculate cumulative growth
    cumulative_growth = (returns + 1).cumprod() * init_value
    
    # Create initial row if frequency is valid
    if dt_freq:
        # Get the date before the first date
        init_dt = returns.index[0] - pd.tseries.frequencies.to_offset(dt_freq)
        
        # Create a DataFrame for the initial row with init_value for all columns
        init_row = pd.DataFrame(
            init_value,
            index=[init_dt],
            columns=returns.columns
        )
        
        # Concatenate initial row with cumulative growth
        cumulative_growth = pd.concat([init_row, cumulative_growth]).sort_index()
    
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



__all__ = ['return_n', "return_ytd", "ann_return", 'ann_return_common_si', 'performance_table',
           'cumgrowth',
           'compute_performance_table', 'compute_cumulative_growth',
           'compute_forward_returns', 'compute_capm',
           'down_capture', 'up_capture', "capture_ratios"
           'batting_averages']