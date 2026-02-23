
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

def compute_capm(returns: pd.DataFrame, benchmark: pd.Series = None) -> pd.DataFrame:

    # Validate inputs
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("quantile_returns must be a pandas DataFrame")
        
    if returns.empty:
        raise ValueError("quantile_returns cannot be empty")

    # Set default benchmark if not provided
    if benchmark is None:
        benchmark = returns.mean(axis=1)
        benchmark.name = "Equal-Weighted Benchmark"
    elif not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series")
        
    # Ensure matching indices
    if not returns.index.equals(benchmark.index):
        raise ValueError("quantile_returns and benchmark must have matching indices")

    # Calculate CAPM metrics for each quantile
    capm_results = []
    
    for col in returns.columns:
        try:
            beta, alpha = qs.stats.greeks(
                returns=returns[col],
                benchmark=benchmark
            )
            capm_results.append({
                "Asset": col,
                "Beta": beta,
                "Alpha": alpha
            })
        except Exception as e:
            raise RuntimeError(f"Error calculating CAPM for {col}: {str(e)}")

    # Create formatted DataFrame
    capm_df = pd.DataFrame(capm_results).set_index("Asset")
    
    return capm_df


 # ============== THE END ============== #     

def down_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Down Capture Ratio using the Geometric Mean method.
    
    Args:
        portfolio_returns (pd.Series): Periodic returns (e.g., monthly) of the strategy.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.
        
    Returns:
        float: The ratio (e.g., 0.95 = 95%). Returns np.nan if no down markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = pd.DataFrame({
        'port': portfolio_returns, 
        'bench': benchmark_returns
    }).dropna()
    
    # 2. Filter for periods where Benchmark was strictly DOWN (< 0)
    #    Standard practice is < 0, but some use <= 0.
    down_market = df[df['bench'] < 0]
    
    # 3. Handle edge case: No down markets
    if len(down_market) == 0:
        return np.nan

    n = len(down_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for down periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: This accurately captures the compounding pain of losses.
    
    port_geo_avg = (np.prod(1 + down_market['port'])) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + down_market['bench'])) ** (1 / n) - 1
    
    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter < 0)
    if bench_geo_avg == 0:
        return np.nan
        
    ratio = port_geo_avg / bench_geo_avg
    
    return ratio

def up_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Up Capture Ratio (Geometric Mean method).
    
    Args:
        portfolio_returns (pd.Series): Periodic returns of the strategy.
        benchmark_returns (pd.Series): Periodic returns of the benchmark.
        
    Returns:
        float: The ratio (e.g., 1.10 = 110%). Returns np.nan if no up markets occur.
    """
    # 1. Align data securely
    df = pd.DataFrame({'port': portfolio_returns, 'bench': benchmark_returns}).dropna()
    
    # 2. Filter for Up markets
    up_market = df[df['bench'] > 0]
    n = len(up_market)
    
    if n == 0:
        return np.nan

    # 3. Calculate Geometric Mean using native Pandas .prod()
    port_geo_avg = (1 + up_market['port']).prod() ** (1 / n) - 1
    bench_geo_avg = (1 + up_market['bench']).prod() ** (1 / n) - 1
    
    # 4. Calculate Ratio (multiplied by 100 for percentage scale)
    if bench_geo_avg == 0:
        return np.nan
        
    return port_geo_avg / bench_geo_avg

def capture_ratios(strategy_returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """Calculates Up and Down capture ratios for multiple strategies."""
    
    # Align data to drop mismatched dates
    data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    
    # The benchmark is always the last column after concat
    bench = data.iloc[:, -1]
    
    capture_dict = {}
    for col in data.columns[:-1]:
        strat = data[col]
        capture_dict[col] = {
            "Up Capture": up_capture(strat, bench),
            "Down Capture": down_capture(strat, bench)
        }
        
    # from_dict with orient='index' avoids the need to transpose (.T)
    return pd.DataFrame.from_dict(capture_dict, orient='index')

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