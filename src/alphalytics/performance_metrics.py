
import pandas as pd
import numpy as np
import quantstats as qs



# ============== PERFORMANCE METRICS ============== #

def compute_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert returns to prices using cumulative product method.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns with dates as index and assets as columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame of prices, with the first date set to 1.0 and subsequent
        prices computed via cumulative returns
        
    Notes
    -----
    - Assumes returns are in decimal form (e.g., 0.01 for 1% return)
    - First date is set to NaN and then filled with 1.0 to anchor the price series
    """
    # Input validation
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    
    # Compute cumulative returns
    prices = (returns + 1).cumprod()
    
    # Set first date and sort
    first_date = returns.index.shift(-1)[0]
    prices.loc[first_date, :] = np.nan
    prices = prices.sort_index()
    
    # Fill first NaN with 1.0 for each column
    prices = prices.apply(lambda col: fill_first_nan(col, value=1.0))
    
    return prices


def compute_performance_table(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
  
    idxs = {
        '3M': int(periods_per_year/4),
        '6M': int(periods_per_year/2),
        '1Y': periods_per_year,
        '3Y': periods_per_year*3,
        '5Y': periods_per_year*5,
        '10Y': periods_per_year*10
    }
    
    qtr_idx = int(periods_per_year/4)
    semiannual_idx = int(periods_per_year/2)
    year1_idx = int(periods_per_year)
    year3_idx = int(periods_per_year*3)
    year5_idx = int(periods_per_year*5)
    year10_idx = int(periods_per_year*10)
    
    qtr_ret = qtr_ret = qs.stats.comp(returns.iloc[-qtr_idx-1:-1]) 
    semiannual_ret = qs.stats.comp(returns.iloc[-semiannual_idx-1:-1]) 
    year1_ret = qs.stats.cagr(returns.iloc[-year1_idx-1:-1]) 
    year3_ret = qs.stats.cagr(returns.iloc[-year3_idx-1:-1]) 
    year5_ret = qs.stats.cagr(returns.iloc[-year5_idx-1:-1]) 
    year10_ret = qs.stats.cagr(returns.iloc[-year10_idx-1:-1]) 
    si_ret = qs.stats.cagr(returns) 

    performance_table = pd.concat([qtr_ret, semiannual_ret, year1_ret, year3_ret, year5_ret, year10_ret, si_ret], axis = 1)
    performance_table.columns = ["3M", "6M", "1-Year", "3-Year", "5-Year", "10-Year", "SI"]

    return performance_table

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


 __all__ = [
    "compute_prices", 
    "compute_performance_table", "compute_cumulative_growth",
    "compute_forward_returns", "compute_capm"
 ]