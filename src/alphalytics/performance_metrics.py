
import pandas as pd
import numpy as np
import quantstats as qs


# ============== PERFORMANCE METRICS ============== #

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

def compute_forward_returns(returns:pd.DataFrame, forward_periods:int) -> pd.DataFrame:

    # Compute cumulative growth and forward returns
    cumulative_growth = (returns + 1).cumprod()
    # Compute forward returns
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