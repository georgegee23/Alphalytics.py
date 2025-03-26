
import pandas as pd
import numpy as np
from scipy.stats import norm, stats, probplot, ttest_1samp


# ============== QUANTILE PERFORMANCE ANALYSIS ============== #

def compute_series_quantiles(series: pd.Series, n_quantiles: int) -> pd.Series:
    """
    Divide non-NaN values in a series into q quantiles.
    
    Parameters:
    -----------
    series : pd.Series
        Input data to be divided into quantiles
    q : int
        Number of quantiles to compute
        
    Returns:
    --------
    pd.Series
        Series with same index as input, where each value is replaced
        with its quantile rank (1 to q), and NaN values preserved
    """
    mask = series.notna()  # Identify non-NaN positions
    if mask.any():  # If there are any non-NaN values
        try:
            # Compute quantiles on non-NaN values
            ranks = pd.qcut(series[mask], q=n_quantiles, labels=False, duplicates="drop")
            ranks += 1  # Adjust to start from 1 if desired

            # Create result with NaNs, then fill in ranks
            result = pd.Series(index=series.index, data=np.nan)
            result[mask] = ranks
        except ValueError:
            # Handle case where all non-NaN values are identical
            default_rank = 1
            result = pd.Series(index=series.index, data=np.nan)
            result[mask] = default_rank
    else:
        # All values are NaN
        result = pd.Series(index=series.index, data=np.nan)
    return result

def to_quantiles(factors: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """
    Convert a DataFrame of continuous factor values to quantile rankings.
    
    Parameters:
    -----------
    factors : pd.DataFrame
        DataFrame where each row contains factor values to be ranked
    
    n_quantiles : int
        Number of quantile groups to divide values into
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with same shape as input, containing quantile rankings
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if n_quantiles <= 0 or not isinstance(n_quantiles, int):
        raise ValueError("n_quantiles must be a positive integer")
        
    # Create empty DataFrame to store results
    quantiles_df = pd.DataFrame(
        index=factors.index,
        columns=factors.columns,
        dtype=float
    )
    
    # Process each row (e.g., time period)
    for idx, row in factors.iterrows():
        quantiles_df.loc[idx] = compute_series_quantiles(row, n_quantiles)
    
    return quantiles_df

def compute_quantile_returns(quantiles: pd.DataFrame, returns: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Calculate the average returns for each quantile based on lagged quantile assignments.

    Parameters:
    -----------
    quantiles : pd.DataFrame
        DataFrame with quantile assignments (integers starting from 1).
    returns : pd.DataFrame
        DataFrame with returns, same shape as quantiles.
    lag : int
        Number of periods to lag the quantile assignments.

    Returns:
    --------
    pd.DataFrame
        DataFrame with average returns for each quantile, with columns "Q1", "Q2", etc.
    """
    # Input validation
    if not isinstance(quantiles, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise ValueError("quantiles and returns must be pandas DataFrames")
    if not quantiles.index.equals(returns.index) or not quantiles.columns.equals(returns.columns):
        raise ValueError("quantiles and returns must have the same index and columns")
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("lag must be a non-negative integer")

    # Shift quantiles by the specified lag
    quantiles_lagged = quantiles.shift(lag)

    # Stack the DataFrames into Series
    quantiles_stacked = quantiles_lagged.stack()
    returns_stacked = returns.stack()

    # Create a DataFrame with quantile and return columns
    data = pd.DataFrame({'quantile': quantiles_stacked, 'return': returns_stacked})

    # Group by time (level 0) and quantile, compute mean returns, and unstack into a DataFrame
    quantile_returns = data.groupby([data.index.get_level_values(0), 'quantile'])['return'].mean().unstack()

    # Rename columns to "Q1", "Q2", etc.
    quantile_returns.columns = [f"Q{int(col)}" for col in quantile_returns.columns]

    # Drop rows where all values are NaN (e.g., initial lagged periods)
    return quantile_returns.dropna(how="all")


# ============== QUANTILE HORIZON ANALYSIS ============== #  

def get_quantile_holdings(quantiles, target_quantile) -> pd.Series:

    """Extracts holdings (column names) matching a target quantile for each row in a DataFrame.

    Args:
        quantiles_df (pd.DataFrame): A DataFrame where rows represent time periods (or entities) 
            and columns represent holdings (e.g., assets), with values indicating quantiles.
        target_quantile (int or float): The quantile value to match (e.g., 1, 2, 3).

    Returns:
        pd.Series: A Series with the same index as the filtered quantiles_df (after dropping 
            rows with all NaN), where each value is a list of column names (holdings) 
            whose quantile equals the target_quantile in that row.

    Example:
        >>> df = pd.DataFrame({
        ...     'A': [1, 3, np.nan],
        ...     'B': [2, 1, np.nan],
        ...     'C': [np.nan, 1, np.nan]
        ... }, index=[0, 1, 2])
        >>> get_quantile_holdings_ts(df, 1)
        0      ['A']
        1    ['B', 'C']
        dtype: object
    """

    quantiles = quantiles.dropna(how="all")
    q_mask = (quantiles == target_quantile)

    return q_mask.apply(lambda row: list(q_mask.columns[row]), axis = 1)

def compute_mean_quantile_forward_return(returns: pd.DataFrame, quantile_holdings: pd.Series, forward_periods: int) -> pd.DataFrame:
    """
    Compute the mean forward return of assets in a quantile over a specified number of periods.
    
    Parameters:
    - returns: DataFrame of periodic returns (dates x assets).
    - quantile_holdings: Series with dates as index and lists of asset names as values.
    - forward_periods: Number of periods to look forward.
    
    Returns:
    - DataFrame with dates and mean forward returns.
    """
    # Compute cumulative growth and forward returns
    cumulative_growth = (returns + 1).cumprod()
    # Compute forward returns
    forward_returns = (cumulative_growth.shift(-forward_periods) / cumulative_growth) - 1 

    # For each date, get the mean forward return of selected holdings
    mean_q_dict = {}
    for date in quantile_holdings.index:
        if date in forward_returns.index:
            selected_holdings = quantile_holdings.loc[date]
            if selected_holdings:
                mean_q_dict[date] = forward_returns.loc[date, selected_holdings].mean()
            else:
                mean_q_dict[date] = np.nan
    
    return pd.DataFrame.from_dict(mean_q_dict, orient='index', columns=['Mean'])

def fwd_quantile_stats(returns: pd.DataFrame, quantiles: pd.DataFrame, forward_periods: int) -> pd.DataFrame:
    """
    Compute forward returns for each quantile.
    
    Parameters:
    - returns: DataFrame of periodic returns (dates x assets).
    - quantiles: DataFrame where values indicate quantiles of assets.
    - periods: Number of periods to look forward.
    - timedelta_unit: Time unit for forward periods (e.g., "W" for weeks).
    
    Returns:
    - DataFrame mapping quantile labels to their mean forward returns and risk-adjusted forward returns.
    """
    quantiles_list = range(1, quantiles.nunique().max() + 1, 1)  # Added +1 to include max quantile

    quantile_fwd_rets_dict = {}
    for q in quantiles_list:
        q_holdings = get_quantile_holdings(quantiles, q)  # Fixed function name
        mean_fwd_rets = compute_mean_quantile_forward_return(returns, q_holdings, forward_periods=forward_periods)
        
        if not mean_fwd_rets.empty:
            fwd_std = mean_fwd_rets.std().values[0]
            quantile_fwd_rets_dict[f"Q{q}"] = [mean_fwd_rets.mean().values[0], (mean_fwd_rets.mean().values[0] / fwd_std)]
        else:
            quantile_fwd_rets_dict[f"Q{q}"] = np.nan

    fwd_quantile_df = pd.DataFrame.from_dict(quantile_fwd_rets_dict, orient="index", columns = ["Return", "Risk-Adjusted Return"])
    return fwd_quantile_df




 # ============== THE END ============== #     