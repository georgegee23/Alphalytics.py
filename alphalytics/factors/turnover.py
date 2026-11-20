
import pandas as pd
import numpy as np


# ============== TURNOVER ANALYSIS ============== #

def calculate_autocorrelation(factor_ts, lag=1):
    """
    Calculate the autocorrelation for each column in the DataFrame at a specified lag.

    The lag is applied on the original index, so a missing print stays a gap
    instead of closing up. A column with ``lag + 1`` or fewer finite values
    returns NaN.

    Parameters:
    factor_ts (pd.DataFrame): The input DataFrame with time series data in columns, typically factor scores.
    lag (int): The lag at which to calculate the autocorrelation. Must be a positive integer. Default is 1.

    Returns:
    pd.Series: A Series containing autocorrelation values for each column, indexed by column names.
              Returns NaN for columns with insufficient data (length <= lag + 1 after dropping NaNs).
    """
    autocorrelations = {}
    for column in factor_ts.columns:
        series = factor_ts[column]
        # Lag on the original index. Dropping NaNs first would turn a
        # calendar lag into a lag across the previous valid print.
        if series.dropna().shape[0] <= lag + 1:
            autocorr = np.nan
        else:
            autocorr = series.autocorr(lag=lag)
        autocorrelations[column] = autocorr
    return pd.Series(autocorrelations)

def compute_factor_autocorr(factor_ts: pd.DataFrame, max_lag: int):
    """
    Compute the average autocorrelation across all columns for a range of lags.

    This function calculates the mean autocorrelation across all columns of the input DataFrame
    for lags ranging from 1 to max_lag - 1. It uses `calculate_autocorrelation` to compute per-column
    autocorrelations and averages them for each lag. The result is a Series indexed by lag values.

    Parameters:
    factor_ts (pd.DataFrame): The input DataFrame with factor time series data in columns.
    max_lag (int): The upper bound (exclusive) of the lag range for autocorrelation computation.

    Returns:
    pd.Series: A Series with lags as indices (1 to max_lag - 1) and average autocorrelations as values.
              The mean is computed over non-NaN autocorrelations for each lag.
    """
    lags_range = range(1, max_lag)
    autocorr_lag_dict = {}
    for lag in lags_range:
        autocorr_lag_dict[lag] = calculate_autocorrelation(factor_ts, lag=lag).mean()
    autocorr_series = pd.Series(autocorr_lag_dict)
    return autocorr_series


def compute_quantile_turnover(quantiles:pd.DataFrame, target_quantile) -> pd.Series:

    """
    Turnover of one quantile from the previous date to this one.

    Turnover is ``1 - |intersection| / |union|`` of the names in the
    quantile on the two dates, so names that enter count as well as names
    that leave. The first date is NaN. An empty union is NaN.

    Parameters:
    - quantiles: DataFrame where each row is a time period, each column is an asset,
                and values represent the quantile assignments
    - target_quantile: The specific quantile to calculate turnover for

    Returns:
    - pd.Series with turnover rates for each time period
    """


    quantiles = quantiles.dropna(how="all")
    # 1 where the asset is in the target quantile. NaN membership is not a holding.
    held = (quantiles == target_quantile).astype(float)
    previous = held.shift(1)

    held_now = held.fillna(0.0) > 0
    held_prev = previous.fillna(0.0) > 0
    # The first date has no prior book. An empty union (nobody held on
    # either date) is undefined, not zero turnover.
    has_history = previous.notna().any(axis=1)
    union = (held_now | held_prev).sum(axis=1).astype(float)
    intersection = (held_now & held_prev).sum(axis=1).astype(float)
    turnover = 1 - intersection / union.where(union > 0)
    return turnover.where(has_history)


def compute_quantiles_turnover(quantiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute turnover rates for all quantiles in a DataFrame, proportion of holdings that changed quantiles.

    Parameters:
    - quantiles: DataFrame where each row is a time period, columns are assets,
                values represent quantile assignments (typically integers)

    Returns:
    - DataFrame with turnover rates for each quantile over time
    """
    # Get actual unique quantiles present in the data
    quantile_values = pd.unique(quantiles.values.ravel())
    quantile_list = sorted([q for q in quantile_values if pd.notnull(q)])

    turnover_dict = {}

    for q in quantile_list:
        try:
            # Calculate turnover for each quantile with error handling
            turnover_series = compute_quantile_turnover(quantiles, target_quantile=q)

            # Handle empty results and division by zero cases
            if not turnover_series.empty:
                turnover_dict[f"Q{int(q)}"] = turnover_series.replace([np.inf, -np.inf], np.nan)

        except ZeroDivisionError:
            print(f"Warning: No holdings in quantile {q} for some periods")
            turnover_dict[f"Q{int(q)}"] = pd.Series(index=quantiles.index, data=np.nan)

    # Create DataFrame and align all series by index
    quantiles_turnover_df = pd.DataFrame(turnover_dict).reindex(quantiles.index).dropna(how = "all")
    quantiles_turnover_df.columns = turnover_dict.keys()

    return quantiles_turnover_df
