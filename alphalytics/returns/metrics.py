
import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings

from alphalytics.utils import _infer_periods_per_year


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



def cumgrowth(returns: pd.DataFrame, init_value: float = 1.0) -> pd.DataFrame:
    """Compute cumulative growth from a DataFrame of periodic returns.

    Args:
        returns: Periodic returns in decimal form with a DatetimeIndex.
        init_value: Starting value for the growth series.

    Returns:
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


def annualized_rolling_return(returns: Union[pd.Series, pd.DataFrame], window: int,
    periods_per_year: Optional[int] = None) -> Union[pd.Series, pd.DataFrame]:

    """
    Calculates the annualized rolling return for a time series or DataFrame
    of simple periodic returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Simple periodic returns (e.g., 0.01 for 1%). Each column in a
        DataFrame is processed independently.
    window : int
        Rolling window size in periods (e.g., 252 for 1-year on daily data).
    periods_per_year : int, optional
        Periods in a year. Inferred from a DatetimeIndex if not provided.

    Returns
    -------
    pd.Series or pd.DataFrame
        Annualized rolling returns with NaN for the first (window - 1) rows.
        Shape matches the input — NaN rows are NOT dropped.

    Notes
    -----
    Period inference relies on `alphalytics.utils._infer_periods_per_year`.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(returns)}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > len(returns):
        warnings.warn(
            f"window ({window}) exceeds the length of returns ({len(returns)}). "
            "Result will be all NaN."
        )

    if periods_per_year is None:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError(
                "returns must have a DatetimeIndex to infer periods_per_year "
                "automatically. Provide it explicitly instead."
            )
        periods_per_year = _infer_periods_per_year(returns.index)

    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")

    log_returns = np.log1p(returns)
    rolling_log_sum = log_returns.rolling(window=window).sum()
    rolling_cumulative = np.expm1(rolling_log_sum)
    annualization_factor = periods_per_year / window

    return (1 + rolling_cumulative) ** annualization_factor - 1

