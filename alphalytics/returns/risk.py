
import pandas as pd
import numpy as np

from alphalytics.utils import _infer_periods_per_year


# ============== RISK & RETURN METRICS ============== #

def annual_std(returns: pd.DataFrame, periods_per_year: int = None, ddof: int = 1):
    """
    Calculates the annualized standard deviation of returns.
    
    Parameters:
    returns (pd.DataFrame): Asset returns.
    periods_per_year (int, optional): Trading periods in a year. Inferred if None.
    ddof (int): Delta Degrees of Freedom. Defaults to 1 for sample standard deviation.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)
        
    # Explicitly pass ddof to the pandas .std() method
    return returns.std(ddof=ddof) * np.sqrt(periods_per_year)

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

