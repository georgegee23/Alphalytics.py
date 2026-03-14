
import pandas as pd
import numpy as np


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
