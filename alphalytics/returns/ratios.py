
import pandas as pd
import numpy as np
from typing import Union

from .risk import downside_variance, max_drawdown
from alphalytics.utils import _infer_periods_per_year


# ============== RATIO STATISTICS ============== #

# ==========================================
# SHARPE RATIO
# ==========================================

def sharpe_ratio(returns: Union[pd.Series, pd.DataFrame], rfr: float = 0.0,
    periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Annualised Sharpe ratio.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        rfr: Risk-free rate per period. Defaults to 0.0.
        periods_per_year: Annualisation factor (inferred from index if None).

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    excess = returns - rfr
    std = excess.std()
    if isinstance(std, pd.Series):
        std = std.where(~np.isclose(std, 0, atol=1e-12), np.nan)
    elif std == 0 or np.isclose(std, 0, atol=1e-12):
        return np.nan
    return (excess.mean() / std) * np.sqrt(periods_per_year)

# ==========================================
# SORTINO RATIO
# ==========================================

def sortino_ratio(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    periods_per_year: int = None, ddof: int = 1) -> Union[float, pd.Series]:
    """
    Annualized Sortino Ratio.

    Improves on the Sharpe Ratio by penalizing only downside volatility,
    making it more appropriate for strategies with asymmetric return distributions.

    Annualization: mean excess return scaled by periods_per_year,
    downside deviation scaled by sqrt(periods_per_year).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        periods_per_year (int): Annualization factor. Inferred from index if None.
        ddof (int): Degrees of freedom for downside variance. Defaults to 1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when downside deviation is zero or undefined.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    down_var = downside_variance(returns, mar=mar, ddof=ddof)
    if isinstance(down_var, pd.Series):
        down_dev = np.sqrt(down_var).replace(0, np.nan)
    else:
        down_dev = np.sqrt(down_var)
        if down_dev == 0:
            down_dev = np.nan

    mean_excess_return = returns.mean() - mar

    # Calculate the ratio and annualize
    # Note: We scale the final ratio by the square root of periods_per_year
    sortino = (mean_excess_return / down_dev) * np.sqrt(periods_per_year)

    return sortino.squeeze()

# ==========================================
# CALMAR RATIO
# ==========================================

def calmar_ratio(returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = None) -> Union[float, pd.Series]:

    """
    Calmar ratio — annualised return divided by maximum drawdown.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        periods_per_year: Annualisation factor (inferred from index if None).

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
    """

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    mdd = np.abs(max_drawdown(returns))

    if isinstance(mdd, pd.Series):
        mdd = mdd.replace(0, np.nan)
    elif mdd == 0:
        return np.nan

    return ann_ret / mdd

# ==========================================
# OMEGA RATIO
# ==========================================

def omega_ratio(returns: Union[pd.Series, pd.DataFrame],
    mar: float = 0.0) -> Union[float, pd.Series]:

    """
    Omega ratio — probability-weighted gains over losses relative to a threshold.

    Ratio of the sum of returns above the MAR to the sum of returns below it.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar: Minimum Acceptable Return threshold per period. Defaults to 0.0.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN if there are no returns below the threshold.
    """

    excess = returns - mar
    gains = excess[excess > 0].sum()
    losses = np.abs(excess[excess < 0].sum())

    if isinstance(losses, pd.Series):
        return gains / losses.replace(0, np.nan)
    else:
        return gains / losses if losses != 0 else np.nan
