
import pandas as pd
import numpy as np
from typing import Union

from .risk import downside_variance, max_drawdown
from .utils import _infer_periods_per_year


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
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

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

    down_dev = np.sqrt(downside_variance(returns, mar=mar, ddof=ddof))
    down_dev = down_dev.replace(0, np.nan)  # guard: strategy never fell below MAR

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
    mdd = max_drawdown(returns).abs()

    return ann_ret / mdd.replace(0, np.nan)

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
    losses = excess[excess < 0].sum().abs()

    if isinstance(losses, pd.Series):
        return gains / losses.replace(0, np.nan)
    else:
        return gains / losses if losses != 0 else np.nan
