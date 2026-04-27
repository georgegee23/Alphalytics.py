
import pandas as pd
import numpy as np
from typing import Union

from .risk import downside_variance, max_drawdown
from .relative import omega_ratio
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
# TAIL RATIO
# ==========================================

def tail_ratio(returns: Union[pd.Series, pd.DataFrame],
    upper: float = 0.9, lower: float = 0.1) -> Union[float, pd.Series]:
    """
    Tail Gain/Loss Ratio (conditional tail expectation ratio).

    Ratio of the average upper-tail return to the absolute average lower-tail
    return:

        Tail G/L = avg(r | r > q_upper) / |avg(r | r < q_lower)|

    A value above 1 indicates the average winning tail exceeds the average
    losing tail in magnitude — a positively skewed payoff profile.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        upper: Upper quantile threshold. Defaults to 0.9.
        lower: Lower quantile threshold. Defaults to 0.1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when either tail is empty or the lower-tail mean is zero.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: tail_ratio(col, upper, lower))

    r = returns.dropna()
    if r.empty: return np.nan

    q_up = r.quantile(upper)
    q_lo = r.quantile(lower)

    upper_tail = r[r > q_up]
    lower_tail = r[r < q_lo]

    if upper_tail.empty or lower_tail.empty: return np.nan

    avg_up = upper_tail.mean()
    avg_lo = abs(lower_tail.mean())

    if avg_lo == 0: return np.nan
    return float(avg_up / avg_lo)


# ==========================================
# PAIN RATIO (Gain-to-Pain)
# ==========================================

def pain_ratio(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Pain Ratio (a.k.a. Gain-to-Pain Ratio, Schwager).

        Pain Ratio = sum of gains / sum of absolute losses

    Mathematically identical to the Omega Ratio with a zero threshold, so
    this is implemented as a thin wrapper over `omega_ratio(returns, mar=0.0)`.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when there are no losses.
    """
    return omega_ratio(returns, mar=0.0)


# ==========================================
# UPSIDE/DOWNSIDE DEVIATION RATIO
# ==========================================

def deviation_ratio(returns: Union[pd.Series, pd.DataFrame],
    mar: float = 0.0) -> Union[float, pd.Series]:
    """
    Upside/Downside Deviation Ratio.

        U/D Dev = sqrt(E[(r - τ)² | r > τ]) / sqrt(E[(τ - r)² | r < τ])

    Ratio of conditional upside deviation to conditional downside deviation,
    both relative to threshold τ (MAR). Values above 1 indicate upside
    dispersion exceeds downside dispersion — a positively-skewed risk profile.

    Note: denominators are the *conditional* counts (observations in each
    tail), not the full sample size. This differs from `downside_variance`,
    which follows the Sortino convention of dividing by total N.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar: Minimum Acceptable Return threshold τ. Defaults to 0.0.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when either tail is empty or downside deviation is zero.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: deviation_ratio(col, mar))

    r = returns.dropna()
    if r.empty: return np.nan

    up = r[r > mar]
    dn = r[r < mar]

    if up.empty or dn.empty: return np.nan

    up_dev = np.sqrt(((up - mar) ** 2).mean())
    dn_dev = np.sqrt(((mar - dn) ** 2).mean())

    if dn_dev == 0: return np.nan
    return float(up_dev / dn_dev)


# ==========================================
# TAIL DISPERSION RATIO
# ==========================================

def tail_dispersion_ratio(returns: Union[pd.Series, pd.DataFrame],
    upper: float = 0.9, lower: float = 0.1) -> Union[float, pd.Series]:
    """
    Tail Dispersion Ratio — std of upper-tail returns ÷ std of lower-tail returns.

        Tail Dispersion = std(r | r > q_upper) / std(r | r < q_lower)

    Quantile-cutoff counterpart to `deviation_ratio`: measures how spread-out
    the extreme returns are within each tail. A value above 1 indicates the
    upper tail is more dispersed (fatter right tail) than the lower tail.

    Complements `tail_ratio`, which compares tail *means* at the same cutoffs;
    `tail_dispersion_ratio` compares tail *spreads*.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        upper: Upper quantile threshold. Defaults to 0.9.
        lower: Lower quantile threshold. Defaults to 0.1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when either tail has fewer than 2 observations or the lower-tail
        std is zero.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: tail_dispersion_ratio(col, upper, lower))

    r = returns.dropna()
    if r.empty: return np.nan

    q_up = r.quantile(upper)
    q_lo = r.quantile(lower)

    upper_tail = r[r > q_up]
    lower_tail = r[r < q_lo]

    if len(upper_tail) < 2 or len(lower_tail) < 2: return np.nan

    up_std = upper_tail.std(ddof=1)
    lo_std = lower_tail.std(ddof=1)

    if lo_std == 0: return np.nan
    return float(up_std / lo_std)
