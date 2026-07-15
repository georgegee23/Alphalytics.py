
import pandas as pd
import numpy as np
from typing import Union

from .risk import downside_variance, upside_variance, max_drawdown, ulcer_index, cvar
from .relative import omega_ratio
from .capm import beta
from alphalytics.utils import _infer_periods_per_year


# ============== RATIO STATISTICS ============== #

# ==========================================
# SHARPE RATIO
# ==========================================

def sharpe_ratio(returns: Union[pd.Series, pd.DataFrame], rfr: float = 0.0,
    periods_per_year: int = None, annualize: bool = True) -> Union[float, pd.Series]:
    """
    Sharpe ratio — mean excess return per unit of total volatility.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        rfr: Risk-free rate per period. Defaults to 0.0.
        periods_per_year: Annualisation factor (inferred from index if None).
            Ignored when ``annualize=False``.
        annualize: If True (default), scale the per-period ratio by
            ``sqrt(periods_per_year)``. Set to False to return the raw
            per-period ratio.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
    """
    excess = returns - rfr
    std = excess.std()
    if isinstance(std, pd.Series):
        std = std.where(~np.isclose(std, 0, atol=1e-12), np.nan)
    elif std == 0 or np.isclose(std, 0, atol=1e-12):
        return np.nan

    ratio = excess.mean() / std

    if not annualize:
        return ratio

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)
    return ratio * np.sqrt(periods_per_year)

# ==========================================
# TREYNOR RATIO
# ==========================================

def treynor_ratio(returns: Union[pd.Series, pd.DataFrame], benchmark: pd.Series,
    rfr: float = 0.0, periods_per_year: int = None,
    annualize: bool = True) -> Union[float, pd.Series]:
    """
    Treynor ratio — excess return per unit of systematic risk.

        Treynor = mean(r - rfr) / β                       (per-period)
                = (mean(r - rfr) * periods_per_year) / β  (annualised)

    Where β is the strategy's beta to the benchmark (sample covariance with
    ddof=1). Unlike the Sharpe ratio, the denominator is systematic risk
    only, so Treynor rewards strategies that earn excess return without
    taking on additional market exposure. Annualisation is linear (not
    sqrt) because β is dimensionless — only the numerator scales.

    Inputs are aligned on their common index and rows with NaNs are dropped
    before computing both the mean excess return and beta.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        benchmark: Periodic returns of the benchmark.
        rfr: Risk-free rate per period. Defaults to 0.0.
        periods_per_year: Annualisation factor (inferred from index if None).
            Ignored when ``annualize=False``.
        annualize: If True (default), scale the per-period ratio by
            ``periods_per_year``. Set to False to return the raw
            per-period ratio.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN where beta is zero, undefined, or the benchmark is flat.
    """
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    is_series = isinstance(returns, pd.Series)
    if is_series:
        returns = returns.to_frame()

    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()
    aligned_returns = combined[returns.columns]
    aligned_bench = combined["__benchmark__"]

    betas = beta(aligned_returns, aligned_bench)
    betas = betas.where(~np.isclose(betas, 0, atol=1e-15), np.nan)

    mean_excess = aligned_returns.mean() - rfr
    treynor = mean_excess / betas

    if annualize:
        if periods_per_year is None:
            periods_per_year = _infer_periods_per_year(returns.index)
        treynor = treynor * periods_per_year

    return treynor.iloc[0] if is_series else treynor

# ==========================================
# SORTINO RATIO
# ==========================================

def sortino_ratio(returns: Union[pd.Series, pd.DataFrame], mar: float = 0.0,
    periods_per_year: int = None, ddof: int = 1,
    annualize: bool = True) -> Union[float, pd.Series]:
    """
    Sortino Ratio — mean excess return per unit of downside deviation.

    Improves on the Sharpe Ratio by penalizing only downside volatility,
    making it more appropriate for strategies with asymmetric return distributions.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar (float): Minimum Acceptable Return threshold. Defaults to 0.0.
        periods_per_year (int): Annualization factor. Inferred from index if None.
            Ignored when ``annualize=False``.
        ddof (int): Degrees of freedom for downside variance. Defaults to 1.
        annualize: If True (default), scale the per-period ratio by
            ``sqrt(periods_per_year)``. Set to False to return the raw
            per-period ratio.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when downside deviation is zero or undefined.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    down_var = downside_variance(returns, mar=mar, ddof=ddof)
    if isinstance(down_var, pd.Series):
        down_dev = np.sqrt(down_var).replace(0, np.nan)
    else:
        down_dev = np.sqrt(down_var)
        if down_dev == 0:
            down_dev = np.nan

    mean_excess_return = returns.mean() - mar

    sortino = mean_excess_return / down_dev

    if annualize:
        if periods_per_year is None:
            periods_per_year = _infer_periods_per_year(returns.index)
        sortino = sortino * np.sqrt(periods_per_year)

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
# MARTIN RATIO
# ==========================================

def martin_ratio(returns: Union[pd.Series, pd.DataFrame], rfr: float = 0.0,
    periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Martin Ratio (a.k.a. Ulcer Performance Index) — annualised excess return
    per unit of Ulcer Index.

        Martin = (annualised return - annualised rfr) / Ulcer Index

    A drawdown-based analogue of the Sharpe ratio: the denominator is the
    root-mean-square drawdown rather than return volatility, so the metric
    rewards strategies that produce excess return without spending long
    stretches underwater. Compared to Calmar, which divides by max drawdown
    only, Martin penalises the full drawdown trajectory (depth × duration).
    Returns and rfr are annualised geometrically, matching `calmar_ratio`.
    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        rfr: Risk-free rate per period. Defaults to 0.0.
        periods_per_year: Annualisation factor (inferred from index if None).

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN where the Ulcer Index is zero (no drawdowns) or undefined.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    ann_rfr = (1 + rfr) ** periods_per_year - 1
    ui = ulcer_index(returns)

    if isinstance(ui, pd.Series):
        ui = ui.where(~np.isclose(ui, 0, atol=1e-12), np.nan)
    elif ui == 0 or np.isclose(ui, 0, atol=1e-12):
        return np.nan

    return (ann_ret - ann_rfr) / ui


# ==========================================
# STARR RATIO
# ==========================================

def starr(returns: Union[pd.Series, pd.DataFrame], rfr: float = 0.0,
    alpha: float = 0.95, annualize: bool = True,
    periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    STARR (Stable Tail-Adjusted Return Ratio, Rachev et al.).

        STARR        = mean(r - rfr) / |CVaR_α(r)|                            (per-period)
        STARR_ann    = (R_ann - R_f_ann) / (|CVaR_α(r)| * sqrt(P))            (annualised)

    A coherent-risk analogue of the Sharpe ratio: replaces the standard
    deviation in the denominator with Conditional VaR, so dispersion above
    the mean is ignored and only worst-tail losses penalise the ratio.

    By default the result is per-period — pre-scale `rfr` to the same period
    as `returns`. With ``annualize=True`` the numerator is the geometric
    annualised excess return and the per-period CVaR magnitude is scaled by
    sqrt(P), matching the standard convention used to compare against an
    annualised Sharpe ratio. The sqrt(P) scaling is exact only for i.i.d.
    Gaussian returns; for fat-tailed series it understates the true annual
    tail risk and the metric becomes approximate.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        rfr: Risk-free rate per period. Defaults to 0.0.
        alpha: CVaR confidence level. Defaults to 0.95 (5% tail).
        annualize: If True (default), return the annualised form. Set to
            False for the raw per-period ratio.
        periods_per_year: Annualisation factor (inferred from index if
            None). Ignored when ``annualize=False``.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN where CVaR is zero or undefined (e.g. empty tail).
    """
    es_abs = np.abs(cvar(returns, alpha=alpha))

    if isinstance(es_abs, pd.Series):
        es_abs = es_abs.where(~np.isclose(es_abs, 0, atol=1e-12), np.nan)
    elif es_abs == 0 or np.isclose(es_abs, 0, atol=1e-12):
        return np.nan

    if not annualize:
        return (returns - rfr).mean() / es_abs

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    ann_rfr = (1 + rfr) ** periods_per_year - 1
    return (ann_ret - ann_rfr) / (es_abs * np.sqrt(periods_per_year))


# ==========================================
# RACHEV RATIO
# ==========================================

def rachev_ratio(returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.95) -> Union[float, pd.Series]:
    """
    Rachev Ratio — ratio of upper-tail expected gain to lower-tail expected loss.

        Rachev = CVaR_α(gains) / |CVaR_α(losses)|

    Both tails are evaluated at the same confidence level α: the numerator is
    the mean return in the best (1 - α) fraction of observations, the
    denominator is the magnitude of the mean return in the worst (1 - α)
    fraction. A value above 1 means the average win in the upper tail
    exceeds the average loss in the lower tail.

    Conceptual sibling of `tail_ratio`, which uses tail *means* above/below
    quantile cutoffs. Rachev uses Expected Shortfall on both sides — a
    coherent risk measure — so the result is more stable in fat-tailed
    samples.

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        alpha: Confidence level applied symmetrically to both tails.
            Defaults to 0.95 (5% tail on each side).

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN where either tail is empty or the loss tail magnitude is zero.
    """
    upper = -cvar(-returns, alpha=alpha)
    lower = -cvar(returns, alpha=alpha)

    if isinstance(lower, pd.Series):
        lower = lower.where(~np.isclose(lower, 0, atol=1e-12), np.nan)
    elif lower == 0 or np.isclose(lower, 0, atol=1e-12):
        return np.nan

    return upper / lower


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
    mar: float = 0.0, ddof: int = 1) -> Union[float, pd.Series]:
    """
    Upside / Downside Deviation Ratio (UD/DD).

        UD/DD = sqrt(upside_variance) / sqrt(downside_variance)

    Ratio of upside deviation to downside deviation relative to the MAR.
    Values above 1 indicate upside dispersion exceeds downside dispersion —
    a positively-skewed risk profile.

    Both sides use the Sortino convention (denominator = N - ddof) via
    `upside_variance` and `downside_variance`, so the ratio simplifies to
    sqrt( sum((r - mar)₊²) / sum((mar - r)₊²) ).

    Args:
        returns: A pandas Series or DataFrame of periodic returns.
        mar: Minimum Acceptable Return threshold. Defaults to 0.0.
        ddof (int): Degrees of freedom shared by both variances. Defaults to 1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when downside variance is zero or observations are insufficient.
    """
    up_var = upside_variance(returns, mar=mar, ddof=ddof)
    dn_var = downside_variance(returns, mar=mar, ddof=ddof)

    if isinstance(dn_var, pd.Series):
        return np.sqrt(up_var / dn_var.where(dn_var > 0))

    if pd.isna(dn_var) or dn_var == 0:
        return np.nan
    return float(np.sqrt(up_var / dn_var))


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
