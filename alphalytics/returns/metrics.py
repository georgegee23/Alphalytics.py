
import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings

from alphalytics.utils import _infer_periods_per_year


# ============== RETURNS FUNCTIONS ============== #

def _covers_window(rets: pd.DataFrame, start: int, end: int = -1) -> pd.Series:
    """Columns whose track record spans rows ``start`` through ``end``.

    ``end`` defaults to the last row. A column covers the window when it has
    an observation on or before the window's first row and an observation on
    its last row. A strategy that launched inside the window, or stopped
    reporting before it ended, does not cover it: compounding its shorter
    history would understate a multi-year return once annualized over the
    full window. A gap strictly inside the window is still skipped.
    """
    started = rets.iloc[: start + 1].notna().any()
    current = rets.iloc[end].notna()
    return started & current


def return_n(rets: pd.DataFrame, n: int) -> pd.Series:
    """
    Calculates the cumulative trailing return over the last 'n' periods.

    A window shorter than one period is undefined (``iloc[-0:]`` would
    otherwise compound the whole history) and returns NaN. A column whose
    track record does not span the window is NaN (see `_covers_window`).
    """
    if n < 1 or len(rets) < n:
        return pd.Series(np.nan, index=rets.columns, dtype=float, name=f"{n} Period")

    window = rets.iloc[-n:]
    ret_n = (window + 1).prod() - 1
    ret_n = ret_n.where(_covers_window(rets, len(rets) - n), np.nan)
    ret_n.name = f"{n} Period"
    return ret_n

def return_ytd(rets: pd.DataFrame) -> pd.Series:
    """
    Calculates YTD return using the calendar year of the last available date.
    Assumes rets has a sorted DatetimeIndex.

    A column with no observation on or before the first date of that year
    (it launched during the year) or on the last date is NaN.
    """
    if rets.empty:
        return pd.Series(0.0, index=rets.columns)

    # Identify the current year based on the very last data point
    current_year = rets.index[-1].year

    # Slice only the returns belonging to that year
    in_year = rets.index.year == current_year
    ytd_data = rets[in_year]

    # Calculate geometric return for that slice
    ytd_rets = (ytd_data + 1).prod() - 1
    ytd_rets = ytd_rets.where(_covers_window(rets, int(np.argmax(in_year))), np.nan)
    ytd_rets.name = "YTD"
    return ytd_rets

# Weekends and holidays move the first and last trading days of a year by up
# to three days, so sub-monthly data gets that much slack at the year's edges.
_EDGE_SLACK_DAYS = 3


def _starts_year(date: pd.Timestamp, periods_per_year: int) -> bool:
    """Whether ``date`` falls in the first period of its calendar year."""
    if periods_per_year <= 12:
        return date.month <= 12 // periods_per_year
    return date.dayofyear <= 365.25 / periods_per_year + _EDGE_SLACK_DAYS


def _ends_year(date: pd.Timestamp, periods_per_year: int) -> bool:
    """Whether ``date`` falls in the last period of its calendar year."""
    if periods_per_year <= 12:
        return date.month > 12 - 12 // periods_per_year
    days_left = (date.replace(month=12, day=31).normalize() - date.normalize()).days
    return days_left <= 365.25 / periods_per_year + _EDGE_SLACK_DAYS


def calendar_returns(rets: Union[pd.Series, pd.DataFrame], periods_per_year: int = None,
                     ytd_label: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Compounded return for each calendar year.

    A year is reported only when the data begin by its first period, so a
    first year that starts part-way through is left out, even when it is
    also the final year. The final year is reported year-to-date when the
    data end before its last period; that value equals `return_ytd`.

    Each column follows the same coverage rule as the trailing returns (see
    `_covers_window`): it needs an observation on or before the year's first
    row and on its last row, or that year is NaN for it -- it launched during
    the year, or stopped reporting before the year ended. A gap strictly
    inside a year is skipped.

    Args:
        rets: Periodic returns (Series or DataFrame) with a sorted DatetimeIndex.
        periods_per_year: Used only to recognize the first and last period of
            a year. Inferred from the index when omitted.
        ytd_label: If given, an incomplete final year is labelled with this
            string (e.g. "YTD") instead of its year number.

    Returns:
        Series (Series input) or DataFrame (DataFrame input) indexed by
        calendar year, oldest first. The index is named "Year".
    """
    frame = rets.to_frame() if isinstance(rets, pd.Series) else rets

    rows = {}
    if not frame.empty:
        if periods_per_year is None:
            periods_per_year = _infer_periods_per_year(frame.index)
        years = frame.index.year
        for year in pd.unique(years):
            pos = np.flatnonzero(years == year)
            start, end = int(pos[0]), int(pos[-1])
            if start == 0 and not _starts_year(frame.index[0], periods_per_year):
                continue  # the data begin part-way through this year
            ret = (frame.iloc[start:end + 1] + 1).prod() - 1
            rows[year] = ret.where(_covers_window(frame, start, end), np.nan)

    out = pd.DataFrame(list(rows.values()), index=list(rows.keys()),
                       columns=frame.columns, dtype=float)
    if (ytd_label is not None and len(out)
            and out.index[-1] == frame.index[-1].year
            and not _ends_year(frame.index[-1], periods_per_year)):
        out.index = list(out.index[:-1]) + [ytd_label]
    out.index.name = "Year"

    if isinstance(rets, pd.Series):
        return out.iloc[:, 0].rename(rets.name)
    return out

def ann_return(rets: pd.DataFrame, years: int = 3, periods_per_year: int = None) -> pd.Series:
    """
    Calculates the annualized trailing return over a specific number of years.

    ``periods_per_year`` is inferred from the index when omitted. A column
    whose track record does not span the window is NaN (see `_covers_window`).
    """
    if years < 1:
        raise ValueError(f"years must be >= 1 to annualize, got {years}")

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(rets.index)

    n = years * periods_per_year

    # Safety check: If history is shorter than requested periods, return NaN.
    # Keep the label so a table built from these columns stays readable.
    if len(rets) < n:
        return pd.Series(np.nan, index=rets.columns, dtype=float, name=f"{years} Year")

    # 1. Calculate cumulative return over the sliced last 'n' periods.
    window = rets.iloc[-n:]
    n_ret = (window + 1).prod() - 1
    n_ret = n_ret.where(_covers_window(rets, len(rets) - n), np.nan)

    # 2. Annualize the cumulative return
    ann_ret = (n_ret + 1) ** (1 / years) - 1
    ann_ret.name = f"{years} Year"

    return ann_ret

def ann_return_common_si(rets: pd.DataFrame, periods_per_year: int = None) -> pd.Series:
    """
    Calculates the annualized Since Inception return over the COMMON period.
    Forces an apples-to-apples comparison by only evaluating dates where
    all strategies in the DataFrame have overlapping data.

    ``periods_per_year`` is inferred from the index when omitted, using the
    calendar before incomplete rows are dropped.
    """
    if rets.empty:
        return pd.Series(np.nan, index=rets.columns)

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(rets.index)

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


def compounded_ann_return(returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: Optional[int] = None) -> Union[float, pd.Series]:
    """Annualized compounded return over the whole sample.

        R_ann = prod(1 + r) ** (periods_per_year / n) - 1

    ``n`` is the count of finite returns in each column. NaNs are skipped
    per column. A non-positive growth path (a return below -100%, or a
    negative product) is NaN. A path that hits exactly -100% annualizes
    to -1.

    Args:
        returns: Periodic simple returns.
        periods_per_year: Annualization factor. Inferred from the index
            when omitted.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")

    def _one(series: pd.Series):
        r = series.dropna()
        n = len(r)
        if n == 0:
            return np.nan
        growth = float((1 + r).prod())
        if growth < 0:
            return np.nan
        if growth == 0:
            return -1.0
        return float(growth ** (periods_per_year / n) - 1)

    if isinstance(returns, pd.Series):
        return _one(returns)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a Series or DataFrame, got {type(returns).__name__}")
    return returns.apply(_one, axis=0)


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


def average_gain(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Mean of the positive periodic returns -- the typical winning period.

    Zero returns and gaps (NaN) are in neither `average_gain` nor
    `average_loss`. NaN when there is no positive period.

    Returns:
        float (Series input) or pd.Series (DataFrame input).
    """
    return returns.where(returns > 0).mean()


def average_loss(returns: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Mean of the negative periodic returns -- the typical losing period.

    The result is negative. Zero returns and gaps (NaN) are excluded. NaN when
    there is no negative period.

    Returns:
        float (Series input) or pd.Series (DataFrame input).
    """
    return returns.where(returns < 0).mean()