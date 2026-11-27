
import datetime
import pandas as pd
import numpy as np
from typing import Literal, Union, Optional
import warnings

from alphalytics.utils import _infer_periods_per_year

# Anything pd.Timestamp accepts as a single date (pd.Timestamp is a datetime.date).
DateLike = Union[str, datetime.date, np.datetime64]

# Average days per year, counting leap years.
_DAYS_PER_YEAR = 365.25

# How far a trailing window's base date may sit before the window's calendar
# start: weekends and holiday closures. Monthly or coarser data gets no more
# than this, so a quarterly row cannot stand in for a one-month window.
_WINDOW_SLACK_DAYS = 6


class InsufficientHistoryError(ValueError):
    """The history is too short to annualize (less than one year)."""


def _check_annualize(annualize: str, start=None) -> None:
    """Validate an ``annualize`` choice and that ``start`` is used only with calendar."""
    if annualize not in ("periods", "calendar"):
        raise ValueError(f"annualize must be 'periods' or 'calendar', got {annualize!r}")
    if start is not None and annualize != "calendar":
        raise ValueError("start only applies when annualize='calendar'.")


def _require_calendar_index(index: pd.Index) -> None:
    """Calendar annualization reads dates, so they must be datetimes in ascending order."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            "annualize='calendar' needs a DatetimeIndex. "
            "Convert it with rets.index = pd.to_datetime(rets.index)."
        )
    if not index.is_monotonic_increasing:
        raise ValueError(
            "annualize='calendar' needs dates in ascending order. Sort with rets.sort_index()."
        )


def _as_base_date(start: DateLike, index: pd.DatetimeIndex) -> pd.Timestamp:
    """``start`` as a Timestamp in the index's time zone."""
    start = pd.Timestamp(start)
    if index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(index.tz)
    return start


def _calendar_window_start(index: pd.DatetimeIndex, months: int) -> Optional[int]:
    """Position of the first row of the trailing window ``months`` calendar months long.

    The window ends on the last date. Its calendar start is that date
    ``months`` months earlier, moved to its month end when the last date is
    a month end or the last business day of its month, so month-end data
    (business month-ends included) lines up month for month. The window
    holds the rows dated after the start.

    The base -- the last date on or before the start -- must be in the
    index, and no further before the start than the slack: one period less
    a day for sub-monthly data, at least ``_WINDOW_SLACK_DAYS``, and only
    ``_WINDOW_SLACK_DAYS`` for monthly or coarser data. Otherwise the data
    do not reach back to the start, or are too coarse to form the window
    (one month of quarterly data), and the result is None.
    """
    if len(index) < 2:
        return None
    end = index[-1]
    cutoff = end - pd.DateOffset(months=months)
    if end.is_month_end or end + pd.offsets.BMonthEnd(0) == end:
        cutoff = cutoff + pd.offsets.MonthEnd(0)

    base_pos = int(index.searchsorted(cutoff, side="right")) - 1
    if base_pos < 0:
        return None

    median_gap = pd.Series(index).diff().dt.days.median()
    slack = _WINDOW_SLACK_DAYS
    if median_gap < 27:
        slack = max(slack, median_gap - 1)
    if (cutoff - index[base_pos]).days > slack:
        return None
    return base_pos + 1


def _calendar_trailing_return(rets: pd.DataFrame, months: int) -> pd.Series:
    """Compounded return over the trailing ``months`` calendar months, per column.

    NaN for every column when the data do not form the window (see
    `_calendar_window_start`), and for a column whose track record does
    not span it (see `_covers_window`).
    """
    _require_calendar_index(rets.index)
    start = _calendar_window_start(rets.index, months)
    if start is None:
        return pd.Series(np.nan, index=rets.columns, dtype=float)
    ret = (rets.iloc[start:] + 1).prod() - 1
    return ret.where(_covers_window(rets, start), np.nan)


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

def ann_return(rets: pd.DataFrame, years: int = 3, periods_per_year: int | float | None = None,
    annualize: Literal["periods", "calendar"] = "periods") -> pd.Series:
    """
    Calculates the annualized trailing return over a specific number of years.

        R_ann = (1 + R_window) ** (1 / years) - 1

    The two methods choose the window differently:

    - ``"periods"`` (default): the last ``years * periods_per_year`` rows.
      ``periods_per_year`` is inferred from the index when omitted.
    - ``"calendar"``: the rows dated within ``years`` calendar years of the
      last date -- the exact span a trailing return "as of" that date
      covers, however many sessions it held. The data must reach back to
      the window's start: the base date before its first return has to be
      in the index (see `_calendar_window_start`). ``periods_per_year`` is
      ignored. For month-end or coarser data with no missing rows, both
      methods select the same rows.

    A column whose track record does not span the window is NaN (see
    `_covers_window`).
    """
    if years < 1:
        raise ValueError(f"years must be >= 1 to annualize, got {years}")
    _check_annualize(annualize)

    if annualize == "calendar":
        n_ret = _calendar_trailing_return(rets, 12 * years)
        ann_ret = (n_ret + 1) ** (1 / years) - 1
        ann_ret.name = f"{years} Year"
        return ann_ret

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

def ann_return_common_si(rets: pd.DataFrame, periods_per_year: int | float | None = None,
    annualize: Literal["periods", "calendar"] = "periods",
    start: DateLike | None = None) -> pd.Series:
    """
    Calculates the annualized Since Inception return over the COMMON period.
    Forces an apples-to-apples comparison by only evaluating dates where
    all strategies in the DataFrame have overlapping data.

    With R the compounded return over the common rows:

        periods:   R_ann = (1 + R) ** (P / n) - 1          (default)
        calendar:  R_ann = (1 + R) ** (365.25 / days) - 1

    ``periods`` divides the count of common returns ``n`` by a fixed
    ``P = periods_per_year``. ``calendar`` uses the elapsed time
    ``days = (end - start).days``, where ``end`` is the last date of the
    common period and ``start`` is its base date. That is the
    Morningstar / GIPS convention, and with daily data it does not drift
    when a year has more or fewer than 252 sessions.

    Args:
        rets: Periodic simple returns, one column per strategy.
        periods_per_year: ``P`` for the periods method. Inferred from the
            index when omitted, using the calendar before incomplete rows
            are dropped. Ignored when ``annualize="calendar"``.
        annualize: ``"periods"`` (default) or ``"calendar"``.
        start: Calendar method only. The inception (base) date: the date of
            the starting level, before the first common return. A return is
            dated at the end of its period, so this date is not one of the
            return dates. When omitted, the index date immediately before
            the first common return is used -- for example the leading NaN
            row that ``prices.pct_change()`` leaves on the base date. Pass
            the base date of the common period, not an earlier inception of
            one strategy, or time when not every strategy was present is
            counted.

    Returns:
        Annualized return per strategy, named "SI".

    Raises:
        InsufficientHistoryError: A ``ValueError`` raised when the common
            period is shorter than one year (``P`` returns for the periods
            method, 365.25 days for calendar) or there is none.
        ValueError: ``annualize`` is not ``"periods"`` or ``"calendar"``;
            ``start`` is given with the periods method, or falls after the
            first common return; the calendar method has no ``start`` and no
            index date before the first common return; or the dates are not
            in ascending order.
        TypeError: The calendar method is used on a non-datetime index.

    Note:
        A row dropped inside the common period (a gap in any one strategy)
        is skipped by both methods. The periods method also leaves it out
        of ``n``; the calendar method still counts its time.

    Example:
        >>> idx = pd.date_range("2020-12-31", periods=25, freq="ME")
        >>> rets = pd.DataFrame({"Fund": 0.01}, index=idx)
        >>> rets.iloc[0] = np.nan  # base-date row, as prices.pct_change() leaves it
        >>> ann_return_common_si(rets).round(4)  # 24 returns / 12 per year
        Fund    0.1268
        Name: SI, dtype: float64
        >>> ann_return_common_si(rets, annualize="calendar").round(4)  # 730 days
        Fund    0.1269
        Name: SI, dtype: float64
        >>> ann_return_common_si(rets.iloc[1:], annualize="calendar",
        ...                      start="2020-12-31").round(4)
        Fund    0.1269
        Name: SI, dtype: float64
    """
    _check_annualize(annualize, start)

    if rets.empty:
        return pd.Series(np.nan, index=rets.columns)

    if annualize == "calendar":
        return _common_si_calendar(rets, start)

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(rets.index)

    # 1. Align data: Keep only dates where ALL columns have data
    aligned_rets = rets.dropna()

    total_periods = len(aligned_rets)

    # 2. Safety Check: Ensure the overlapping period is at least 1 year
    if total_periods < periods_per_year:
        raise InsufficientHistoryError("The overlapping 'apples-to-apples' history is less than 1 year. Cannot annualize.")

    years = total_periods / periods_per_year

    # 3. Calculate total cumulative return over the perfectly aligned period
    cumret = (aligned_rets + 1).prod() - 1

    # 4. Annualize based on the exact fractional years of the common period
    si_ret = (cumret + 1) ** (1 / years) - 1
    si_ret.name = "SI"

    return si_ret


def _common_si_calendar(rets: pd.DataFrame, start: DateLike | None) -> pd.Series:
    """Calendar-time branch of `ann_return_common_si` for a non-empty frame."""
    dates = rets.index
    _require_calendar_index(dates)

    complete = rets.notna().all(axis=1).to_numpy()
    if not complete.any():
        raise InsufficientHistoryError(
            "There is no overlapping 'apples-to-apples' history. Cannot annualize.")
    complete_pos = np.flatnonzero(complete)
    first, end = dates[complete_pos[0]], dates[complete_pos[-1]]

    if start is None:
        if complete_pos[0] == 0:
            raise ValueError(
                f"annualize='calendar' needs the base date before the first common return "
                f"({first:%Y-%m-%d}), and the index has no earlier date. Pass start= with the "
                "inception date: the date of the starting level, one period before the first return."
            )
        start = dates[complete_pos[0] - 1]
    else:
        start = _as_base_date(start, dates)
        if start > first:
            raise ValueError(
                f"start ({start:%Y-%m-%d}) is after the first common return ({first:%Y-%m-%d}). "
                "start is the base date before that return."
            )

    days = (end - start).days
    if days < _DAYS_PER_YEAR:
        raise InsufficientHistoryError(
            f"The overlapping 'apples-to-apples' history is {days} days, less than 1 calendar "
            "year. Cannot annualize."
        )
    years = days / _DAYS_PER_YEAR

    cumret = (rets[complete] + 1).prod() - 1
    si_ret = (cumret + 1) ** (1 / years) - 1
    si_ret.name = "SI"
    return si_ret


def compounded_ann_return(returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int | float | None = None,
    annualize: Literal["periods", "calendar"] = "periods",
    start: DateLike | None = None) -> Union[float, pd.Series]:
    """Annualized compounded return over the whole sample.

        periods:   R_ann = prod(1 + r) ** (periods_per_year / n) - 1   (default)
        calendar:  R_ann = prod(1 + r) ** (365.25 / days) - 1

    ``n`` is the count of finite returns in each column. NaNs are skipped
    per column. A non-positive growth path (a return below -100%, or a
    negative product) is NaN. A path that hits exactly -100% annualizes
    to -1.

    For ``"calendar"``, ``days`` runs from each column's base date to its
    last finite return. The base date is the index date immediately before
    the column's first finite return, so a strategy that launched later
    uses its own. A column whose first return is on the first date has no
    such date in the index: ``start`` supplies it. A gap inside a column
    still counts as elapsed time. Unlike `ann_return_common_si`, a history
    shorter than a year is annualized, as in the periods method.

    Args:
        returns: Periodic simple returns.
        periods_per_year: Annualization factor. Inferred from the index
            when omitted. Ignored for ``"calendar"``.
        annualize: ``"periods"`` (default) or ``"calendar"``.
        start: Calendar method only: the base date before the first row of
            the index, used by columns whose first return is on that row.

    Raises:
        ValueError: An invalid ``annualize``; ``start`` with the periods
            method, or after the first date; or, for ``"calendar"``, a
            column whose first return is on the first date when ``start``
            is not given.
    """
    _check_annualize(annualize, start)
    if annualize == "calendar":
        return _compounded_ann_return_calendar(returns, start)

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


def _compounded_ann_return_calendar(returns: Union[pd.Series, pd.DataFrame],
    start: DateLike | None) -> Union[float, pd.Series]:
    """Calendar-time branch of `compounded_ann_return`."""
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError(f"returns must be a Series or DataFrame, got {type(returns).__name__}")
    dates = returns.index
    _require_calendar_index(dates)
    if start is not None:
        start = _as_base_date(start, dates)
        if len(dates) and start > dates[0]:
            raise ValueError(
                f"start ({start:%Y-%m-%d}) is after the first date ({dates[0]:%Y-%m-%d}). "
                "start is the base date before the first row."
            )

    def _one(series: pd.Series):
        pos = np.flatnonzero(series.notna().to_numpy())
        if len(pos) == 0:
            return np.nan
        growth = float((1 + series.iloc[pos]).prod())
        if growth < 0:
            return np.nan
        if growth == 0:
            return -1.0
        if pos[0] > 0:
            base = dates[pos[0] - 1]
        elif start is not None:
            base = start
        else:
            raise ValueError(
                f"annualize='calendar' needs the base date before {series.name!r}'s first "
                f"return ({dates[0]:%Y-%m-%d}), and the index has no earlier date. Pass "
                "start= with the date of the starting level."
            )
        days = (dates[pos[-1]] - base).days
        if days <= 0:
            return np.nan
        return float(growth ** (_DAYS_PER_YEAR / days) - 1)

    if isinstance(returns, pd.Series):
        return _one(returns)
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