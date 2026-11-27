"""Tests for alphalytics.returns.metrics — return calculations."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.metrics import (
    InsufficientHistoryError,
    return_n,
    return_ytd,
    ann_return,
    ann_return_common_si,
    compounded_ann_return,
    calendar_returns,
    cumgrowth,
    annualized_rolling_return,
    average_gain,
    average_loss,
)


class TestReturnN:
    def test_basic(self, returns_df):
        result = return_n(returns_df, n=12)
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_insufficient_data(self, returns_df):
        result = return_n(returns_df, n=100)
        assert result.isna().all()

    def test_non_positive_window_is_nan(self, returns_df):
        result = return_n(returns_df, n=0)
        assert result.isna().all()
        assert list(result.index) == list(returns_df.columns)


class TestReturnYTD:
    def test_basic(self, returns_df):
        result = return_ytd(returns_df)
        assert isinstance(result, pd.Series)

    def test_empty(self, monthly_index):
        empty = pd.DataFrame(index=monthly_index[:0], columns=["A"])
        result = return_ytd(empty)
        assert (result == 0).all()


class TestAnnReturn:
    def test_basic(self, returns_df):
        result = ann_return(returns_df, years=1, periods_per_year=12)
        assert isinstance(result, pd.Series)

    def test_insufficient_history(self, returns_df):
        result = ann_return(returns_df, years=10, periods_per_year=12)
        assert result.isna().all()

    def test_years_below_one_raises(self, returns_df):
        with pytest.raises(ValueError):
            ann_return(returns_df, years=0)

    def test_infers_monthly_frequency(self, returns_df):
        inferred = ann_return(returns_df, years=1)
        explicit = ann_return(returns_df, years=1, periods_per_year=12)
        assert inferred.equals(explicit)


class TestTrailingCoverage:
    """A trailing window needs a track record that spans it."""

    @staticmethod
    def _frame():
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        new = np.full(60, np.nan)
        new[-24:] = 0.01  # launched two years before the end
        return pd.DataFrame({"OLD": np.full(60, 0.01), "NEW": new}, index=idx)

    def test_late_launch_is_nan_beyond_its_history(self):
        frame = self._frame()
        three = ann_return(frame, years=3, periods_per_year=12)
        assert three["OLD"] == pytest.approx(1.01 ** 12 - 1)
        assert np.isnan(three["NEW"])
        two = ann_return(frame, years=2, periods_per_year=12)
        assert two["NEW"] == pytest.approx(1.01 ** 12 - 1)

    def test_late_launch_return_n(self):
        frame = self._frame()
        assert np.isnan(return_n(frame, 36)["NEW"])
        assert return_n(frame, 24)["NEW"] == pytest.approx(1.01 ** 24 - 1)

    def test_stale_series_is_nan(self):
        frame = self._frame()
        frame.iloc[-1, 0] = np.nan  # OLD has no return on the last date
        assert np.isnan(ann_return(frame, years=1, periods_per_year=12)["OLD"])

    def test_internal_gap_is_skipped(self):
        frame = self._frame()
        frame.iloc[-6, 0] = np.nan
        result = ann_return(frame, years=1, periods_per_year=12)["OLD"]
        assert result == pytest.approx(1.01 ** 11 - 1)

    def test_ytd_needs_start_of_year(self):
        idx = pd.date_range("2020-01-31", periods=18, freq="ME")  # ends Jun 2021
        frame = pd.DataFrame({"A": 0.01, "B": np.nan}, index=idx)
        frame.loc["2021-03-31":, "B"] = 0.01
        ytd = return_ytd(frame)
        assert ytd["A"] == pytest.approx(1.01 ** 6 - 1)
        assert np.isnan(ytd["B"])

    def test_short_history_keeps_label(self):
        frame = self._frame()
        assert ann_return(frame, years=10, periods_per_year=12).name == "10 Year"
        assert return_n(frame, 100).name == "100 Period"


class TestCalendarTrailing:
    """ann_return(annualize="calendar") selects an exact calendar span."""

    @staticmethod
    def _daily():
        idx = pd.bdate_range("2021-01-01", "2024-12-31")
        return pd.DataFrame({"A": 0.0004}, index=idx)

    def test_daily_one_year_is_the_calendar_year(self):
        rets = self._daily()
        n_2024 = int((rets.index.year == 2024).sum())  # 262 weekdays, not 252
        result = ann_return(rets, years=1, annualize="calendar")["A"]
        assert result == pytest.approx(1.0004 ** n_2024 - 1)
        periods = ann_return(rets, years=1, periods_per_year=252)["A"]
        assert periods == pytest.approx(1.0004 ** 252 - 1)

    def test_daily_three_years_annualized_over_exactly_three(self):
        rets = self._daily()
        n = int((rets.index.year >= 2022).sum())
        result = ann_return(rets, years=3, annualize="calendar")["A"]
        assert result == pytest.approx(1.0004 ** (n / 3) - 1)

    def test_month_end_data_matches_periods(self):
        idx = pd.date_range("2015-01-31", periods=73, freq="ME")
        rets = pd.DataFrame({"A": np.random.default_rng(1).normal(0.01, 0.03, 73)}, index=idx)
        for k in (1, 3, 5):
            calendar = ann_return(rets, years=k, annualize="calendar")
            assert calendar.equals(ann_return(rets, years=k, periods_per_year=12))

    def test_business_month_end_data_lines_up(self):
        idx = pd.date_range("2015-01-01", periods=73, freq="BME")
        rets = pd.DataFrame({"A": np.random.default_rng(2).normal(0.01, 0.03, 73)}, index=idx)
        for k in (1, 3, 5):
            calendar = ann_return(rets, years=k, annualize="calendar")
            assert calendar.to_numpy() == pytest.approx(
                ann_return(rets, years=k, periods_per_year=12).to_numpy())

    def test_weekly_one_year(self):
        idx = pd.date_range("2018-01-05", periods=298, freq="W-FRI")  # ends mid-month
        rets = pd.DataFrame({"A": 0.001}, index=idx)
        weeks = int((idx > idx[-1] - pd.DateOffset(years=1)).sum())
        result = ann_return(rets, years=1, annualize="calendar")["A"]
        assert result == pytest.approx(1.001 ** weeks - 1)

    def test_window_needs_its_base_date(self):
        idx = pd.date_range("2020-01-31", periods=36, freq="ME")  # no row for 2019-12-31
        rets = pd.DataFrame({"A": 0.01}, index=idx)
        assert np.isnan(ann_return(rets, years=3, annualize="calendar")["A"])
        with_base = pd.concat([pd.DataFrame({"A": [np.nan]}, index=[pd.Timestamp("2019-12-31")]), rets])
        result = ann_return(with_base, years=3, annualize="calendar")["A"]
        assert result == pytest.approx(1.01 ** 12 - 1)

    def test_late_launch_is_nan(self):
        rets = self._daily()
        rets["NEW"] = np.nan
        rets.loc["2023-06-01":, "NEW"] = 0.0004
        assert np.isnan(ann_return(rets, years=3, annualize="calendar")["NEW"])
        assert ann_return(rets, years=1, annualize="calendar")["NEW"] == pytest.approx(
            ann_return(rets, years=1, annualize="calendar")["A"])

    def test_keeps_label(self):
        assert ann_return(self._daily(), years=10, annualize="calendar").name == "10 Year"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="annualize"):
            ann_return(self._daily(), years=1, annualize="actual")


class TestCompoundedAnnReturnCalendar:
    @staticmethod
    def _with_base_row():
        idx = pd.date_range("2020-12-31", periods=25, freq="ME")
        rets = pd.DataFrame({"A": 0.01}, index=idx)
        rets.iloc[0] = np.nan
        return rets

    def test_base_row(self):
        result = compounded_ann_return(self._with_base_row(), annualize="calendar")["A"]
        assert result == pytest.approx(1.01 ** (24 * 365.25 / 730) - 1)

    def test_later_column_uses_its_own_base(self):
        rets = self._with_base_row()
        rets["B"] = np.nan
        rets.loc["2021-07-31":, "B"] = 0.02  # base 2021-06-30, 18 returns
        days = (rets.index[-1] - pd.Timestamp("2021-06-30")).days
        result = compounded_ann_return(rets, annualize="calendar")["B"]
        assert result == pytest.approx(1.02 ** (18 * 365.25 / days) - 1)

    def test_first_row_return_needs_start(self):
        rets = self._with_base_row().iloc[1:]
        with pytest.raises(ValueError, match="start="):
            compounded_ann_return(rets, annualize="calendar")
        result = compounded_ann_return(rets["A"], annualize="calendar", start="2020-12-31")
        assert result == pytest.approx(1.01 ** (24 * 365.25 / 730) - 1)

    def test_start_after_first_date_raises(self):
        rets = self._with_base_row().iloc[1:]
        with pytest.raises(ValueError, match="after the first date"):
            compounded_ann_return(rets, annualize="calendar", start="2021-02-15")

    def test_start_with_periods_raises(self, returns_df):
        with pytest.raises(ValueError, match="calendar"):
            compounded_ann_return(returns_df, start="2019-12-31")


class TestAnnReturnCommonSI:
    def test_short_history_is_insufficient_history_error(self, monthly_index):
        short = pd.DataFrame({"A": [0.01] * 6}, index=monthly_index[:6])
        with pytest.raises(InsufficientHistoryError):
            ann_return_common_si(short, periods_per_year=12)

    def test_basic(self, returns_df):
        result = ann_return_common_si(returns_df, periods_per_year=12)
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_too_short_raises(self, monthly_index):
        short = pd.DataFrame({"A": [0.01] * 6}, index=monthly_index[:6])
        with pytest.raises(ValueError):
            ann_return_common_si(short, periods_per_year=12)

    def test_infers_monthly_frequency(self, returns_df):
        inferred = ann_return_common_si(returns_df)
        explicit = ann_return_common_si(returns_df, periods_per_year=12)
        assert inferred.equals(explicit)

    def test_daily_default_is_not_monthly(self, daily_index):
        rets = pd.DataFrame({"A": np.full(len(daily_index), 0.001)}, index=daily_index)
        inferred = ann_return_common_si(rets).iloc[0]
        as_monthly = ann_return_common_si(rets, periods_per_year=12).iloc[0]
        assert inferred == pytest.approx(ann_return_common_si(rets, periods_per_year=252).iloc[0])
        assert inferred != pytest.approx(as_monthly)

    def test_periods_is_the_default(self, returns_df):
        default = ann_return_common_si(returns_df)
        assert default.equals(ann_return_common_si(returns_df, annualize="periods"))
        expected = (returns_df + 1).prod() ** (12 / 36) - 1
        assert default.to_numpy() == pytest.approx(expected.to_numpy())

    def test_float_periods_per_year(self):
        idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        rets = pd.DataFrame({"A": 0.002}, index=idx)
        ppy = 365.25 / 7
        result = ann_return_common_si(rets, periods_per_year=ppy)["A"]
        assert result == pytest.approx(1.002 ** ppy - 1)


class TestAnnReturnCommonSICalendar:
    @staticmethod
    def _monthly_with_base_row():
        """2020-12-31 base row (NaN), then 24 monthly returns of 1%."""
        idx = pd.date_range("2020-12-31", periods=25, freq="ME")
        rets = pd.DataFrame({"A": 0.01}, index=idx)
        rets.iloc[0] = np.nan
        return rets

    def test_base_row_is_the_default_start(self):
        result = ann_return_common_si(self._monthly_with_base_row(), annualize="calendar")
        # 2020-12-31 to 2022-12-31 is 730 days.
        assert result["A"] == pytest.approx(1.01 ** (24 * 365.25 / 730) - 1)
        assert result.name == "SI"

    def test_explicit_start(self):
        rets = self._monthly_with_base_row().iloc[1:]
        result = ann_return_common_si(rets, annualize="calendar", start="2020-12-31")
        assert result["A"] == pytest.approx(1.01 ** (24 * 365.25 / 730) - 1)

    def test_default_start_is_date_before_common_period(self):
        idx = pd.date_range("2019-12-31", periods=37, freq="ME")
        rets = pd.DataFrame({"A": 0.01, "B": np.nan}, index=idx)
        rets.iloc[0, 0] = np.nan
        rets.loc["2021-01-31":, "B"] = 0.02  # common period opens 2021-01-31
        result = ann_return_common_si(rets, annualize="calendar")
        # Base date 2020-12-31 to 2022-12-31: 730 days, 24 common returns.
        assert result.to_numpy() == pytest.approx(
            [1.01 ** (24 * 365.25 / 730) - 1, 1.02 ** (24 * 365.25 / 730) - 1])

    def test_no_earlier_date_needs_start(self):
        rets = self._monthly_with_base_row().iloc[1:]
        with pytest.raises(ValueError, match="start="):
            ann_return_common_si(rets, annualize="calendar")

    def test_start_after_first_return_raises(self):
        rets = self._monthly_with_base_row()
        with pytest.raises(ValueError, match="after the first common return"):
            ann_return_common_si(rets, annualize="calendar", start="2021-02-15")

    def test_under_one_calendar_year_raises(self, daily_index):
        # 252 sessions passes the periods check, but spans only 354 days.
        rets = pd.DataFrame({"A": 0.001}, index=daily_index)
        ann_return_common_si(rets, periods_per_year=252)
        with pytest.raises(ValueError, match="less than 1 calendar year"):
            ann_return_common_si(rets, annualize="calendar", start="2022-12-30")

    def test_irregular_dates_need_no_frequency(self):
        dates = pd.to_datetime(["2020-02-15", "2020-06-01", "2020-09-30", "2021-03-31", "2021-06-30"])
        rets = pd.DataFrame({"A": 0.05}, index=dates)
        result = ann_return_common_si(rets, annualize="calendar", start="2020-01-01")
        assert result["A"] == pytest.approx(1.05 ** (5 * 365.25 / 546) - 1)

    def test_tz_aware_index_accepts_naive_start(self):
        rets = self._monthly_with_base_row().iloc[1:]
        aware = rets.tz_localize("UTC")
        result = ann_return_common_si(aware, annualize="calendar", start="2020-12-31")
        expected = ann_return_common_si(rets, annualize="calendar", start="2020-12-31")
        assert result.equals(expected)

    def test_unknown_method_raises(self, returns_df):
        with pytest.raises(ValueError, match="annualize"):
            ann_return_common_si(returns_df, annualize="actual")

    def test_start_with_periods_method_raises(self, returns_df):
        with pytest.raises(ValueError, match="calendar"):
            ann_return_common_si(returns_df, start="2019-12-31")

    def test_descending_dates_raise(self):
        rets = self._monthly_with_base_row().iloc[::-1]
        with pytest.raises(ValueError, match="ascending"):
            ann_return_common_si(rets, annualize="calendar")

    def test_non_datetime_index_raises(self):
        rets = pd.DataFrame({"A": [0.01] * 24})
        with pytest.raises(TypeError):
            ann_return_common_si(rets, annualize="calendar", start="2020-12-31")


class TestCompoundedAnnReturn:
    def test_alternating_monthly(self, alternating_monthly_returns):
        result = compounded_ann_return(alternating_monthly_returns, periods_per_year=12)
        assert result == pytest.approx((1.08 ** 6) - 1)


class TestCalendarReturns:
    @staticmethod
    def _monthly(start, periods, value=0.01):
        idx = pd.date_range(start, periods=periods, freq="ME")
        return pd.DataFrame({"A": np.full(periods, value)}, index=idx)

    def test_full_years(self):
        result = calendar_returns(self._monthly("2020-01-31", 24), periods_per_year=12)
        assert list(result.index) == [2020, 2021]
        assert result["A"].tolist() == pytest.approx([1.01 ** 12 - 1] * 2)
        assert result.index.name == "Year"

    def test_partial_first_year_is_left_out(self):
        result = calendar_returns(self._monthly("2019-03-31", 22), periods_per_year=12)
        assert list(result.index) == [2020]

    def test_partial_first_year_that_is_also_final_is_left_out(self):
        result = calendar_returns(self._monthly("2021-03-31", 6), periods_per_year=12)
        assert result.empty

    def test_partial_final_year_is_ytd(self):
        frame = self._monthly("2020-01-31", 18)  # ends Jun 2021
        result = calendar_returns(frame, periods_per_year=12)
        assert list(result.index) == [2020, 2021]
        assert result.loc[2021, "A"] == pytest.approx(1.01 ** 6 - 1)
        assert result.loc[2021, "A"] == pytest.approx(return_ytd(frame)["A"])

    def test_ytd_label(self):
        result = calendar_returns(self._monthly("2020-01-31", 18), periods_per_year=12,
                                  ytd_label="YTD")
        assert list(result.index) == [2020, "YTD"]

    def test_ytd_label_unused_when_final_year_is_complete(self):
        result = calendar_returns(self._monthly("2020-01-31", 24), periods_per_year=12,
                                  ytd_label="YTD")
        assert list(result.index) == [2020, 2021]

    def test_late_launch_is_nan_in_launch_year(self):
        frame = self._monthly("2019-01-31", 36)
        frame["B"] = np.nan
        frame.loc["2020-03-31":, "B"] = 0.02
        result = calendar_returns(frame, periods_per_year=12)
        assert np.isnan(result.loc[2020, "B"])
        assert result.loc[2021, "B"] == pytest.approx(1.02 ** 12 - 1)
        assert result.loc[2020, "A"] == pytest.approx(1.01 ** 12 - 1)

    def test_stale_column_is_nan_in_unfinished_year(self):
        frame = self._monthly("2019-01-31", 36)
        frame["B"] = 0.01
        frame.loc["2021-10-31":, "B"] = np.nan  # stopped reporting in 2021
        result = calendar_returns(frame, periods_per_year=12)
        assert np.isnan(result.loc[2021, "B"])
        assert result.loc[2020, "B"] == pytest.approx(1.01 ** 12 - 1)

    def test_internal_gap_is_skipped(self):
        frame = self._monthly("2020-01-31", 12)
        frame.iloc[5, 0] = np.nan
        result = calendar_returns(frame, periods_per_year=12)
        assert result.loc[2020, "A"] == pytest.approx(1.01 ** 11 - 1)

    def test_series_in_series_out(self):
        series = self._monthly("2020-01-31", 24)["A"]
        result = calendar_returns(series, periods_per_year=12)
        assert isinstance(result, pd.Series)
        assert result.name == "A"
        assert len(result) == 2

    def test_infers_monthly_frequency(self):
        frame = self._monthly("2019-03-31", 30)
        assert calendar_returns(frame).equals(calendar_returns(frame, periods_per_year=12))

    def test_quarterly(self):
        idx = pd.date_range("2020-03-31", periods=8, freq="QE")
        frame = pd.DataFrame({"A": 0.02}, index=idx)
        result = calendar_returns(frame)
        assert list(result.index) == [2020, 2021]
        assert result["A"].tolist() == pytest.approx([1.02 ** 4 - 1] * 2)

    def test_daily_year_edges(self):
        # 2021 opens on a Monday, Jan 4 (Jan 1 holiday, then a weekend): still a full year.
        idx = pd.bdate_range("2021-01-04", "2022-06-30")
        frame = pd.DataFrame({"A": 0.001}, index=idx)
        result = calendar_returns(frame, ytd_label="YTD")
        assert list(result.index) == [2021, "YTD"]
        n_2021 = int((idx.year == 2021).sum())
        assert result.loc[2021, "A"] == pytest.approx(1.001 ** n_2021 - 1)

    def test_daily_mid_january_start_is_partial(self):
        idx = pd.bdate_range("2021-01-15", "2022-12-30")
        frame = pd.DataFrame({"A": 0.001}, index=idx)
        assert list(calendar_returns(frame).index) == [2022]

    def test_empty(self, monthly_index):
        empty = pd.DataFrame(index=monthly_index[:0], columns=["A"], dtype=float)
        result = calendar_returns(empty)
        assert result.empty
        assert list(result.columns) == ["A"]


class TestAverageGainLoss:
    def test_series(self):
        r = pd.Series([0.02, -0.01, 0.04, -0.03, 0.0])
        assert average_gain(r) == pytest.approx(0.03)
        assert average_loss(r) == pytest.approx(-0.02)

    def test_zero_and_nan_are_excluded(self):
        r = pd.Series([0.02, 0.0, np.nan, -0.04])
        assert average_gain(r) == pytest.approx(0.02)
        assert average_loss(r) == pytest.approx(-0.04)

    def test_no_losses_is_nan(self):
        assert np.isnan(average_loss(pd.Series([0.01, 0.02])))

    def test_dataframe_per_column(self):
        df = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [-0.01, -0.03, 0.05]})
        assert average_gain(df).tolist() == pytest.approx([0.02, 0.05])
        assert average_loss(df).tolist() == pytest.approx([-0.02, -0.02])


class TestCumgrowth:
    def test_basic(self, returns_df):
        result = cumgrowth(returns_df)
        assert isinstance(result, pd.DataFrame)
        assert (result.iloc[-1] > 0).all()

    def test_custom_init(self, returns_df):
        result = cumgrowth(returns_df, init_value=100)
        assert result.iloc[0].eq(100).all() or result.iloc[0].ge(99).all()


class TestAnnualizedRollingReturn:
    def test_basic(self, returns_df):
        result = annualized_rolling_return(returns_df, window=12, periods_per_year=12)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(returns_df)

    def test_window_too_large_warns(self, returns_df):
        with pytest.warns(UserWarning):
            annualized_rolling_return(returns_df, window=100, periods_per_year=12)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            annualized_rolling_return([0.01, 0.02], window=1, periods_per_year=12)
