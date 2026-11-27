"""Tests for performance tables."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.aggregators import performance_table, evaluate_asymmetry


def test_quarterly_one_month_column_is_nan():
    idx = pd.date_range("2015-03-31", periods=40, freq="QE")
    returns = pd.DataFrame(
        {"A": np.full(len(idx), 0.02), "B": np.full(len(idx), 0.01)},
        index=idx,
    )
    table = performance_table(returns)
    assert table["1M"].isna().all()
    assert table["3M"].to_numpy() == pytest.approx([0.02, 0.01])


def test_one_strategy_hole_does_not_shorten_the_other():
    idx = pd.date_range("2019-01-31", periods=14, freq="ME")
    a = np.array([0.0, 0.0] + [0.01] * 12)
    b = a.copy()
    b[-1] = np.nan
    frame = pd.DataFrame({"A": a, "B": b}, index=idx)
    table = performance_table(frame, periods_per_year=12)
    assert table.loc["A", "1 Year"] == pytest.approx((1.01 ** 12) - 1)


def test_short_history_columns_keep_their_labels():
    idx = pd.date_range("2019-01-31", periods=72, freq="ME")
    frame = pd.DataFrame({"A": 0.01, "B": 0.005}, index=idx)
    table = performance_table(frame)
    assert list(table.columns) == ["1M", "3M", "YTD", "1 Year", "3 Year", "5 Year", "10 Year", "SI"]
    assert table["10 Year"].isna().all()


def test_late_launch_is_blank_beyond_its_track_record():
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    new = np.full(len(idx), np.nan)
    new[-24:] = 0.01
    frame = pd.DataFrame({"OLD": np.full(len(idx), 0.01), "NEW": new}, index=idx)
    table = performance_table(frame)
    assert table.loc["NEW", "1 Year"] == pytest.approx((1.01 ** 12) - 1)
    assert table.loc["NEW", ["3 Year", "5 Year", "10 Year"]].isna().all()
    assert table.loc["OLD", "10 Year"] == pytest.approx((1.01 ** 12) - 1)


class TestPerformanceTableCalendar:
    @staticmethod
    def _daily():
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        return pd.DataFrame({"A": 0.0004, "B": 0.0002}, index=idx)

    def test_columns_and_windows(self):
        rets = self._daily()
        table = performance_table(rets, annualize="calendar", start="2014-12-31")
        assert list(table.columns) == ["1M", "3M", "YTD", "1 Year", "3 Year", "5 Year", "10 Year", "SI"]
        dec = int(((rets.index.year == 2024) & (rets.index.month == 12)).sum())
        assert table.loc["A", "1M"] == pytest.approx(1.0004 ** dec - 1)
        n_2024 = int((rets.index.year == 2024).sum())
        assert table.loc["A", "1 Year"] == pytest.approx(1.0004 ** n_2024 - 1)
        days = (rets.index[-1] - pd.Timestamp("2014-12-31")).days
        assert table.loc["A", "SI"] == pytest.approx(1.0004 ** (len(rets) * 365.25 / days) - 1)

    def test_calendar_si_without_base_date_raises(self):
        with pytest.raises(ValueError, match="start="):
            performance_table(self._daily(), annualize="calendar")

    def test_short_history_blanks_si(self):
        rets = self._daily().iloc[-200:]
        table = performance_table(rets, annualize="calendar", start="2024-03-01")
        assert table["SI"].isna().all()
        assert table["1 Year"].isna().all()

    def test_quarterly_has_no_one_month(self):
        idx = pd.date_range("2015-03-31", periods=40, freq="QE")
        rets = pd.DataFrame({"A": 0.02}, index=idx)
        table = performance_table(rets, annualize="calendar", start="2014-12-31")
        assert np.isnan(table.loc["A", "1M"])
        assert table.loc["A", "3M"] == pytest.approx(0.02)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="annualize"):
            performance_table(self._daily(), annualize="actual")
        with pytest.raises(ValueError, match="calendar"):
            performance_table(self._daily(), start="2014-12-31")


def test_negative_down_capture_has_no_overall_capture():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    bench = pd.Series([0.03, -0.02] * 12, index=idx, name="Bench")
    convex = pd.Series(np.where(bench > 0, 0.3 * bench, -0.2 * bench), index=idx, name="Convex")
    table = evaluate_asymmetry(convex, bench)
    assert table.loc["Convex", "Down Capture"] < 0
    assert np.isnan(table.loc["Convex", "Overall Capture"])
