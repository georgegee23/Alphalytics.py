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


def test_negative_down_capture_has_no_overall_capture():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    bench = pd.Series([0.03, -0.02] * 12, index=idx, name="Bench")
    convex = pd.Series(np.where(bench > 0, 0.3 * bench, -0.2 * bench), index=idx, name="Convex")
    table = evaluate_asymmetry(convex, bench)
    assert table.loc["Convex", "Down Capture"] < 0
    assert np.isnan(table.loc["Convex", "Overall Capture"])
