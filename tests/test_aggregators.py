"""Tests for performance tables."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.aggregators import performance_table


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
