"""Tests for alphalytics.returns.metrics — return calculations."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.metrics import (
    return_n,
    return_ytd,
    ann_return,
    ann_return_common_si,
    compounded_ann_return,
    cumgrowth,
    annualized_rolling_return,
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


class TestAnnReturnCommonSI:
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


class TestCompoundedAnnReturn:
    def test_alternating_monthly(self, alternating_monthly_returns):
        result = compounded_ann_return(alternating_monthly_returns, periods_per_year=12)
        assert result == pytest.approx((1.08 ** 6) - 1)


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
