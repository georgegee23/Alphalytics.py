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
