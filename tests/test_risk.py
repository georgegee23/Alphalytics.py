"""Tests for alphalytics.returns.risk — volatility, drawdowns."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.risk import (
    annual_std,
    downside_variance,
    to_drawdowns,
    max_drawdown,
    top_drawdowns,
    average_drawdown,
    average_drawdown_duration,
)


class TestAnnualStd:
    def test_basic(self, returns_df):
        result = annual_std(returns_df, periods_per_year=12)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_flat_returns_near_zero(self, flat_returns):
        result = annual_std(flat_returns, periods_per_year=12)
        assert np.allclose(result, 0, atol=1e-10)


class TestDownsideVariance:
    def test_basic(self, returns_df):
        result = downside_variance(returns_df)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()

    def test_no_downside(self, positive_only_returns):
        result = downside_variance(positive_only_returns, mar=0.0)
        assert (result == 0).all()

    def test_single_obs_returns_nan(self, monthly_index):
        single = pd.Series([0.01], index=monthly_index[:1])
        result = downside_variance(single)
        if isinstance(result, pd.Series):
            assert result.isna().all()
        else:
            assert np.isnan(result)

    def test_series_input(self, returns_df):
        result = downside_variance(returns_df["StratA"])
        assert isinstance(result, (float, np.floating))


class TestDrawdowns:
    def test_drawdown_always_nonpositive(self, returns_df):
        dd = to_drawdowns(returns_df)
        assert (dd.dropna() <= 0).all().all()

    def test_max_drawdown_nonpositive(self, returns_df):
        mdd = max_drawdown(returns_df)
        assert (mdd <= 0).all()

    def test_no_drawdown_for_positive_returns(self, positive_only_returns):
        mdd = max_drawdown(positive_only_returns)
        assert (mdd == 0).all()

    def test_top_drawdowns_returns_dataframe(self, returns_df):
        result = top_drawdowns(returns_df["StratA"], n=3, periods_per_year=12)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3

    def test_top_drawdowns_empty_for_positive(self, positive_only_returns):
        result = top_drawdowns(positive_only_returns["AlwaysUp"], n=3, periods_per_year=12)
        assert result.empty

    def test_top_drawdowns_requires_series(self, returns_df):
        with pytest.raises(TypeError):
            top_drawdowns(returns_df, n=3)


class TestAverageDrawdown:
    def test_basic(self, returns_df):
        result = average_drawdown(returns_df)
        assert isinstance(result, pd.Series)

    def test_never_underwater(self, positive_only_returns):
        result = average_drawdown(positive_only_returns)
        assert (result == 0).all()


class TestAverageDrawdownDuration:
    def test_basic(self, returns_df):
        result = average_drawdown_duration(returns_df)
        assert isinstance(result, pd.Series)

    def test_never_underwater(self, positive_only_returns):
        result = average_drawdown_duration(positive_only_returns)
        assert (result == 0).all()
