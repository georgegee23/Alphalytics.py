"""Tests for alphalytics.returns.relative — hit rates, capture ratios, IR."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.relative import (
    hit_rate,
    bull_hit_rate,
    bear_hit_rate,
    win_loss_ratio,
    active_return,
    tracking_error,
    information_ratio,
    rolling_information_ratio,
    hit_rates,
    up_capture,
    down_capture,
    capture_ratios,
)


class TestHitRate:
    def test_series_returns_float(self, returns_df, benchmark_series):
        result = hit_rate(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_dataframe_returns_series(self, returns_df, benchmark_series):
        result = hit_rate(returns_df, benchmark_series)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_empty_returns_nan(self, monthly_index):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        bench = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        result = hit_rate(empty, bench)
        assert np.isnan(result)


class TestBullBearHitRate:
    def test_bull(self, returns_df, benchmark_series):
        result = bull_hit_rate(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)

    def test_bear(self, returns_df, benchmark_series):
        result = bear_hit_rate(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)


class TestWinLossRatio:
    def test_basic(self, returns_df, benchmark_series):
        result = win_loss_ratio(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)
        assert result >= 0

    def test_always_wins(self, monthly_index):
        strat = pd.Series(0.02, index=monthly_index)
        bench = pd.Series(0.01, index=monthly_index)
        result = win_loss_ratio(strat, bench)
        assert result == np.inf


class TestActiveReturn:
    def test_basic(self, returns_df, benchmark_series):
        result = active_return(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)

    def test_dataframe(self, returns_df, benchmark_series):
        result = active_return(returns_df, benchmark_series)
        assert isinstance(result, pd.Series)


class TestTrackingError:
    def test_basic(self, returns_df, benchmark_series):
        result = tracking_error(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)
        assert result >= 0

    def test_identical_returns_zero(self, benchmark_series):
        result = tracking_error(benchmark_series, benchmark_series)
        assert result == 0.0

    def test_too_few_obs(self, monthly_index):
        single = pd.Series([0.01], index=monthly_index[:1])
        bench = pd.Series([0.02], index=monthly_index[:1])
        result = tracking_error(single, bench)
        assert np.isnan(result)


class TestInformationRatio:
    def test_basic(self, returns_df, benchmark_series):
        result = information_ratio(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)

    def test_identical_returns_nan(self, benchmark_series):
        result = information_ratio(benchmark_series, benchmark_series)
        assert np.isnan(result), "Zero TE → NaN IR"


class TestRollingIR:
    def test_returns_dataframe(self, returns_df, benchmark_series):
        result = rolling_information_ratio(returns_df, benchmark_series, window=12)
        assert isinstance(result, pd.DataFrame)

    def test_series_input(self, returns_df, benchmark_series):
        result = rolling_information_ratio(returns_df["StratA"], benchmark_series, window=12)
        assert isinstance(result, pd.DataFrame)


class TestHitRates:
    def test_basic(self, returns_df, benchmark_series):
        result = hit_rates(returns_df, benchmark_series)
        assert isinstance(result, pd.DataFrame)
        assert "Hit Rate" in result.columns

    def test_wrong_types(self, returns_df, benchmark_series):
        with pytest.raises(TypeError):
            hit_rates(returns_df["StratA"], benchmark_series)


class TestCaptureRatios:
    def test_up_capture(self, returns_df, benchmark_series):
        result = up_capture(returns_df, benchmark_series)
        assert isinstance(result, pd.Series)

    def test_down_capture(self, returns_df, benchmark_series):
        result = down_capture(returns_df, benchmark_series)
        assert isinstance(result, pd.Series)

    def test_capture_ratios_table(self, returns_df, benchmark_series):
        result = capture_ratios(returns_df, benchmark_series)
        assert isinstance(result, pd.DataFrame)
        assert "Up Capture" in result.columns
        assert "Down Capture" in result.columns
        assert "Capture Spread" in result.columns

    def test_no_down_market(self, returns_df, monthly_index):
        bench = pd.Series(0.01, index=monthly_index, name="up_only")
        result = down_capture(returns_df, bench)
        assert result.isna().all()
