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
    active_risk,
    information_ratio,
    rolling_information_ratio,
    rolling_active_return,
    rolling_hit_rate,
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


class TestBullHitRate:
    def test_gap_is_not_a_miss(self, bull_gap_strategy, bull_gap_benchmark):
        assert bull_hit_rate(bull_gap_strategy, bull_gap_benchmark) == pytest.approx(1.0)
        assert hit_rate(bull_gap_strategy, bull_gap_benchmark) == pytest.approx(1.0)


class TestRollingActiveReturn:
    def test_internal_gap_is_not_compressed(self):
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        strat = pd.Series([0.01, 0.02, np.nan, 0.03, 0.04, 0.05], index=idx, name="S")
        bench = pd.Series(0.0, index=idx, name="B")
        result = rolling_active_return(strat, bench, window=3, periods_per_year=12)
        assert result.loc[idx[-1]] == pytest.approx(0.04 * 12)
        assert idx[2] not in result.index or pd.isna(result.loc[idx[2]])
        # The compressed calendar would score mean(0.01, 0.02, 0.03) on the
        # date of the first valid point after the gap.
        assert idx[3] not in result.index or pd.isna(result.loc[idx[3]])


class TestRollingHitRate:
    def test_internal_gap_is_not_a_miss(self):
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        strat = pd.Series([0.02, 0.02, np.nan, -0.01, 0.02, 0.02], index=idx, name="S")
        bench = pd.Series(0.0, index=idx, name="B")
        result = rolling_hit_rate(strat, bench, window=3)
        assert result.loc[idx[-1]] == pytest.approx(2 / 3)
        assert idx[3] not in result.index or pd.isna(result.loc[idx[3]])


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


class TestActiveRisk:
    def test_basic(self, returns_df, benchmark_series):
        result = active_risk(returns_df["StratA"], benchmark_series)
        assert isinstance(result, float)
        assert result >= 0

    def test_identical_returns_zero(self, benchmark_series):
        result = active_risk(benchmark_series, benchmark_series)
        assert result == 0.0

    def test_too_few_obs(self, monthly_index):
        single = pd.Series([0.01], index=monthly_index[:1])
        bench = pd.Series([0.02], index=monthly_index[:1])
        result = active_risk(single, bench)
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
        # "Overall Capture" is consumed by plot_capture_hit_rate — keep it named.
        assert "Overall Capture" in result.columns
        assert "Spread" in result.columns

    def test_overall_capture_undefined_when_down_capture_not_positive(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        bench = pd.Series([0.03, -0.02] * 12, index=idx, name="Bench")
        frame = pd.DataFrame({
            # Gains in both regimes: 30% up capture, -20% down capture.
            "Convex": np.where(bench > 0, 0.3 * bench, -0.2 * bench),
            "Index": bench.to_numpy(),
        }, index=idx)
        table = capture_ratios(frame, bench)
        assert table.loc["Convex", "Down Capture"] < 0
        assert np.isnan(table.loc["Convex", "Overall Capture"])
        assert table.loc["Convex", "Spread"] > 0
        assert table.loc["Index", "Overall Capture"] == pytest.approx(1.0)

    def test_no_down_market(self, returns_df, monthly_index):
        bench = pd.Series(0.01, index=monthly_index, name="up_only")
        result = down_capture(returns_df, bench)
        assert result.isna().all()

    def test_one_strategy_gap_does_not_change_the_other(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        bench = pd.Series([-0.10, -0.20, 0.10, 0.05], index=idx, name="Benchmark")
        frame = pd.DataFrame({
            "A": [-0.05, -0.10, 0.10, 0.05],
            "B": [-0.05, np.nan, 0.10, 0.05],
        }, index=idx)
        alone = down_capture(frame["A"], bench)
        assert down_capture(frame, bench)["A"] == pytest.approx(alone)
        named = frame["A"].rename("bench")
        assert down_capture(named, bench) == pytest.approx(alone)

    def test_total_loss_is_nan(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        bench = pd.Series([-0.10, -0.20, 0.10], index=idx, name="Benchmark")
        dead = pd.Series([-1.0, -0.10, 0.10], index=idx, name="Dead")
        assert np.isnan(down_capture(dead, bench))
