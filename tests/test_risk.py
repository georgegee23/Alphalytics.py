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
    compare_drawdowns,
    time_to_recovery_table,
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
        result = downside_variance(single, ddof=1)
        if isinstance(result, pd.Series):
            assert result.isna().all()
        else:
            assert np.isnan(result)

    def test_series_input(self, returns_df):
        result = downside_variance(returns_df["StratA"])
        assert isinstance(result, (float, np.floating))

    def test_gap_uses_finite_count(self, gapped_downside_returns):
        result = downside_variance(gapped_downside_returns, ddof=0)
        assert result == pytest.approx(0.000325)


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

    def test_opening_loss_is_a_drawdown(self, opening_loss_returns):
        assert max_drawdown(opening_loss_returns) == pytest.approx(-0.505)
        episodes = top_drawdowns(opening_loss_returns, n=1, periods_per_year=12)
        assert episodes.loc[0, "Depth"] == pytest.approx(-0.505)

    def test_internal_gap_is_one_episode(self):
        idx = pd.date_range("2020-01-31", periods=5, freq="ME")
        series = pd.Series([0.10, -0.20, np.nan, -0.10, 0.50], index=idx)
        episodes = top_drawdowns(series, n=5, periods_per_year=12)
        assert len(episodes) == 1
        assert episodes.loc[0, "Depth"] == pytest.approx(-0.28)
        assert episodes.loc[0, "Peak"] == idx[0].strftime("%Y-%m-%d")
        assert episodes.loc[0, "Trough"] == idx[3].strftime("%Y-%m-%d")

    def test_top_drawdowns_infers_monthly_frequency(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        series = pd.Series([0.0, -0.10, -0.20, 0.40], index=idx)
        inferred = top_drawdowns(series, n=1)
        monthly = top_drawdowns(series, n=1, periods_per_year=12)
        daily = top_drawdowns(series, n=1, periods_per_year=252)
        assert inferred.loc[0, "Volatility (Ann)"] == pytest.approx(monthly.loc[0, "Volatility (Ann)"])
        assert inferred.loc[0, "Volatility (Ann)"] != pytest.approx(daily.loc[0, "Volatility (Ann)"])

    def test_drawdown_columns_are_independent(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        frame = pd.DataFrame({
            "A": [-0.5, 0.1, -0.1, 0.2],
            "B": [-0.5, np.nan, -0.1, 0.2],
        }, index=idx)
        alone = to_drawdowns(frame["A"])
        assert np.allclose(to_drawdowns(frame)["A"], alone, equal_nan=True)


class TestRecovery:
    def _benchmark(self):
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        bench = pd.Series([0.0, -0.30, -0.10, 0.05, 0.80, 0.01], index=idx, name="Bench")
        return idx, bench

    def test_never_underwater_is_zero(self):
        idx, bench = self._benchmark()
        strat = pd.Series([0.01] * len(idx), index=idx, name="Up")
        table = time_to_recovery_table(strat, bench, n=1)
        assert table.iloc[0, 0] == pytest.approx(0.0)

    def test_recovery_before_benchmark_trough(self):
        idx, bench = self._benchmark()
        strat = pd.Series([0.0, -0.05, 0.10, 0.02, 0.02, 0.01], index=idx, name="Early")
        table = time_to_recovery_table(strat, bench, n=1)
        # Feb is the breach (period 1) and March is back above the peak (period 2).
        assert table.iloc[0, 0] == pytest.approx(2.0)
        comparison = compare_drawdowns(strat, bench, n=1, periods_per_year=12)
        depth = next(iter(comparison.values()))["Depth"].iloc[0]
        assert depth == pytest.approx(-0.05)

    def test_recovery_after_benchmark_trough(self):
        idx, bench = self._benchmark()
        strat = pd.Series([0.0, -0.10, 0.0, 0.05, 0.10, 0.01], index=idx, name="Slow")
        table = time_to_recovery_table(strat, bench, n=1)
        # Breach in Feb, inside the window. Back above the peak in May (period 4).
        assert table.iloc[0, 0] == pytest.approx(4.0)

    def test_drawdown_after_the_episode_is_not_a_recovery(self):
        idx = pd.date_range("2020-01-31", periods=10, freq="ME")
        bench = pd.Series([0.0, -0.30, -0.10, 0.05, 0.80] + [0.01] * 5, index=idx, name="Bench")
        # Rises through the benchmark's Feb-Mar decline, then crashes in June.
        strat = pd.Series([0.0, 0.02, 0.02, 0.02, 0.02, -0.40, 0.10, 0.50, 0.10, 0.10],
            index=idx, name="Late")
        table = time_to_recovery_table(strat, bench, n=1)
        assert table.iloc[0, 0] == pytest.approx(0.0)

    def test_no_returns_in_window_is_nan(self):
        idx, bench = self._benchmark()
        strat = pd.Series([np.nan, np.nan, np.nan, 0.01, 0.01, 0.01], index=idx, name="After")
        row = next(iter(compare_drawdowns(strat, bench, n=1, periods_per_year=12).values())).iloc[0]
        assert np.isnan(row["Depth"])
        assert np.isnan(row["Time to Recovery"])
        assert np.isnan(row["Periods Underwater"])

    def test_inception_episode_ignores_history_before_benchmark(self):
        idx = pd.date_range("2020-01-31", periods=8, freq="ME")
        # Benchmark starts in April with a loss, so the episode opens at inception.
        bench = pd.Series([np.nan, np.nan, np.nan, -0.10, -0.05, 0.20, 0.01, 0.01],
            index=idx, name="Bench")
        strat = pd.Series([-0.30, -0.30, -0.30] + [0.01] * 5, index=idx, name="Older")
        row = next(iter(compare_drawdowns(strat, bench, n=1, periods_per_year=12).values())).iloc[0]
        assert row["Peak"] == idx[3].strftime("%Y-%m-%d")
        assert row["Depth"] == pytest.approx(0.0)
        assert row["Time to Recovery"] == pytest.approx(0.0)
        assert row["Periods Underwater"] == 0


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
