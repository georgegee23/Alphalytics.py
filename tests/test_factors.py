"""Tests for alphalytics.factors — quantiles, turnover, IC."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.factors.quantiles import (
    to_quantiles,
    compute_quantile_returns,
    fwd_quantile_stats,
)
from alphalytics.factors.turnover import (
    calculate_autocorrelation,
    compute_factor_autocorr,
    compute_quantile_turnover,
    compute_quantiles_turnover,
)
from alphalytics.factors.ic import cs_spearmanr, ts_spearmanr, factor_decay


class TestToQuantiles:
    def test_basic(self, factor_df):
        result = to_quantiles(factor_df, n_quantiles=5, axis=1)
        assert result.shape == factor_df.shape
        unique_vals = result.stack().dropna().unique()
        assert set(unique_vals).issubset({1, 2, 3, 4, 5})

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            to_quantiles("not a df", n_quantiles=3)

    def test_bad_n(self, factor_df):
        with pytest.raises(ValueError):
            to_quantiles(factor_df, n_quantiles=-1)

    def test_bad_axis(self, factor_df):
        with pytest.raises(ValueError):
            to_quantiles(factor_df, n_quantiles=3, axis=2)

    def test_default_axis_is_cross_sectional(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        factors = pd.DataFrame({"A": [1.0, 3.0, 5.0], "B": [2.0, 4.0, 6.0]}, index=idx)
        assert to_quantiles(factors, n_quantiles=2).equals(
            to_quantiles(factors, n_quantiles=2, axis=1)
        )

    def test_collapsed_bins_are_nan(self):
        idx = pd.date_range("2020-01-31", periods=1, freq="ME")
        factors = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=idx, columns=list("ABCD"))
        result = to_quantiles(factors, n_quantiles=4)
        assert result.isna().all().all()


class TestComputeQuantileReturns:
    def test_basic(self, factor_df, asset_returns):
        quantiles = to_quantiles(factor_df, n_quantiles=3, axis=1)
        result = compute_quantile_returns(quantiles, asset_returns)
        assert isinstance(result, pd.DataFrame)
        assert all(c.startswith("Q") for c in result.columns)

    def test_assignments_are_lagged_one_period(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        quantiles = pd.DataFrame(
            {"A": [1, 2, 1], "B": [2, 1, 2]},
            index=idx,
        )
        returns = pd.DataFrame(
            {"A": [100.0, 1.0, 2.0], "B": [100.0, 2.0, 1.0]},
            index=idx,
        )
        result = compute_quantile_returns(quantiles, returns)
        # Quantiles from t0 earn the returns at t1: Q1 -> A -> 1, Q2 -> B -> 2.
        assert result.loc[idx[1], "Q1"] == pytest.approx(1.0)
        assert result.loc[idx[1], "Q2"] == pytest.approx(2.0)
        assert idx[0] not in result.index

    def test_mismatched_shape_raises(self, factor_df, monthly_index):
        quantiles = to_quantiles(factor_df, n_quantiles=3, axis=1)
        bad_returns = pd.DataFrame(
            {"X": np.zeros(len(monthly_index))}, index=monthly_index
        )
        with pytest.raises(ValueError):
            compute_quantile_returns(quantiles, bad_returns)


class TestFwdQuantileStats:
    def test_basic(self, factor_df, asset_returns):
        quantiles = to_quantiles(factor_df, n_quantiles=3, axis=1)
        result = fwd_quantile_stats(asset_returns, quantiles, forward_periods=1)
        assert isinstance(result, pd.DataFrame)
        assert "Return" in result.columns
        assert "Risk-Adjusted Return" in result.columns

    def test_zero_std_gives_nan_not_inf(self, monthly_index):
        """When all forward returns are identical, risk-adjusted should be NaN."""
        cols = ["A", "B", "C"]
        # Constant returns → zero std in forward returns
        rets = pd.DataFrame(0.01, index=monthly_index, columns=cols)
        factors = pd.DataFrame(
            np.tile([1, 2, 3], (len(monthly_index), 1)),
            index=monthly_index,
            columns=cols,
        )
        result = fwd_quantile_stats(rets, factors, forward_periods=1)
        risk_adj = result["Risk-Adjusted Return"]
        # Should be NaN where std is 0, never inf
        assert not np.isinf(risk_adj.dropna()).any()


class TestAutocorrelation:
    def test_basic(self, factor_df):
        result = calculate_autocorrelation(factor_df, lag=1)
        assert isinstance(result, pd.Series)
        assert len(result) == factor_df.shape[1]

    def test_insufficient_data(self, monthly_index):
        short = pd.DataFrame({"A": [1.0, 2.0]}, index=monthly_index[:2])
        result = calculate_autocorrelation(short, lag=1)
        assert result.isna().all()

    def test_lag_keeps_calendar_gaps(self, monthly_index):
        values = [1.0, 2.0, np.nan, 4.0, 5.0]
        frame = pd.DataFrame({"A": values}, index=monthly_index[:5])
        result = calculate_autocorrelation(frame, lag=1)
        assert result["A"] == pytest.approx(frame["A"].autocorr(lag=1))
        assert result["A"] != pytest.approx(frame["A"].dropna().autocorr(lag=1))

    def test_factor_autocorr_range(self, factor_df):
        result = compute_factor_autocorr(factor_df, max_lag=5)
        assert isinstance(result, pd.Series)
        assert len(result) == 4  # lags 1..4


class TestQuantileTurnover:
    def test_basic(self, factor_df):
        quantiles = to_quantiles(factor_df, n_quantiles=3, axis=1)
        result = compute_quantile_turnover(quantiles, target_quantile=1)
        assert isinstance(result, pd.Series)
        assert result.dropna().between(0, 1).all()

    def test_entries_count(self):
        book = pd.DataFrame({"A": [1, 1], "B": [2, 1], "C": [2, 1]})
        result = compute_quantile_turnover(book, target_quantile=1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(2 / 3)

    def test_zero_holdings_gives_nan(self, monthly_index):
        """When no assets are in target quantile, turnover should be NaN."""
        # All assets in quantile 2 — no assets in quantile 1
        data = pd.DataFrame(
            2, index=monthly_index[:5], columns=["A", "B", "C"]
        )
        result = compute_quantile_turnover(data, target_quantile=1)
        assert result.isna().all(), "Empty quantile → NaN turnover, not inf"

    def test_all_quantiles(self, factor_df):
        quantiles = to_quantiles(factor_df, n_quantiles=3, axis=1)
        result = compute_quantiles_turnover(quantiles)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] >= 1


class TestIC:
    def test_cs_spearmanr(self, factor_df, asset_returns):
        result = cs_spearmanr(factor_df, asset_returns)
        assert isinstance(result, pd.Series)
        # Correlations should be in [-1, 1]
        vals = result.dropna()
        assert (vals >= -1).all() and (vals <= 1).all()

    def test_ts_spearmanr(self, factor_df, asset_returns):
        result = ts_spearmanr(factor_df, asset_returns)
        assert isinstance(result, pd.Series)

    def test_factor_decay_is_forward(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(
            rng.normal(0, 0.05, (len(idx), 6)),
            index=idx,
            columns=list("ABCDEF"),
        )
        # Known at t, equal to the return earned over the next period.
        factor = returns.shift(-1)
        decay = factor_decay(factor, returns, max_periods=1)
        assert decay.loc[1, "IC"] == pytest.approx(1.0)
