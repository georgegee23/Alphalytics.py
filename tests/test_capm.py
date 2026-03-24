"""Tests for alphalytics.returns.capm — beta, CAPM, rolling beta."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.capm import (
    beta,
    bull_bear_beta,
    compute_capm,
    rolling_beta,
)


class TestBeta:
    def test_basic(self, returns_df, benchmark_series):
        result = beta(returns_df, benchmark_series)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.notna().all()

    def test_zero_variance_benchmark(self, returns_df, monthly_index):
        flat = pd.Series(0.005, index=monthly_index, name="flat")
        result = beta(returns_df, flat)
        assert result.isna().all(), "Flat benchmark → NaN beta"

    def test_single_obs_returns_nan(self, monthly_index):
        rets = pd.DataFrame({"A": [0.01]}, index=monthly_index[:1])
        bench = pd.Series([0.02], index=monthly_index[:1])
        result = beta(rets, bench)
        assert result.isna().all()

    def test_misaligned_index_raises(self, returns_df, monthly_index):
        bad_bench = pd.Series(0.01, index=pd.date_range("2000-01-31", periods=36, freq="ME"))
        with pytest.raises(ValueError):
            beta(returns_df, bad_bench)


class TestBullBearBeta:
    def test_returns_dataframe(self, returns_df, benchmark_series):
        result = bull_bear_beta(returns_df, benchmark_series)
        assert isinstance(result, pd.DataFrame)
        assert "Bull Beta" in result.columns
        assert "Bear Beta" in result.columns

    def test_wrong_types(self, returns_df, benchmark_series):
        with pytest.raises(TypeError):
            bull_bear_beta(returns_df["StratA"], benchmark_series)
        with pytest.raises(TypeError):
            bull_bear_beta(returns_df, returns_df)


class TestComputeCAPM:
    def test_basic(self, returns_df, benchmark_series):
        result = compute_capm(returns_df, benchmark_series, periods_per_year=12)
        assert isinstance(result, pd.DataFrame)
        assert "Beta" in result.columns
        assert "Annualized Alpha" in result.columns
        assert len(result) == 2

    def test_zero_variance_benchmark(self, returns_df, flat_benchmark):
        result = compute_capm(returns_df, flat_benchmark, periods_per_year=12)
        assert result["Beta"].isna().all(), "Flat benchmark → NaN beta"

    def test_no_benchmark_uses_mean(self, returns_df):
        result = compute_capm(returns_df, periods_per_year=12)
        assert isinstance(result, pd.DataFrame)

    def test_empty_raises(self, monthly_index):
        empty_df = pd.DataFrame(index=monthly_index[:0], columns=["A"])
        with pytest.raises(ValueError):
            compute_capm(empty_df)


class TestRollingBeta:
    def test_basic(self, returns_df, benchmark_series):
        result = rolling_beta(returns_df, benchmark_series, window=12)
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == returns_df.columns.tolist()

    def test_flat_benchmark_gives_nan(self, returns_df, flat_benchmark):
        result = rolling_beta(returns_df, flat_benchmark, window=12)
        assert result.isna().all().all(), "Flat benchmark windows → NaN"
