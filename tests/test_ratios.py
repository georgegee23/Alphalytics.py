"""Tests for alphalytics.returns.ratios — Sharpe, Sortino, Calmar, Omega."""

import numpy as np
import pandas as pd
import pytest

from alphalytics.returns.ratios import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from alphalytics.returns.relative import omega_ratio


class TestSharpeRatio:
    def test_basic(self, returns_df):
        result = sharpe_ratio(returns_df, periods_per_year=12)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.notna().all()

    def test_series_input(self, returns_df):
        result = sharpe_ratio(returns_df["StratA"], periods_per_year=12)
        assert isinstance(result, (float, np.floating))

    def test_zero_volatility_returns_nan(self, flat_returns):
        result = sharpe_ratio(flat_returns, periods_per_year=12)
        assert result.isna().all(), "Zero-vol returns should produce NaN, not inf"

    def test_zero_vol_series_returns_nan(self, monthly_index):
        flat = pd.Series(0.01, index=monthly_index, name="flat")
        result = sharpe_ratio(flat, periods_per_year=12)
        assert np.isnan(result), "Zero-vol Series should return NaN"

    def test_positive_for_positive_mean(self, monthly_index):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.02, 0.01, len(monthly_index)), index=monthly_index)
        assert sharpe_ratio(rets, periods_per_year=12) > 0


class TestSortinoRatio:
    def test_basic(self, returns_df):
        result = sortino_ratio(returns_df, periods_per_year=12)
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_no_downside_returns_nan(self, positive_only_returns):
        result = sortino_ratio(positive_only_returns, periods_per_year=12)
        if isinstance(result, pd.Series):
            assert result.isna().all(), "No downside periods → NaN (not inf)"
        else:
            assert np.isnan(result), "No downside periods → NaN (not inf)"

    def test_series_input(self, returns_df):
        result = sortino_ratio(returns_df["StratA"], periods_per_year=12)
        assert isinstance(result, (float, np.floating))


class TestCalmarRatio:
    def test_basic(self, returns_df):
        result = calmar_ratio(returns_df, periods_per_year=12)
        assert isinstance(result, pd.Series)

    def test_no_drawdown_returns_nan(self, positive_only_returns):
        result = calmar_ratio(positive_only_returns, periods_per_year=12)
        assert result.isna().all(), "Zero max-drawdown should give NaN, not inf"


class TestOmegaRatio:
    def test_basic(self, returns_df):
        result = omega_ratio(returns_df)
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_no_losses_returns_nan(self, positive_only_returns):
        result = omega_ratio(positive_only_returns)
        assert result.isna().all(), "No losses → NaN"

    def test_series_no_losses(self, positive_only_returns):
        result = omega_ratio(positive_only_returns["AlwaysUp"])
        assert np.isnan(result)
