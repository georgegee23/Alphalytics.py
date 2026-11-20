"""Shared test fixtures for alphalytics."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def monthly_index():
    """36 monthly dates (3 years)."""
    return pd.date_range("2020-01-31", periods=36, freq="ME")


@pytest.fixture
def daily_index():
    """~1 year of business days."""
    return pd.date_range("2023-01-02", periods=252, freq="B")


@pytest.fixture
def returns_df(monthly_index):
    """DataFrame with two strategies of monthly returns."""
    rng = np.random.default_rng(42)
    data = {
        "StratA": rng.normal(0.005, 0.03, len(monthly_index)),
        "StratB": rng.normal(0.003, 0.04, len(monthly_index)),
    }
    return pd.DataFrame(data, index=monthly_index)


@pytest.fixture
def benchmark_series(monthly_index):
    """Benchmark monthly returns."""
    rng = np.random.default_rng(99)
    return pd.Series(
        rng.normal(0.004, 0.035, len(monthly_index)),
        index=monthly_index,
        name="Benchmark",
    )


@pytest.fixture
def flat_returns(monthly_index):
    """Constant-return series (zero volatility)."""
    return pd.DataFrame(
        {"Flat": np.full(len(monthly_index), 0.01)},
        index=monthly_index,
    )


@pytest.fixture
def flat_benchmark(monthly_index):
    """Constant benchmark (zero variance)."""
    return pd.Series(
        np.full(len(monthly_index), 0.005),
        index=monthly_index,
        name="FlatBench",
    )


@pytest.fixture
def positive_only_returns(monthly_index):
    """Returns that are always positive (no drawdowns, no downside)."""
    rng = np.random.default_rng(7)
    vals = np.abs(rng.normal(0.01, 0.005, len(monthly_index)))
    return pd.DataFrame({"AlwaysUp": vals}, index=monthly_index)


@pytest.fixture
def factor_df(monthly_index):
    """Factor scores for quantile / turnover tests."""
    rng = np.random.default_rng(123)
    cols = [f"Asset{i}" for i in range(10)]
    data = rng.normal(0, 1, (len(monthly_index), len(cols)))
    return pd.DataFrame(data, index=monthly_index, columns=cols)


@pytest.fixture
def opening_loss_returns():
    """-50%, +10%, -10%. Anchored max drawdown is -50.5%."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.Series([-0.50, 0.10, -0.10], index=idx, name="OpenLoss")


@pytest.fixture
def alternating_monthly_returns():
    """24 months of +20%, -10%. Compounded annual return is 1.08**6 - 1."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.Series([0.20, -0.10] * 12, index=idx, name="Alternate")


@pytest.fixture
def gapped_downside_returns():
    """One missing row. Downside variance at ddof=0 is 0.000325."""
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    return pd.Series([0.01, np.nan, -0.02, -0.03, 0.01], index=idx, name="Gapped")


@pytest.fixture
def beta_gap_returns():
    """Strategy with one gap. Beta against beta_gap_benchmark is 1.4."""
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    return pd.DataFrame({"A": [0.02, np.nan, -0.02, 0.04, 0.01]}, index=idx)


@pytest.fixture
def beta_gap_benchmark(beta_gap_returns):
    return pd.Series([0.01, 0.02, -0.01, 0.03, 0.00], index=beta_gap_returns.index, name="Bench")


@pytest.fixture
def bull_gap_strategy():
    """Beats the benchmark in every valid up month. One up month is missing."""
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    return pd.Series([0.02, np.nan, 0.03, 0.01], index=idx, name="BullGap")


@pytest.fixture
def bull_gap_benchmark(bull_gap_strategy):
    return pd.Series([0.01, 0.01, 0.01, -0.02], index=bull_gap_strategy.index, name="Bench")


@pytest.fixture
def asset_returns(monthly_index):
    """Asset-level returns matching factor_df shape."""
    rng = np.random.default_rng(456)
    cols = [f"Asset{i}" for i in range(10)]
    data = rng.normal(0.005, 0.04, (len(monthly_index), len(cols)))
    return pd.DataFrame(data, index=monthly_index, columns=cols)
