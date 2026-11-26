"""Tests for alphalytics.plotting.performance — plot_capture_hit_rate."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alphalytics.plotting.performance import (
    plot_capture_hit_rate, plot_risk_return, plot_sortino,
    plot_cumulative_performance, plot_trailing_performance,
)
from alphalytics.returns.aggregators import performance_table
from alphalytics.returns.metrics import ann_return_common_si
from alphalytics.returns.ratios import sharpe_ratio, sortino_ratio
from alphalytics.returns.risk import annual_std


@pytest.fixture
def strategy_returns():
    """Synthetic multi-strategy return DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    return pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        index=dates,
        columns=["Fund A", "Fund B", "Fund C"],
    )


@pytest.fixture
def benchmark_returns(strategy_returns):
    """Synthetic benchmark return Series aligned to strategy_returns."""
    np.random.seed(99)
    return pd.Series(
        np.random.randn(len(strategy_returns)) * 0.01,
        index=strategy_returns.index,
        name="Benchmark",
    )


class TestPlotCaptureHitRate:
    def test_returns_fig_and_ax(self, strategy_returns, benchmark_returns):
        fig, ax = plot_capture_hit_rate(strategy_returns, benchmark_returns)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_axes_labels(self, strategy_returns, benchmark_returns):
        fig, ax = plot_capture_hit_rate(strategy_returns, benchmark_returns)
        assert ax.get_xlabel() == "Overall Capture"
        assert ax.get_ylabel() == "Hit Rate"
        plt.close(fig)

    def test_custom_title(self, strategy_returns, benchmark_returns):
        fig, ax = plot_capture_hit_rate(
            strategy_returns, benchmark_returns, title="My Title"
        )
        assert ax.get_title() == "My Title"
        plt.close(fig)

    def test_reference_lines_present(self, strategy_returns, benchmark_returns):
        fig, ax = plot_capture_hit_rate(strategy_returns, benchmark_returns)
        lines = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
        assert len(lines) >= 2
        plt.close(fig)

    def test_custom_figsize(self, strategy_returns, benchmark_returns):
        fig, ax = plot_capture_hit_rate(
            strategy_returns, benchmark_returns, figsize=(8, 6)
        )
        w, h = fig.get_size_inches()
        assert abs(w - 8) < 0.5 and abs(h - 6) < 0.5
        plt.close(fig)

    def test_single_marker_string(self, strategy_returns, benchmark_returns):
        """A single marker string should apply to all points without error."""
        fig, ax = plot_capture_hit_rate(strategy_returns, benchmark_returns, markers='D')
        plt.close(fig)

    def test_marker_dict(self, strategy_returns, benchmark_returns):
        """A dict of markers should map correctly."""
        m = {"Fund A": "o", "Fund B": "s", "Fund C": "D"}
        fig, ax = plot_capture_hit_rate(strategy_returns, benchmark_returns, markers=m)
        plt.close(fig)

    def test_undefined_overall_capture_warns(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        bench = pd.Series([0.03, -0.02] * 12, index=idx, name="Bench")
        frame = pd.DataFrame({
            "Convex": np.where(bench > 0, 0.3 * bench, -0.2 * bench),
            "Index": bench.to_numpy() * 1.1,
        }, index=idx)
        with pytest.warns(UserWarning, match="Convex"):
            fig, ax = plot_capture_hit_rate(frame, bench, markers="o")
        plt.close(fig)


class TestFrequencyDefaults:
    def test_cumulative_performance_infers_monthly(self):
        idx = pd.date_range("2015-01-31", periods=72, freq="ME")
        rets = pd.DataFrame({"A": np.linspace(0.0, 0.02, len(idx))}, index=idx)
        fig, axes = plot_cumulative_performance(rets)
        try:
            cells = axes[2].tables[0].get_celld()
            header = [cells[(0, c)].get_text().get_text() for c in range(9)]
            assert header[1:] == ["1M", "3M", "YTD", "1 Year", "3 Year", "5 Year", "10 Year", "SI"]
            # 1M is the last monthly return, not 21 months compounded.
            assert cells[(1, 1)].get_text().get_text() == f"{rets['A'].iloc[-1]:.2%}"
            assert cells[(1, 7)].get_text().get_text() == "N/A"
        finally:
            plt.close(fig)

    def test_trailing_performance_infers_daily(self, strategy_returns):
        fig, ax = plot_trailing_performance(strategy_returns)
        try:
            expected = performance_table(strategy_returns, periods_per_year=252)["1 Year"]
            n_periods = len(ax.get_xticks())
            year_col = [t.get_text() for t in ax.get_xticklabels()].index("1-Year")
            heights = [ax.patches[i * n_periods + year_col].get_height()
                for i in range(len(strategy_returns.columns))]
            assert heights == pytest.approx(expected.to_list())
        finally:
            plt.close(fig)


class TestRiskReturnSlope:
    def test_slope_matches_sharpe(self, strategy_returns):
        fig, ax = plot_risk_return(strategy_returns, periods_per_year=252)
        try:
            for i, col in enumerate(strategy_returns.columns):
                x, y = ax.collections[i].get_offsets()[0]
                expected = sharpe_ratio(strategy_returns[col], periods_per_year=252)
                assert y / x == pytest.approx(expected)
        finally:
            plt.close(fig)

    def test_geometric_matches_table_figures(self):
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(7)
        rets = pd.DataFrame(rng.normal(0.008, 0.04, (60, 2)), index=idx, columns=["S", "B"])
        fig, ax = plot_risk_return(rets, return_method="geometric")
        try:
            exp_ret = ann_return_common_si(rets, periods_per_year=12)
            exp_vol = annual_std(rets, periods_per_year=12)
            for i, col in enumerate(rets.columns):
                x, y = ax.collections[i].get_offsets()[0]
                assert x == pytest.approx(exp_vol[col])
                assert y == pytest.approx(exp_ret[col])
                # Geometric sits below arithmetic for volatile returns.
                assert y < rets[col].mean() * 12
        finally:
            plt.close(fig)

    def test_geometric_uses_shared_periods(self):
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(3)
        rets = pd.DataFrame(rng.normal(0.008, 0.04, (60, 2)), index=idx, columns=["S", "B"])
        rets.iloc[:12, 0] = np.nan  # S starts a year later
        fig, ax = plot_risk_return(rets, return_method="geometric")
        try:
            x, _ = ax.collections[1].get_offsets()[0]
            assert x == pytest.approx(annual_std(rets["B"].iloc[12:], periods_per_year=12))
        finally:
            plt.close(fig)

    def test_unknown_return_method_raises(self, strategy_returns):
        with pytest.raises(ValueError):
            plot_risk_return(strategy_returns, return_method="log")

    def test_sortino_slope_uses_excess_over_mar(self, strategy_returns):
        mar = 0.001
        fig, ax = plot_sortino(strategy_returns, mar=mar, periods_per_year=252)
        try:
            for i, col in enumerate(strategy_returns.columns):
                x, y = ax.collections[i].get_offsets()[0]
                expected = float(sortino_ratio(strategy_returns[col], mar=mar, periods_per_year=252))
                assert y / x == pytest.approx(expected)
        finally:
            plt.close(fig)
