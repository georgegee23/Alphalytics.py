"""Tests for alphalytics.plotting.performance — plot_capture_hit_rate."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alphalytics.plotting.performance import plot_capture_hit_rate


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
