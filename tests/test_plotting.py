"""Tests for alphalytics.plotting.performance — plot_capture_hit_rate."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from alphalytics.plotting.performance import plot_capture_hit_rate


class TestPlotCaptureVsHitRate:
    def test_returns_fig_and_ax(self, returns_df, benchmark_series):
        fig, ax = plot_capture_hit_rate(returns_df, benchmark_series)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_axes_labels(self, returns_df, benchmark_series):
        fig, ax = plot_capture_hit_rate(returns_df, benchmark_series)
        assert ax.get_xlabel() == "Overall Capture"
        assert ax.get_ylabel() == "Hit Rate"
        plt.close(fig)

    def test_custom_title(self, returns_df, benchmark_series):
        fig, ax = plot_capture_hit_rate(
            returns_df, benchmark_series, title="My Title"
        )
        assert ax.get_title() == "My Title"
        plt.close(fig)

    def test_reference_lines_present(self, returns_df, benchmark_series):
        fig, ax = plot_capture_hit_rate(returns_df, benchmark_series)
        # Should have at least 2 reference lines (axhline + axvline)
        lines = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
        assert len(lines) >= 2
        plt.close(fig)

    def test_custom_figsize(self, returns_df, benchmark_series):
        fig, ax = plot_capture_hit_rate(
            returns_df, benchmark_series, figsize=(8, 6)
        )
        w, h = fig.get_size_inches()
        assert abs(w - 8) < 0.5 and abs(h - 6) < 0.5
        plt.close(fig)
