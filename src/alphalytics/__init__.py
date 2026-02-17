import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "Alphalytics"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .ic_analysis import cs_spearmanr, ts_spearmanr, compute_ic_stats, factor_decay

from .quantile_analysis import to_quantiles, compute_quantile_returns, get_quantile_holdings, \
    compute_mean_quantile_forward_return, fwd_quantile_stats

from .turnover_analysis import calculate_autocorrelation, compute_factor_autocorr, compute_quantile_turnover, compute_quantiles_turnover

from .performance_metrics import cumgrowth, compute_performance_table, compute_cumulative_growth, \
    compute_forward_returns, compute_capm, down_capture, up_capture, batting_averages

from .plotting import plot_factor_data, plot_cumulative_performance, plot_quantiles_risk_metrics, plot_quantile_correlations, plot_spearman_rank, \
    plot_ic_hist, qqplot_ic, plot_ic_summary, plot_factor_decay, plot_forward_returns, plot_quantiles_annual_turnover

from .utils import detect_extreme_outliers, detect_internal_nan, fill_first_nan

__all__ = [
    "cs_spearmanr", "ts_spearmanr", "compute_ic_stats", "factor_decay",
    "to_quantiles", "compute_quantile_returns", "get_quantile_holdings",
    "compute_mean_quantile_forward_return", "fwd_quantile_stats",
    "calculate_autocorrelation", "compute_factor_autocorr", "compute_quantile_turnover", "compute_quantiles_turnover",
    "compute_prices", "compute_performance_table", "compute_cumulative_growth", "compute_forward_returns", "compute_capm",
    "down_capture", "up capture", batting_averages,
    "plot_factor_data", "plot_cumulative_performance", "plot_quantiles_risk_metrics", "plot_quantile_correlations",
    "plot_spearman_rank", "plot_ic_hist", "qqplot_ic", "plot_ic_summary", "plot_factor_decay", "plot_forward_returns",
    "plot_quantiles_annual_turnover", 
    "detect_extreme_outliers", "detect_internal_nan", "fill_first_nan"
]