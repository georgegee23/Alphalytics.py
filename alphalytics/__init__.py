from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("Alphalytics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# --- Returns-based analysis ---
from .returns import (
    return_n, return_ytd, ann_return, ann_return_common_si,
    cumgrowth,
    annual_std, downside_variance, sortino_ratio,
    compute_capm, beta, bull_bear_beta, rolling_beta,
    active_return, tracking_error, information_ratio, rolling_information_ratio, 
    batting_average, bull_batting_average, bear_batting_average,
    batting_averages,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    up_capture, down_capture, capture_ratios,
    evaluate_consistency, performance_table
)

# --- Factor analysis ---
from .factors import (
    cs_spearmanr, ts_spearmanr, compute_ic_stats, factor_decay,
    to_quantiles, compute_quantile_returns, get_quantile_holdings,
    compute_mean_quantile_forward_return, fwd_quantile_stats,
    calculate_autocorrelation, compute_factor_autocorr,
    compute_quantile_turnover, compute_quantiles_turnover,
    compute_forward_returns,
)

# --- Plotting ---
from .plotting import (
    plot_growth, plot_cumulative_performance, plot_risk_return,
    plot_capture_ratios, plot_batting_averages,
    plot_area, plot_rolling_overunder, plot_xy_symmetric,
    plot_rolling_information_ratio,
    plot_factor_data, plot_quantiles_risk_metrics,
    plot_quantile_correlations, plot_spearman_rank,
    plot_ic_hist, qqplot_ic, plot_ic_summary,
    plot_factor_decay, plot_forward_returns,
    plot_quantiles_annual_turnover,
)

# --- Utilities ---
from .utils import detect_extreme_outliers, detect_internal_nan, fill_first_nan

__all__ = [
    # returns.metrics
    "return_n", "return_ytd", "ann_return", "ann_return_common_si",
    "cumgrowth",
    # returns.risk
    "annual_std", "downside_variance", "sortino_ratio",
    # returns.capm
    "compute_capm", "beta", "bull_bear_beta", "rolling_beta",
    # returns.relative
    "active_return", "tracking_error", "information_ratio", "rolling_information_ratio",
    "batting_average", "bull_batting_average", "bear_batting_average",
    "batting_averages",
    "win_loss_ratio", "bull_win_loss_ratio", "bear_win_loss_ratio",
    "up_capture", "down_capture", "capture_ratios",
    # returns.aggregators
    "evaluate_consistency", "performance_table",
    # factors.ic
    "cs_spearmanr", "ts_spearmanr", "compute_ic_stats", "factor_decay",
    # factors.quantiles
    "to_quantiles", "compute_quantile_returns", "get_quantile_holdings",
    "compute_mean_quantile_forward_return", "fwd_quantile_stats",
    # factors.turnover
    "calculate_autocorrelation", "compute_factor_autocorr",
    "compute_quantile_turnover", "compute_quantiles_turnover",
    # factors.utils
    "compute_forward_returns",
    # plotting
    "plot_growth", "plot_cumulative_performance", "plot_risk_return",
    "plot_capture_ratios", "plot_batting_averages",
    "plot_area", "plot_rolling_overunder", "plot_xy_symmetric",
    "plot_factor_data", "plot_quantiles_risk_metrics",
    "plot_quantile_correlations", "plot_spearman_rank",
    "plot_ic_hist", "qqplot_ic", "plot_ic_summary",
    "plot_factor_decay", "plot_forward_returns",
    "plot_quantiles_annual_turnover",
    "plot_rolling_information_ratio",
    # utils
    "detect_extreme_outliers", "detect_internal_nan", "fill_first_nan",
]
