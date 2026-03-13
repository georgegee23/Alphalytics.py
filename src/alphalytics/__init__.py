from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("Alphalytics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .ic_analysis import cs_spearmanr, ts_spearmanr, compute_ic_stats, factor_decay

from .quantile_analysis import (
    to_quantiles, compute_quantile_returns, get_quantile_holdings,
    compute_mean_quantile_forward_return, fwd_quantile_stats,
)

from .turnover_analysis import (
    calculate_autocorrelation, compute_factor_autocorr,
    compute_quantile_turnover, compute_quantiles_turnover,
)

from .performance_metrics import (
    return_n, return_ytd, ann_return, ann_return_common_si, performance_table,
    downside_variance, sortino_ratio,
    compute_capm, beta, bull_bear_beta,
    evaluate_consistency,
    batting_average, bull_batting_average, bear_batting_average,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    active_return, tracking_error, information_ratio,
    batting_averages,
    down_capture, up_capture, capture_ratios,
    rolling_beta,
    cumgrowth, compute_forward_returns,
)

from .plotting import (
    plot_factor_data, plot_cumulative_performance, plot_quantiles_risk_metrics,
    plot_quantile_correlations, plot_spearman_rank,
    plot_ic_hist, qqplot_ic, plot_ic_summary,
    plot_factor_decay, plot_forward_returns, plot_quantiles_annual_turnover,
    plot_growth, plot_risk_return, plot_capture_ratios, plot_batting_averages,
    plot_area, plot_rolling_overunder,
)

from .utils import detect_extreme_outliers, detect_internal_nan, fill_first_nan

__all__ = [
    # ic_analysis
    "cs_spearmanr", "ts_spearmanr", "compute_ic_stats", "factor_decay",
    # quantile_analysis
    "to_quantiles", "compute_quantile_returns", "get_quantile_holdings",
    "compute_mean_quantile_forward_return", "fwd_quantile_stats",
    # turnover_analysis
    "calculate_autocorrelation", "compute_factor_autocorr",
    "compute_quantile_turnover", "compute_quantiles_turnover",
    # performance_metrics
    "return_n", "return_ytd", "ann_return", "ann_return_common_si", "performance_table",
    "downside_variance", "sortino_ratio",
    "compute_capm", "beta", "bull_bear_beta",
    "evaluate_consistency",
    "batting_average", "bull_batting_average", "bear_batting_average",
    "win_loss_ratio", "bull_win_loss_ratio", "bear_win_loss_ratio",
    "active_return", "tracking_error", "information_ratio",
    "batting_averages",
    "down_capture", "up_capture", "capture_ratios",
    "rolling_beta",
    "cumgrowth", "compute_forward_returns",
    # plotting
    "plot_factor_data", "plot_cumulative_performance", "plot_quantiles_risk_metrics",
    "plot_quantile_correlations", "plot_spearman_rank",
    "plot_ic_hist", "qqplot_ic", "plot_ic_summary",
    "plot_factor_decay", "plot_forward_returns", "plot_quantiles_annual_turnover",
    "plot_growth", "plot_risk_return", "plot_capture_ratios", "plot_batting_averages",
    "plot_area", "plot_rolling_overunder",
    # utils
    "detect_extreme_outliers", "detect_internal_nan", "fill_first_nan",
]