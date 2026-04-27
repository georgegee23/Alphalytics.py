from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("Alphalytics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# --- Returns-based analysis ---
from .returns import (

    return_n, return_ytd, ann_return, ann_return_common_si, performance_table,
    cumgrowth,

    annual_std, downside_variance, ann_downside_deviation,
    to_drawdowns, max_drawdown, top_drawdowns, compare_drawdowns, 
    average_drawdown, average_drawdown_duration, 
    
    sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio, tail_ratio, pain_ratio, deviation_ratio,
    tail_dispersion_ratio,
    compute_capm, beta, bull_bear_beta, rolling_beta,
    active_return, active_risk, information_ratio,
    rolling_active_return, rolling_hit_rate, rolling_active_risk, rolling_information_ratio,
    hit_rate, bull_hit_rate, bear_hit_rate,
    hit_rates,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    up_capture, down_capture, capture_spread, capture_ratios,
    rolling_up_capture, rolling_down_capture, rolling_capture_spread,
    rolling_omega_ratio,
    evaluate_consistency, evaluate_asymmetry,
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
    plot_capture_ratios, plot_capture, plot_tail_capture, plot_capm, plot_capture_hit_rate, plot_hit_rates, plot_win_loss,
    plot_area, plot_rolling_overunder, plot_xy_symmetric,
    plot_rolling_information_ratio, plot_rolling_active_return, plot_rolling_return,
    plot_trailing_performance,
    plot_factor_data, plot_quantiles_risk_metrics,
    plot_quantile_correlations, plot_spearman_rank,
    plot_ic_hist, qqplot_ic, plot_ic_summary,
    plot_factor_decay, plot_forward_returns,
    plot_quantiles_annual_turnover,

    # Drawdown Analysis
    plot_compare_drawdowns, plot_compare_drawdown_volatility,
)

# --- Utilities ---
from .utils import detect_extreme_outliers, detect_internal_nan, fill_first_nan

__all__ = [
    # returns.metrics
    "return_n", "return_ytd", "ann_return", "ann_return_common_si", "performance_table",
    "cumgrowth",
    # returns.risk
    "annual_std", "downside_variance", "ann_downside_deviation",
    "to_drawdowns", "max_drawdown", "top_drawdowns", "compare_drawdowns", 
    "average_drawdown", "average_drawdown_duration",
    # returns.ratios
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio", "rolling_omega_ratio", "tail_ratio", "pain_ratio", "deviation_ratio", "tail_dispersion_ratio",
    # returns.capm
    "compute_capm", "beta", "bull_bear_beta", "rolling_beta",
    # returns.relative
    "active_return", "active_risk", "information_ratio",
    "rolling_active_return", "rolling_hit_rate", "rolling_active_risk", "rolling_information_ratio",
    "hit_rate", "bull_hit_rate", "bear_hit_rate",
    "hit_rates",
    "win_loss_ratio", "bull_win_loss_ratio", "bear_win_loss_ratio",
    "up_capture", "down_capture", "capture_spread", "capture_ratios",
    "rolling_up_capture", "rolling_down_capture", "rolling_capture_spread",
    # returns.aggregators
    "evaluate_consistency", "evaluate_asymmetry",
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
    "plot_capture_ratios", "plot_capture", "plot_tail_capture", "plot_capm", "plot_capture_hit_rate", "plot_hit_rates", "plot_win_loss",
    "plot_area", "plot_rolling_overunder", "plot_xy_symmetric",
    "plot_factor_data", "plot_quantiles_risk_metrics",
    "plot_quantile_correlations", "plot_spearman_rank",
    "plot_ic_hist", "qqplot_ic", "plot_ic_summary",
    "plot_factor_decay", "plot_forward_returns",
    "plot_quantiles_annual_turnover",
    "plot_rolling_information_ratio", "plot_rolling_active_return", "plot_rolling_return",
    "plot_trailing_performance",
    "plot_compare_drawdowns", "plot_compare_drawdown_volatility",
    # utils
    "detect_extreme_outliers", "detect_internal_nan", "fill_first_nan",
]
