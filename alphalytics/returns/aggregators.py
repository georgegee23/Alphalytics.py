
import pandas as pd
import numpy as np

from .relative import (
    hit_rate, bull_hit_rate, bear_hit_rate,
    rolling_active_return, rolling_active_risk,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    active_return, active_risk, information_ratio,
    up_capture, down_capture, tail_capture_ratio,
)

# ==========================================
# TRAILING PERFORMANCE TABLE
# ==========================================

from .metrics import (
    return_n, return_ytd, ann_return, ann_return_common_si,
)


def performance_table(rets: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """Builds a standard institutional performance table (1M, 3M, YTD, 1Y, 3Y, 5Y, 10Y, SI).

    Forces an apples-to-apples comparison by aligning all dates first.

    Args:
        rets: Periodic returns of the strategies.
        periods_per_year: Frequency of the data (12 for monthly, 252 for daily).

    Returns:
        A table where rows are strategies and columns are trailing timeframes.
    """
    # 1. Align all data to guarantee apples-to-apples comparison
    aligned_rets = rets.dropna()

    if aligned_rets.empty:
        raise ValueError("No common dates found across all strategies.")

    # 2. Calculate individual metrics using your toolkit

    # 1-Month Return (Safely compounds daily/weekly data if needed)
    m1_n = int(periods_per_year / 12)
    m1 = return_n(aligned_rets, m1_n)
    m1.name = "1M"

    # 3-Month Return
    m3_n = int(periods_per_year / 4)
    m3 = return_n(aligned_rets, m3_n)
    m3.name = "3M"

    ytd = return_ytd(aligned_rets)
    yr1 = ann_return(aligned_rets, years=1, periods_per_year=periods_per_year)
    yr3 = ann_return(aligned_rets, years=3, periods_per_year=periods_per_year)
    yr5 = ann_return(aligned_rets, years=5, periods_per_year=periods_per_year)
    yr10 = ann_return(aligned_rets, years=10, periods_per_year=periods_per_year)

    # 3. Calculate Common SI safely (in case they share less than 1 year)
    try:
        si = ann_return_common_si(aligned_rets, periods_per_year=periods_per_year)
    except ValueError:
        si = pd.Series(np.nan, index=aligned_rets.columns, name="Common SI")

    # 4. Concatenate into a single presentation DataFrame
    perf_table = pd.concat([m1, m3, ytd, yr1, yr3, yr5, yr10, si], axis=1)

    return perf_table

# ==========================================
# EVALUATE INVESTMENTS CONSISTENCY
# ==========================================

def evaluate_consistency(strategy_returns: pd.DataFrame | pd.Series, benchmark_returns: pd.Series,
    window: int = 36, periods_per_year: int = 12) -> pd.DataFrame:
    """Evaluates the relative consistency and active risk of one or multiple strategies.

    Args:
        strategy_returns: Periodic returns for the strategy/strategies.
        benchmark_returns: Periodic returns for the benchmark.
        window: Rolling window size in periods for the active-return stability metric.
        periods_per_year: Annualization factor (252 for daily, 12 for monthly).

    Returns:
        A tearsheet with strategies as rows and consistency/risk metrics as columns,
        with all values as floats.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns

    results = {}

    for col in strat_df.columns:
        strat = strat_df[col]

        results[col] = {
            "Hit Rate": hit_rate(strat, benchmark_returns),
            "Bull Hit Rate": bull_hit_rate(strat, benchmark_returns),
            "Bear Hit Rate": bear_hit_rate(strat, benchmark_returns),
            "Rolling AR Std": rolling_active_return(strat, benchmark_returns, window, periods_per_year).std(),
            "Act. Ret. (Ann.)": active_return(strat, benchmark_returns, periods_per_year),
            "TE (Ann.)": active_risk(strat, benchmark_returns, periods_per_year),
            "Info Ratio": information_ratio(strat, benchmark_returns, periods_per_year),
        }

    return pd.DataFrame(results).T


# ==========================================
# EVALUATE INVESTMENTS ASYMMETRY
# ==========================================

def evaluate_asymmetry(strategy_returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series) -> pd.DataFrame:
    """Evaluates the up/down asymmetry and payoff profile of one or multiple strategies.

    Args:
        strategy_returns: Periodic returns for the strategy/strategies.
        benchmark_returns: Periodic returns for the benchmark.

    Returns:
        A tearsheet with strategies as rows and asymmetry/payoff metrics as columns,
        with all values as floats.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns

    results = {}

    for col in strat_df.columns:
        strat = strat_df[col]

        up = up_capture(strat, benchmark_returns)
        down = down_capture(strat, benchmark_returns)

        results[col] = {
            "Up Capture": up,
            "Down Capture": down,
            "Overall Capture": np.nan if pd.isna(down) or down == 0 else up / down,
            "Tail Capture": tail_capture_ratio(strat, benchmark_returns),
            "Bull Win/Loss": bull_win_loss_ratio(strat, benchmark_returns),
            "Bear Win/Loss": bear_win_loss_ratio(strat, benchmark_returns),
            "Win/Loss Ratio": win_loss_ratio(strat, benchmark_returns),
            
        }

    return pd.DataFrame(results).T