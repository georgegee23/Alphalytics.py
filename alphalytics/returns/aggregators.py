
import pandas as pd
import numpy as np

from .relative import (
    batting_average, bull_batting_average, bear_batting_average,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    active_return, tracking_error, information_ratio,
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

def evaluate_consistency(
    strategy_returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12,
    formatted: bool = False
) -> pd.DataFrame:
    """Evaluates the relative consistency and active risk of one or multiple strategies.

    Args:
        strategy_returns: Periodic returns for the strategy/strategies.
        benchmark_returns: Periodic returns for the benchmark.
        periods_per_year: Annualization factor (252 for daily, 12 for monthly).
        formatted: If True, returns strings with %, x, and decimals.
            If False, returns raw floats.

    Returns:
        A tearsheet with strategies as rows and consistency/risk metrics as columns.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns

    results = {}

    # Calculate metrics, ordering them exactly how they should appear left-to-right
    for col in strat_df.columns:
        strat = strat_df[col]

        results[col] = {
            "Batting Average (Overall)": batting_average(strat, benchmark_returns),
            "Batting Average (Bull)": bull_batting_average(strat, benchmark_returns),
            "Batting Average (Bear)": bear_batting_average(strat, benchmark_returns),
            "Win/Loss Ratio (Overall)": win_loss_ratio(strat, benchmark_returns),
            "Win/Loss Ratio (Bull)": bull_win_loss_ratio(strat, benchmark_returns),
            "Win/Loss Ratio (Bear)": bear_win_loss_ratio(strat, benchmark_returns),
            # Placed at the end per your request
            "Active Return (Ann.)": active_return(strat, benchmark_returns, periods_per_year),
            "Tracking Error (Ann.)": tracking_error(strat, benchmark_returns, periods_per_year),
            "Information Ratio": information_ratio(strat, benchmark_returns, periods_per_year)
        }

    # Convert dictionary to DataFrame and transpose (.T) so strategies are rows
    tearsheet = pd.DataFrame(results).T

    # Apply string formatting for presentation if requested
    if formatted:
        format_rules = {
            "Batting Average (Overall)": "{:.2%}",
            "Batting Average (Bull)": "{:.2%}",
            "Batting Average (Bear)": "{:.2%}",
            "Win/Loss Ratio (Overall)": "{:.2f}x",
            "Win/Loss Ratio (Bull)": "{:.2f}x",
            "Win/Loss Ratio (Bear)": "{:.2f}x",
            "Active Return (Ann.)": "{:.2%}",
            "Tracking Error (Ann.)": "{:.2%}",
            "Information Ratio": "{:.2f}"
        }
        for metric, fmt in format_rules.items():
            if metric in tearsheet.columns:
                tearsheet[metric] = tearsheet[metric].apply(
                    lambda x: fmt.format(x) if pd.notnull(x) else "N/A"
                )

    return tearsheet
