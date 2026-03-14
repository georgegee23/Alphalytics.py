
import pandas as pd
import numpy as np

from .relative import (
    batting_average, bull_batting_average, bear_batting_average,
    win_loss_ratio, bull_win_loss_ratio, bear_win_loss_ratio,
    active_return, tracking_error, information_ratio,
)


# ==========================================
# EVALUATE INVESTMENTS CONSISTENCY
# ==========================================

def evaluate_consistency(
    strategy_returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12,
    formatted: bool = False
) -> pd.DataFrame:
    """
    Evaluates the relative consistency and active risk of one or multiple strategies.

    Parameters
    ----------
    strategy_returns : pd.Series or pd.DataFrame
        Periodic returns for the strategy/strategies.
    benchmark_returns : pd.Series
        Periodic returns for the benchmark.
    periods_per_year : int
        Annualization factor (252 for daily, 12 for monthly).
    formatted : bool
        If True, returns strings with %, x, and decimals. If False, returns raw floats.

    Returns
    -------
    pd.DataFrame
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
