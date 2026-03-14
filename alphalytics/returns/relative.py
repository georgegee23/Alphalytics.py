
import pandas as pd
import numpy as np
from typing import Union


# ==========================================
# BATTING AVERAGES (Hit Rates)
# ==========================================

def batting_average(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall percentage of periods the strategy beats the benchmark."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: batting_average(col, benchmark_returns))

    excess = (strategy_returns - benchmark_returns).dropna()
    if excess.empty: return np.nan

    return float((excess > 0).mean())

def bull_batting_average(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Batting average only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_batting_average(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = strat[bench > 0] - bench[bench > 0]

    if bull_excess.empty: return np.nan

    return float((bull_excess > 0).mean())

def bear_batting_average(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Batting average only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_batting_average(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = strat[bench < 0] - bench[bench < 0]

    if bear_excess.empty: return np.nan

    return float((bear_excess > 0).mean())

# ==========================================
# WIN/LOSS RATIOS (Payoff Ratios)
# ==========================================

def win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall ratio of average outperformance to absolute average underperformance."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: win_loss_ratio(col, benchmark_returns))

    excess = (strategy_returns - benchmark_returns).dropna()
    wins, losses = excess[excess > 0], excess[excess < 0]

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

def bull_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = strat[bench > 0] - bench[bench > 0]

    wins, losses = bull_excess[bull_excess > 0], bull_excess[bull_excess < 0]

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

def bear_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = strat[bench < 0] - bench[bench < 0]

    wins, losses = bear_excess[bear_excess > 0], bear_excess[bear_excess < 0]

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf if avg_win > 0 else 0.0
    return float(avg_win / avg_loss)

# ==========================================
# ACTIVE RETURN
# ==========================================

def active_return(
    strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series,
    periods_per_year: int = 12
) -> Union[float, pd.Series]:
    """
    Computes the annualized arithmetic mean of excess returns.
    """
    # Handle DataFrame recursion
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: active_return(col, benchmark_returns, periods_per_year))

    # Align data to avoid mismatched date errors
    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()

    if excess.empty:
        return np.nan

    # Annualize the mean excess return
    return float(excess.mean() * periods_per_year)


# ==========================================
# TRACKING ERROR (Active Risk)
# ==========================================

def tracking_error(
    strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series,
    periods_per_year: int = 12
) -> Union[float, pd.Series]:
    """
    Computes the annualized standard deviation of excess returns.
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: tracking_error(col, benchmark_returns, periods_per_year))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()

    # Require at least 2 data points for standard deviation
    if len(excess) < 2:
        return np.nan

    # Calculate sample standard deviation (ddof=1) and annualize it
    return float(excess.std(ddof=1) * np.sqrt(periods_per_year))


# ==========================================
# INFORMATION RATIO
# ==========================================

def information_ratio(
    strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series,
    periods_per_year: int = 12
) -> Union[float, pd.Series]:
    """
    Computes the annualized Information Ratio (Active Return / Tracking Error).
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: information_ratio(col, benchmark_returns, periods_per_year))

    # We can just call our other two functions directly
    ann_active_return = active_return(strategy_returns, benchmark_returns, periods_per_year)
    ann_tracking_error = tracking_error(strategy_returns, benchmark_returns, periods_per_year)

    # Catch division by zero if the strategy perfectly mimics the benchmark
    if pd.isna(ann_tracking_error) or ann_tracking_error == 0:
        return np.nan

    return float(ann_active_return / ann_tracking_error)


# ==========================================
# BATTING AVERAGES (Vectorized)
# ==========================================

def batting_averages(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Calculates Overall, Up Market, and Down Market Batting Averages for multiple strategies.

    IMPORTANT: All metrics are computed on the COMMON overlapping period only
    (i.e., dates where EVERY strategy AND the benchmark have valid returns).
    This guarantees perfect comparability across strategies.

    Batting Average is the percentage of periods where the strategy outperformed the benchmark,
    expressed as a decimal (e.g., 0.65 for 65%). Up/Down markets exclude periods where benchmark == 0.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies (columns = strategy names).
        benchmark (pd.Series): Periodic returns of the benchmark.

    Returns:
        pd.DataFrame: Summary table with the three batting average metrics per strategy.
    """
    # 1. Input validation
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    # 2. Align on index + keep only common valid dates
    data = returns.copy()
    data["bench_"] = benchmark                    # pandas aligns automatically
    data = data.dropna(how="any")                 # drop row if ANY NaN present

    if data.empty:
        # No overlapping data at all
        return pd.DataFrame(
            {
                "Batting Average": np.nan,
                "Up Market Batting Avg": np.nan,
                "Down Market Batting Avg": np.nan,
            },
            index=returns.columns,
        )

    aligned_returns = data[returns.columns]       # or data.drop(columns=["bench_"])
    aligned_bench = data["bench_"]

    # 3. Overall Batting Average: % of time Strategy > Benchmark
    overall_avg = aligned_returns.gt(aligned_bench, axis=0).mean()

    # 4. Up Market Batting Average: % of time Strategy > Benchmark (when Bench > 0)
    up_mask = aligned_bench > 0
    if up_mask.sum() > 0:
        up_avg = aligned_returns.loc[up_mask].gt(aligned_bench.loc[up_mask], axis=0).mean()
    else:
        up_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 5. Down Market Batting Average: % of time Strategy > Benchmark (when Bench < 0)
    down_mask = aligned_bench < 0
    if down_mask.sum() > 0:
        down_avg = aligned_returns.loc[down_mask].gt(aligned_bench.loc[down_mask], axis=0).mean()
    else:
        down_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 6. Final table
    batting_df = pd.DataFrame(
        {
            "Batting Average": overall_avg,
            "Up Market Batting Avg": up_avg,
            "Down Market Batting Avg": down_avg,
        }
    )

    return batting_df


# ==========================================
# CAPTURE RATIOS
# ==========================================

def down_capture(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Down Capture Ratio using the Geometric Mean method.

    Args:
        returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.

    Returns:
        pd.Series: The ratios (e.g., 0.95 = 95%) for each strategy. Returns np.nan if no down markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()

    # 2. Filter for periods where Benchmark was strictly DOWN (< 0)
    #    Standard practice is < 0, but some use <= 0.
    down_market = df[df['bench'] < 0]

    # 3. Handle edge case: No down markets
    if len(down_market) == 0:
        return pd.Series(np.nan, index=returns.columns)

    n = len(down_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for down periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product down each column simultaneously.

    port_geo_avg = (np.prod(1 + down_market[returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + down_market['bench'])) ** (1 / n) - 1

    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter < 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=returns.columns)

    ratio = port_geo_avg / bench_geo_avg

    return ratio

def up_capture(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculates the Up Capture Ratio using the Geometric Mean method.

    Args:
        returns (pd.DataFrame): Periodic returns (e.g., monthly) of the strategies.
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.

    Returns:
        pd.Series: The ratios (e.g., 1.05 = 105%) for each strategy. Returns np.nan if no up markets occur.
    """
    # 1. Align data (intersection of dates) and drop missing values
    #    This ensures we only compare periods where both have data.
    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()

    # 2. Filter for periods where Benchmark was strictly UP (> 0)
    #    Standard practice is > 0, but some use >= 0.
    up_market = df[df['bench'] > 0]

    # 3. Handle edge case: No up markets
    if len(up_market) == 0:
        return pd.Series(np.nan, index=returns.columns)

    n = len(up_market)

    # 4. Calculate Geometric Mean (Compound Annual Growth Rate style) for up periods
    #    Formula: (Product(1 + r)) ^ (1/n) - 1
    #    Note: axis=0 ensures we calculate the product up each column simultaneously.

    port_geo_avg = (np.prod(1 + up_market[returns.columns], axis=0)) ** (1 / n) - 1
    bench_geo_avg = (np.prod(1 + up_market['bench'])) ** (1 / n) - 1

    # 5. Calculate Ratio
    #    Safety check: Ensure benchmark average is not 0 (unlikely given filter > 0)
    if bench_geo_avg == 0:
        return pd.Series(np.nan, index=returns.columns)

    ratio = port_geo_avg / bench_geo_avg

    return ratio

def capture_ratios(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """
    Calculates Up Capture, Down Capture, and Capture Spread for multiple strategies.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark_returns (pd.Series): Periodic returns of the benchmark.

    Returns:
        pd.DataFrame: A summary table with strategies as rows and capture metrics as columns.
    """

    # 1. Calculate ratios using our vectorized functions
    # (These functions already handle the date alignment and dropna internally!)
    up = up_capture(returns, benchmark_returns)
    down = down_capture(returns, benchmark_returns)

    # 2. Combine into a clean summary DataFrame
    summary_df = pd.DataFrame({
        "Up Capture": up,
        "Down Capture": down
    })

    # 3. Add Capture Spread (Up Capture - Down Capture)
    # A positive spread indicates the manager adds value across full market cycles.
    summary_df["Capture Spread"] = summary_df["Up Capture"] - summary_df["Down Capture"]

    # 4. Add Overall Capture Ratio (Up Capture / Down Capture)
    # A ratio > 1.0 generally implies a favorable asymmetric return profile.
    summary_df["Overall Capture"] = summary_df["Up Capture"] / summary_df["Down Capture"]

    return summary_df
