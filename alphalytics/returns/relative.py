
import pandas as pd
import numpy as np
from typing import Union

from alphalytics.utils import _infer_periods_per_year


# ==========================================
# HIT RATE (Batting Average)
# ==========================================

def hit_rate(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall percentage of periods the strategy beats the benchmark."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: hit_rate(col, benchmark_returns))

    excess = (strategy_returns - benchmark_returns).dropna()
    if excess.empty: return np.nan

    return float((excess > 0).mean())

def bull_hit_rate(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Hit rate only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_hit_rate(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = strat[bench > 0] - bench[bench > 0]

    if bull_excess.empty: return np.nan

    return float((bull_excess > 0).mean())

def bear_hit_rate(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Hit rate only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_hit_rate(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = strat[bench < 0] - bench[bench < 0]

    if bear_excess.empty: return np.nan

    return float((bear_excess > 0).mean())


# ==========================================
# HIT RATES (Vectorized)
# ==========================================

def hit_rates(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Calculates Overall, Up Market, and Down Market Hit Rate for multiple strategies.

    IMPORTANT: All metrics are computed on the COMMON overlapping period only
    (i.e., dates where EVERY strategy AND the benchmark have valid returns).
    This guarantees perfect comparability across strategies.

    Hit Rate is the percentage of periods where the strategy outperformed the benchmark,
    expressed as a decimal (e.g., 0.65 for 65%). Up/Down markets exclude periods where benchmark == 0.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies (columns = strategy names).
        benchmark (pd.Series): Periodic returns of the benchmark.

    Returns:
        pd.DataFrame: Summary table with the three hit rate metrics per strategy.
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
                "Hit Rate": np.nan,
                "Up Market Hit Rate": np.nan,
                "Down Market Hit Rate": np.nan,
            },
            index=returns.columns,
        )

    aligned_returns = data[returns.columns]       # or data.drop(columns=["bench_"])
    aligned_bench = data["bench_"]

    # 3. Overall Hit Rate: % of time Strategy > Benchmark
    overall_avg = aligned_returns.gt(aligned_bench, axis=0).mean()

    # 4. Up Market Hit Rate: % of time Strategy > Benchmark (when Bench > 0)
    up_mask = aligned_bench > 0
    if up_mask.sum() > 0:
        up_avg = aligned_returns.loc[up_mask].gt(aligned_bench.loc[up_mask], axis=0).mean()
    else:
        up_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 5. Down Market Hit Rate: % of time Strategy > Benchmark (when Bench < 0)
    down_mask = aligned_bench < 0
    if down_mask.sum() > 0:
        down_avg = aligned_returns.loc[down_mask].gt(aligned_bench.loc[down_mask], axis=0).mean()
    else:
        down_avg = pd.Series(np.nan, index=aligned_returns.columns)

    # 6. Final table
    hitrate_df = pd.DataFrame(
        {
            "Hit Rate": overall_avg,
            "Bull Hit Rate": up_avg,
            "Bear Hit Rate": down_avg,
        }
    )

    return hitrate_df

# ==========================================
# ROLLING HIT RATE
# ==========================================

def rolling_hit_rate(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
                            window: int) -> Union[pd.Series, pd.DataFrame]:
    """Rolling Hit Rate / Batting Average (proportion of periods with positive excess returns).

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.

    Returns:
        pd.Series or pd.DataFrame of rolling batting averages.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns.copy()

    strat_df, bench = strat_df.align(benchmark_returns, join="inner", axis=0)

    # Calculate period-by-period arithmetic excess return
    excess = strat_df.sub(bench, axis=0).dropna()

    # Create boolean mask of positive excess returns (True=1, False=0)
    hits = (excess > 0).astype(int)

    # Rolling mean of a binary mask yields the proportion of 1s (the hit rate)
    result = hits.rolling(window=window).mean().dropna(how="all")

    if result.shape[1] == 1 and isinstance(strategy_returns, pd.Series):
        return result.iloc[:, 0]
    return result


# ==========================================
# WIN/LOSS RATIOS (Payoff Ratios)
# ==========================================

def win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Overall ratio of average outperformance to absolute average underperformance."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()
    if excess.empty: return np.nan

    wins, losses = excess[excess > 0], excess[excess < 0]
    if wins.empty and losses.empty: return np.nan

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf
    return float(avg_win / avg_loss)

def bull_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark up-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bull_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bull_excess = (strat[bench > 0] - bench[bench > 0]).dropna()
    if bull_excess.empty: return np.nan

    wins, losses = bull_excess[bull_excess > 0], bull_excess[bull_excess < 0]
    if wins.empty and losses.empty: return np.nan

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf
    return float(avg_win / avg_loss)

def bear_win_loss_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:

    """Win/loss ratio only during benchmark down-markets."""

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: bear_win_loss_ratio(col, benchmark_returns))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    bear_excess = (strat[bench < 0] - bench[bench < 0]).dropna()
    if bear_excess.empty: return np.nan

    wins, losses = bear_excess[bear_excess > 0], bear_excess[bear_excess < 0]
    if wins.empty and losses.empty: return np.nan

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    if avg_loss == 0: return np.inf
    return float(avg_win / avg_loss)


# ==========================================
# TAIL CAPTURE RATIO
# ==========================================

def tail_capture_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series, upper: float = 0.9, lower: float = 0.1) -> Union[float, pd.Series]:
    """
    Tail Capture Ratio.

    Ratio of upper-tail capture to lower-tail capture, where each capture is
    the strategy's conditional mean divided by the benchmark's conditional
    mean in the same benchmark-tail regime:

        Tail Up Capture   = E[R_strat | R_bench >= q_upper]
                            / E[R_bench | R_bench >= q_upper]
        Tail Down Capture = E[R_strat | R_bench <= q_lower]
                            / E[R_bench | R_bench <= q_lower]
        Tail Capture      = Tail Up Capture / Tail Down Capture

    Values above 1 indicate favorable tail asymmetry: the strategy captures
    more of the benchmark's upper tail than its lower tail. The benchmark
    evaluated against itself returns 1.0 by construction.

    Args:
        strategy_returns: A pandas Series or DataFrame of periodic returns.
        benchmark_returns: A pandas Series of benchmark periodic returns.
        upper: Upper quantile threshold on the benchmark. Defaults to 0.9.
        lower: Lower quantile threshold on the benchmark. Defaults to 0.1.

    Returns:
        A float (Series input) or pd.Series (DataFrame input).
        NaN when either tail is empty, the benchmark's lower-tail mean is zero,
        or the lower-tail capture is zero.
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(
            lambda col: tail_capture_ratio(col, benchmark_returns, upper, lower)
        )

    df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()
    if df.empty: return np.nan
    strat, bench = df.iloc[:, 0], df.iloc[:, 1]

    q_up = bench.quantile(upper)
    q_lo = bench.quantile(lower)

    up_mask = bench >= q_up
    lo_mask = bench <= q_lo

    if not up_mask.any() or not lo_mask.any(): return np.nan

    bench_up_mean = bench[up_mask].mean()
    bench_lo_mean = bench[lo_mask].mean()

    if bench_up_mean == 0 or bench_lo_mean == 0: return np.nan

    up_capture = strat[up_mask].mean() / bench_up_mean
    lo_capture = strat[lo_mask].mean() / bench_lo_mean

    if lo_capture == 0: return np.nan
    return float(up_capture / lo_capture)

# ==========================================
# OMEGA RATIO
# ==========================================

def omega_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    mar: Union[float, pd.Series] = 0.0) -> Union[float, pd.Series]:

    """Ratio of cumulative excess gains to cumulative excess losses over MAR.

    MAR (Minimum Acceptable Return) may be a scalar per-period threshold
    (default 0.0) or a pd.Series benchmark for a relative Omega.
    Returns NaN when there are no losses.
    """

    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: omega_ratio(col, mar))

    if isinstance(mar, pd.Series):
        strategy_returns, mar = strategy_returns.align(mar, join='inner')

    excess = (strategy_returns - mar).dropna()
    if excess.empty: return np.nan

    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())

    if losses == 0: return np.nan
    return float(gains / losses)


# ==========================================
# ROLLING OMEGA RATIO
# ==========================================

def rolling_omega_ratio(strategy_returns: Union[pd.Series, pd.DataFrame],
    mar: Union[float, pd.Series] = 0.0, window: int = 36) -> Union[pd.Series, pd.DataFrame]:
    """Rolling Omega Ratio over a trailing window.

    For each window, delegates to `omega_ratio` so the math stays in one place.
    MAR (Minimum Acceptable Return) may be a scalar per-period threshold
    (default 0.0) or a pd.Series benchmark for a relative Omega.

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        mar: Scalar threshold or pd.Series benchmark.
        window: Rolling window size in periods.

    Returns:
        pd.Series or pd.DataFrame of rolling Omega ratios. NaN in windows with
        no losses or no data.
    """
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns.copy()

    if isinstance(mar, pd.Series):
        strat_df, mar_aligned = strat_df.align(mar, join="inner", axis=0)
    else:
        mar_aligned = mar

    if len(strat_df) < window:
        empty = pd.DataFrame(columns=strat_df.columns)
        return empty.iloc[:, 0] if isinstance(strategy_returns, pd.Series) else empty

    def _slice_mar(i: int):
        if isinstance(mar_aligned, pd.Series):
            return mar_aligned.iloc[i - window + 1 : i + 1]
        return mar_aligned

    records = {
        strat_df.index[i]: omega_ratio(strat_df.iloc[i - window + 1 : i + 1], _slice_mar(i))
        for i in range(window - 1, len(strat_df))
    }

    result = pd.DataFrame(records).T.dropna(how="all")

    if result.shape[1] == 1 and isinstance(strategy_returns, pd.Series):
        return result.iloc[:, 0]
    return result


# ==========================================
# ACTIVE RETURN
# ==========================================

def active_return(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    periods_per_year: int = None) -> Union[float, pd.Series]:
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

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(excess.index)

    # Annualize the mean excess return
    return float(excess.mean() * periods_per_year)


# ==========================================
# TRACKING ERROR (Active Risk)
# ==========================================

def active_risk(strategy_returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series, periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Computes the annualized standard deviation of excess returns.
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: active_risk(col, benchmark_returns, periods_per_year))

    strat, bench = strategy_returns.align(benchmark_returns, join='inner')
    excess = (strat - bench).dropna()

    # Require at least 2 data points for standard deviation
    if len(excess) < 2:
        return np.nan

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(excess.index)

    # Calculate sample standard deviation (ddof=1) and annualize it
    return float(excess.std(ddof=1) * np.sqrt(periods_per_year))


# ==========================================
# INFORMATION RATIO
# ==========================================

def information_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    periods_per_year: int = None) -> Union[float, pd.Series]:
    """
    Computes the annualized Information Ratio (Active Return / Tracking Error).
    """
    if isinstance(strategy_returns, pd.DataFrame):
        return strategy_returns.apply(lambda col: information_ratio(col, benchmark_returns, periods_per_year))

    # We can just call our other two functions directly
    ann_active_return = active_return(strategy_returns, benchmark_returns, periods_per_year)
    ann_active_risk = active_risk(strategy_returns, benchmark_returns, periods_per_year)

    # Catch division by zero if the strategy perfectly mimics the benchmark
    if pd.isna(ann_active_risk) or ann_active_risk == 0:
        return np.nan

    return float(ann_active_return / ann_active_risk)

# ==========================================
# ROLLING ACTIVE RETURN
# ==========================================

def rolling_active_return(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, periods_per_year: int = None, method: str = "arithmetic") -> Union[pd.Series, pd.DataFrame]:
    """Rolling annualized active return (strategy minus benchmark).

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from the index.
        method: 'arithmetic' — rolling mean of excess returns, annualized.
                'geometric' — annualized compound return of each series, then differenced.

    Returns:
        pd.Series or pd.DataFrame of rolling annualized active returns.
    """
    # Standardize input
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns.copy()

    # Align on common index
    strat_df, bench = strat_df.align(benchmark_returns, join="inner", axis=0)

    # Infer annualization factor if not provided
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(strat_df.index)

    if method == "arithmetic":
        excess = strat_df.sub(bench, axis=0).dropna()
        result = excess.rolling(window=window).mean() * periods_per_year
    elif method == "geometric":
        ann_factor = periods_per_year / window
        rol_strat = np.exp(np.log1p(strat_df).rolling(window).sum()) ** ann_factor - 1
        rol_bench = np.exp(np.log1p(bench).rolling(window).sum()) ** ann_factor - 1
        result = rol_strat.sub(rol_bench, axis=0)
    else:
        raise ValueError(f"method must be 'arithmetic' or 'geometric', got '{method}'")

    result = result.dropna(how="all")

    # Return Series if single-column input
    if result.shape[1] == 1 and isinstance(strategy_returns, pd.Series):
        return result.iloc[:, 0]
    return result



# ==========================================
# ROLLING TRACKING ERROR (Active Risk)
# ==========================================

def rolling_active_risk(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, periods_per_year: int = None, method: str = "arithmetic") -> Union[pd.Series, pd.DataFrame]:
    """Rolling annualized tracking error (std of active returns).

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from the index.
        method: 'arithmetic' — rolling std of arithmetic excess returns, annualized by sqrt rule.
                'geometric' — rolling std of per-period geometric excess returns, annualized by sqrt rule.

    Returns:
        pd.Series or pd.DataFrame of rolling annualized tracking error.
    """
    # Standardize input
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns.copy()

    strat_df, bench = strat_df.align(benchmark_returns, join="inner", axis=0)

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(strat_df.index)

    if method == "arithmetic":
        excess = strat_df.sub(bench, axis=0)
        result = excess.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)
    elif method == "geometric":
        # Per-period geometric excess return: (1+r_strat)/(1+r_bench) - 1
        geo_excess = (1 + strat_df).div(1 + bench, axis=0) - 1
        result = geo_excess.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)
    else:
        raise ValueError(f"method must be 'arithmetic' or 'geometric', got '{method}'")

    result = result.dropna(how="all")

    if result.shape[1] == 1 and isinstance(strategy_returns, pd.Series):
        return result.iloc[:, 0]
    return result


# ==========================================
# ROLLING INFORMATION RATIO
# ==========================================

def rolling_information_ratio(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, periods_per_year: int = None) -> pd.DataFrame:
    """Compute rolling annualized information ratio for one or more strategies.

    The information ratio is defined as the annualized mean excess return
    divided by the annualized tracking error over a rolling window.

    Args:
        strategy_returns: Periodic returns in decimal form. Series name or
            DataFrame column names are used as output labels.
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.
        periods_per_year: Annualization factor. If None, inferred from
            the index frequency (252 daily, 52 weekly, 12 monthly,
            4 quarterly).

    Returns:
        DataFrame of rolling information ratios with one column per strategy.
    """
    rol_ar = rolling_active_return(strategy_returns, benchmark_returns,
        window, periods_per_year, method="arithmetic")
    rol_te = rolling_active_risk(strategy_returns, benchmark_returns,
        window, periods_per_year, method="arithmetic")

    # Ensure DataFrame for consistent division
    if isinstance(rol_ar, pd.Series):
        rol_ar = rol_ar.to_frame()
        rol_te = rol_te.to_frame()

    rolling_ir = (rol_ar / rol_te).replace([np.inf, -np.inf], np.nan).dropna(how="all")

    return rolling_ir

# ==========================================
# CAPTURE RATIOS
# ==========================================

def down_capture(returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series) -> Union[float, pd.Series]:
    """
    Calculates the Down Capture Ratio using the Geometric Mean method.

    Args:
        returns: Periodic returns (e.g., monthly) of the strategies (Series or DataFrame).
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.

    Returns:
        float or pd.Series: The ratios (e.g., 0.95 = 95%). Returns np.nan if no down markets occur.
    """
    scalar_input = isinstance(returns, pd.Series)
    if scalar_input:
        returns = returns.to_frame(name=returns.name or "Strategy")

    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()

    down_market = df[df['bench'] < 0]
    n = len(down_market)

    if n == 0:
        result = pd.Series(np.nan, index=returns.columns)
        return float(result.iloc[0]) if scalar_input else result

    # Geometric mean per-period return via log-returns: exp(mean(log(1+r))) - 1
    port_geo_avg = np.exp(np.log1p(down_market[returns.columns]).sum(axis=0) / n) - 1
    bench_geo_avg = np.exp(np.log1p(down_market['bench']).sum() / n) - 1

    if bench_geo_avg == 0:
        result = pd.Series(np.nan, index=returns.columns)
        return float(result.iloc[0]) if scalar_input else result

    ratio = port_geo_avg / bench_geo_avg

    return float(ratio.iloc[0]) if scalar_input else ratio

def up_capture(returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series) -> Union[float, pd.Series]:
    """
    Calculates the Up Capture Ratio using the Geometric Mean method.

    Args:
        returns: Periodic returns (e.g., monthly) of the strategies (Series or DataFrame).
        benchmark_returns (pd.Series): Periodic returns (e.g., monthly) of the benchmark.

    Returns:
        float or pd.Series: The ratios (e.g., 1.05 = 105%). Returns np.nan if no up markets occur.
    """
    scalar_input = isinstance(returns, pd.Series)
    if scalar_input:
        returns = returns.to_frame(name=returns.name or "Strategy")

    df = returns.copy()
    df['bench'] = benchmark_returns
    df = df.dropna()

    up_market = df[df['bench'] > 0]
    n = len(up_market)

    if n == 0:
        result = pd.Series(np.nan, index=returns.columns)
        return float(result.iloc[0]) if scalar_input else result

    # Geometric mean per-period return via log-returns: exp(mean(log(1+r))) - 1
    port_geo_avg = np.exp(np.log1p(up_market[returns.columns]).sum(axis=0) / n) - 1
    bench_geo_avg = np.exp(np.log1p(up_market['bench']).sum() / n) - 1

    if bench_geo_avg == 0:
        result = pd.Series(np.nan, index=returns.columns)
        return float(result.iloc[0]) if scalar_input else result

    ratio = port_geo_avg / bench_geo_avg

    return float(ratio.iloc[0]) if scalar_input else ratio

def capture_spread(returns: Union[pd.Series, pd.DataFrame],
    benchmark_returns: pd.Series) -> Union[float, pd.Series]:
    """
    Calculates the Capture Spread: Up Capture − Down Capture.

    A positive spread indicates asymmetric upside participation (captures
    more of rallies than drawdowns); a negative spread indicates the
    reverse. NaN propagates if either side is undefined.

    Args:
        returns: Periodic returns of the strategies (Series or DataFrame).
        benchmark_returns (pd.Series): Periodic returns of the benchmark.

    Returns:
        float or pd.Series: Up Capture minus Down Capture.
    """
    return up_capture(returns, benchmark_returns) - down_capture(returns, benchmark_returns)


# ==========================================
# ROLLING CAPTURE RATIOS
# ==========================================

def _rolling_capture(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int, static_fn) -> Union[pd.Series, pd.DataFrame]:
    """Shared machinery for rolling capture ratios: slides `window` across the
    aligned series and delegates each slice to a static capture function."""
    if isinstance(strategy_returns, pd.Series):
        strategy_name = strategy_returns.name or "Strategy"
        strat_df = strategy_returns.to_frame(name=strategy_name)
    else:
        strat_df = strategy_returns.copy()

    strat_df, bench = strat_df.align(benchmark_returns, join="inner", axis=0)

    if len(strat_df) < window:
        empty = pd.DataFrame(columns=strat_df.columns)
        return empty.iloc[:, 0] if isinstance(strategy_returns, pd.Series) else empty

    records = {
        strat_df.index[i]: static_fn(strat_df.iloc[i - window + 1 : i + 1],
                                     bench.iloc[i - window + 1 : i + 1])
        for i in range(window - 1, len(strat_df))
    }

    result = pd.DataFrame(records).T.dropna(how="all")

    if result.shape[1] == 1 and isinstance(strategy_returns, pd.Series):
        return result.iloc[:, 0]
    return result


def rolling_up_capture(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int) -> Union[pd.Series, pd.DataFrame]:
    """Rolling Up Capture Ratio using the Geometric Mean method.

    Within each rolling window, only periods where the benchmark return is
    positive are considered. The ratio is the geometric-mean strategy return
    over those up-market periods divided by the geometric-mean benchmark return
    over the same periods.

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.

    Returns:
        pd.Series or pd.DataFrame of rolling up capture ratios. NaN where a
        window contains no up-market periods or the benchmark geo-mean is zero.
    """
    return _rolling_capture(strategy_returns, benchmark_returns, window, up_capture)


def rolling_down_capture(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int) -> Union[pd.Series, pd.DataFrame]:
    """Rolling Down Capture Ratio using the Geometric Mean method.

    Within each rolling window, only periods where the benchmark return is
    negative are considered. The ratio is the geometric-mean strategy return
    over those down-market periods divided by the geometric-mean benchmark
    return over the same periods.

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.

    Returns:
        pd.Series or pd.DataFrame of rolling down capture ratios. NaN where a
        window contains no down-market periods or the benchmark geo-mean is zero.
    """
    return _rolling_capture(strategy_returns, benchmark_returns, window, down_capture)


def rolling_capture_spread(strategy_returns: Union[pd.Series, pd.DataFrame], benchmark_returns: pd.Series,
    window: int) -> Union[pd.Series, pd.DataFrame]:
    """Rolling Capture Spread: rolling Up Capture − rolling Down Capture.

    A positive spread indicates asymmetric upside participation over the
    window (captures more of rallies than drawdowns); a negative spread
    indicates the reverse. NaN propagates from either side.

    Args:
        strategy_returns: Periodic returns in decimal form (Series or DataFrame).
        benchmark_returns: Periodic returns for the benchmark in decimal form.
        window: Rolling window size in periods.

    Returns:
        pd.Series or pd.DataFrame of rolling capture spreads.
    """
    return _rolling_capture(strategy_returns, benchmark_returns, window, capture_spread)


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
    capture = up/down
    spread = up - down
    
    # 2. Combine into a clean summary DataFrame
    summary_df = pd.DataFrame({
        "Up Capture": up,
        "Down Capture": down,
        "Overall Capture":capture,
        "Spread": spread
    })


    return summary_df
