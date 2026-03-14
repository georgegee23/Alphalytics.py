
import pandas as pd
import numpy as np


# ============== CAPM METRICS ============== #

def beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """
    Calculates beta for multiple strategies against a benchmark.

    Beta = Cov(Strategy, Benchmark) / Var(Benchmark) using sample statistics (ddof=1).
    Requires at least 2 observations; otherwise returns NaN for each strategy.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies (columns = strategy names).
        benchmark (pd.Series): Periodic returns of the benchmark.

    Returns:
        pd.Series: Beta for each strategy.

    Raises:
        ValueError: If returns and benchmark do not share the same index.
    """
    if not returns.index.equals(benchmark.index):
        raise ValueError(
            "returns and benchmark must share the same index. "
            "Align them before calling calculate_beta()."
        )

    if len(benchmark) < 2:
        return pd.Series(np.nan, index=returns.columns)

    bm_variance = benchmark.var(ddof=1)

    if bm_variance == 0:
        return pd.Series(np.nan, index=returns.columns)

    ret_centered = returns - returns.mean()
    bm_centered = benchmark - benchmark.mean()

    covariances = ret_centered.mul(bm_centered, axis=0).sum() / (len(benchmark) - 1)

    return covariances / bm_variance

def bull_bear_beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Bull Market Beta (benchmark > 0) and Bear Market Beta (benchmark < 0)
    for each strategy.

    Computed on the common overlapping period only, so all strategies are
    comparable. Zero-return benchmark periods are excluded from both regimes.
    NaN is returned for a regime with fewer than 2 observations.

    Args:
        returns (pd.DataFrame): Periodic returns, one column per strategy.
        benchmark (pd.Series): Periodic benchmark returns.

    Returns:
        pd.DataFrame: Bull Beta and Bear Beta per strategy (index = strategy names).

    Raises:
        TypeError: If inputs are not the expected pandas types.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()

    if combined.empty:
        return pd.DataFrame(
            {"Bull Beta": np.nan, "Bear Beta": np.nan},
            index=returns.columns,
        )

    aligned_returns = combined[returns.columns]
    aligned_bench   = combined["__benchmark__"]

    bull_periods = aligned_bench > 0
    bear_periods = aligned_bench < 0

    bull_beta = beta(aligned_returns[bull_periods], aligned_bench[bull_periods])
    bear_beta = beta(aligned_returns[bear_periods], aligned_bench[bear_periods])

    return pd.DataFrame({"Bull Beta": bull_beta, "Bear Beta": bear_beta})

def compute_capm(returns: pd.DataFrame, benchmark: pd.Series = None, periods_per_year: int = 12) -> pd.DataFrame:
    """
    Calculates CAPM Beta and Annualized Alpha for multiple strategies simultaneously.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark (pd.Series, optional): Periodic returns of the benchmark.
                                         Defaults to an equal-weight average of all strategies.
        periods_per_year (int): Periods per year (e.g., 252 for daily, 12 for monthly).
                                Defaults to 12 (monthly).

    Returns:
        pd.DataFrame: A summary table with Beta, Periodic Alpha, and Annualized Alpha for each strategy.
    """
    # 1. Validate inputs
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    if benchmark is None:
        benchmark = returns.mean(axis=1)
        benchmark.name = "Benchmark"
    elif not isinstance(benchmark, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(benchmark).__name__}")

    # 2. Align data and handle missing values naturally
    data = returns.copy()
    data['bench_'] = benchmark
    data = data.dropna()

    aligned_returns = data.drop(columns=['bench_'])
    aligned_bench = data['bench_']

    # 3. Vectorized CAPM Math
    returns_centered = aligned_returns - aligned_returns.mean()
    bench_centered = aligned_bench - aligned_bench.mean()

    dof = len(data) - 1
    covariances = (returns_centered.mul(bench_centered, axis=0)).sum() / dof
    bench_variance = aligned_bench.var(ddof=1)

    # Calculate Beta
    betas = covariances / bench_variance

    # Calculate Periodic Alpha, then Annualize it
    # Annualized Alpha = Periodic Alpha * Periods per Year
    periodic_alphas = aligned_returns.mean() - (betas * aligned_bench.mean())
    annualized_alphas = periodic_alphas * periods_per_year

    # 4. Format Output
    capm_df = pd.DataFrame({
        "Beta": betas,
        "Periodic Alpha": periodic_alphas,
        "Annualized Alpha": annualized_alphas
    })

    return capm_df


# ============== ROLLING METRICS ================== #

def rolling_beta(returns: pd.DataFrame, benchmark: pd.Series, window: int = 12) -> pd.DataFrame:
    """
    Calculates rolling beta using native vectorized pandas methods.
    Mathematically identical to Cov(R, B) / Var(B) with ddof=1.
    """
    # 1. Align data
    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()
    aligned_returns = combined[returns.columns]
    aligned_bench = combined["__benchmark__"]

    # 2. Vectorized rolling covariance and variance
    # pandas smartly broadcasts the benchmark Series against every column in the DataFrame
    rolling_cov = aligned_returns.rolling(window=window).cov(aligned_bench)
    rolling_var = aligned_bench.rolling(window=window).var()

    # 3. Divide to get rolling beta (div(axis=0) ensures correct date alignment)
    rolling_betas = rolling_cov.div(rolling_var, axis=0)
    rolling_betas[rolling_var == 0] = np.nan  # guard flat-benchmark windows

    return rolling_betas
