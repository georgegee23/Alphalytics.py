
import pandas as pd
import numpy as np

from alphalytics.utils import _infer_periods_per_year


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
            "Align them before calling beta()."
        )

    betas = {}
    for col in returns.columns:
        # Pairwise: a gap in one strategy does not shrink the others,
        # and the covariance and benchmark variance share one sample size.
        pair = pd.concat([returns[col], benchmark], axis=1).dropna()
        if len(pair) < 2:
            betas[col] = np.nan
            continue
        y = pair.iloc[:, 0]
        x = pair.iloc[:, 1]
        bm_variance = x.var(ddof=1)
        if bm_variance == 0 or np.isclose(bm_variance, 0, atol=1e-15):
            betas[col] = np.nan
            continue
        betas[col] = y.cov(x) / bm_variance

    return pd.Series(betas, index=returns.columns, dtype=float)

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

def compute_capm(returns: pd.DataFrame, benchmark: pd.Series = None, periods_per_year: int = None) -> pd.DataFrame:
    """
    Calculates CAPM Beta and Annualized Alpha for multiple strategies simultaneously.

    Args:
        returns (pd.DataFrame): Periodic returns of the strategies.
        benchmark (pd.Series, optional): Periodic returns of the benchmark.
                                         Defaults to an equal-weight average of all strategies.
        periods_per_year (int, optional): Periods per year (e.g., 252 for daily, 12 for monthly).
                                Inferred from the returns' index when None.

    Benchmark alignment follows the same contract as beta(): returns and
    benchmark must share the same index. Annualized Alpha uses the linear
    Jensen's-alpha convention (Periodic Alpha * periods_per_year).

    Returns:
        pd.DataFrame: A summary table with Beta, Periodic Alpha, and Annualized Alpha for each strategy.

    Raises:
        TypeError: If inputs are not the expected pandas types.
        ValueError: If returns is empty, or returns and benchmark do not share the same index.
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

    if not returns.index.equals(benchmark.index):
        raise ValueError(
            "returns and benchmark must share the same index. "
            "Align them before calling compute_capm()."
        )

    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns.index)

    # 2. Align data and drop periods with missing values
    data = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1).dropna()
    aligned_returns = data[returns.columns]
    aligned_bench = data["__benchmark__"]

    empty_result = pd.DataFrame(
        {"Beta": np.nan, "Periodic Alpha": np.nan, "Annualized Alpha": np.nan},
        index=returns.columns,
    )
    if len(data) < 2:
        return empty_result

    # 3. Beta (reuses the validated, zero-variance-guarded implementation)
    betas = beta(aligned_returns, aligned_bench)

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

    The window stays on the original calendar. A missing row is not dropped
    before the roll, so it cannot pull a later return into an earlier window.
    A window with fewer than ``window`` paired observations is NaN.
    """
    # 1. Align data. Keep gaps in place.
    combined = pd.concat([returns, benchmark.rename("__benchmark__")], axis=1)
    aligned_returns = combined[returns.columns]
    aligned_bench = combined["__benchmark__"]

    # 2. Vectorized rolling covariance and variance
    # pandas smartly broadcasts the benchmark Series against every column in the DataFrame
    rolling_cov = aligned_returns.rolling(window=window, min_periods=window).cov(aligned_bench)
    rolling_var = aligned_bench.rolling(window=window, min_periods=window).var()

    # 3. Divide to get rolling beta (div(axis=0) ensures correct date alignment)
    rolling_betas = rolling_cov.div(rolling_var, axis=0)
    # Same flat-benchmark guard as beta(), including near-zero variance.
    flat = np.isclose(rolling_var.to_numpy(dtype=float), 0.0, atol=1e-15)
    rolling_betas.loc[flat] = np.nan

    return rolling_betas
