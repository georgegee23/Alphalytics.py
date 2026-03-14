
import pandas as pd


def compute_forward_returns(returns: pd.DataFrame, forward_periods: int) -> pd.DataFrame:
    """Compute cumulative forward returns over a specified horizon for each asset.

    Args:
        returns: DataFrame of simple returns (not log returns), indexed by date.
        forward_periods: Number of periods ahead to compute the forward return.

    Returns:
        DataFrame of forward returns, aligned with the original index.
    """
    # Compute cumulative product of (1 + returns)
    cumulative_growth = (returns + 1).cumprod()
    # Compute cumulative forward returns: (future value / current value) - 1
    forward_returns = (cumulative_growth.shift(-forward_periods) / cumulative_growth) - 1

    return forward_returns
