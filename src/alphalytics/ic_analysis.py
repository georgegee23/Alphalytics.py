
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, wilcoxon, binomtest, t, ttest_1samp, skew, kurtosis


# ============== INFORMATION COEFFICIENT ANALYSIS ============== #

def cs_spearmanr(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute cross-sectional Spearman rank correlations between factors and returns.

    This function calculates the Spearman rank correlation for each row (date) across
    columns (assets), measuring the monotonic relationship between factor scores and
    returns at each time period. Useful for information coefficient (IC) analysis in
    quantitative finance.

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with dates as index and assets as columns, containing factor values.
    returns : pd.DataFrame
        DataFrame with dates as index and assets as columns, containing return values.
        Must have the same shape and index as `factor`.

    Returns
    -------
    pd.Series
        Series of Spearman correlations, indexed by date. NaN for dates with fewer than
        2 valid (non-NaN) asset pairs.

    Notes
    -----
    - Handles NaNs pairwise per date.
    - Assumes aligned DataFrames; align manually if needed.
    - For small cross-sections (e.g., few assets), results may be unstable or extreme.
    """
    return factor.corrwith(returns, axis=1, method='spearman')


def ts_spearmanr(factor: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute time-series Spearman rank correlations between factors and returns.

    This function calculates the Spearman rank correlation for each column (asset) across
    rows (dates), measuring the monotonic relationship between an asset's factor scores
    and its returns over time.

    Parameters
    ----------
    factor : pd.DataFrame
        DataFrame with dates as index and assets as columns, containing factor values.
    returns : pd.DataFrame
        DataFrame with dates as index and assets as columns, containing return values.
        Must have the same shape and index as `factor`.

    Returns
    -------
    pd.Series
        Series of Spearman correlations, indexed by asset. NaN for assets with fewer than
        2 valid (non-NaN) dates.

    Notes
    -----
    - Handles NaNs pairwise per asset.
    - Assumes aligned DataFrames; align manually if needed.
    - Useful for asset-specific diagnostics rather than broad factor evaluation.
    """
    return factor.corrwith(returns, axis=0, method='spearman')


def compute_ic_stats(factors: pd.DataFrame, returns: pd.DataFrame, alternative: str = 'greater', round_digits: int = 4) -> pd.DataFrame:
    """
    Compute Spearman rank correlation statistics between factor scores and returns.
    
    This function calculates various statistical measures based on the
    cross-sectional Spearman rank correlations (Information Coefficients) between factor scores 
    and returns. These statistics help evaluate the predictive power and 
    consistency of factor scores for financial returns.
    
    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame containing factor values. Each column represents an asset factor value and each row
        represents a time period. The index should be time-based.
    
    returns : pd.DataFrame
        DataFrame containing return values. Each column represents an asset's returns and 
        each row represents a time period. Must have the same index as `factors`.

    alternative : str, optional
        Defines the alternative hypothesis for statistical tests:
        - 'two-sided': Tests if mean/median IC != 0.
        - 'greater': Tests if mean/median IC > 0 (expected positive IC) (default).
        - 'less': Tests if mean/median IC < 0 (expected negative IC).

    round_digits : int, optional
        Number of decimal places to round the output statistics to. Default is 4.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the following statistics:
        - Mean: Average Information Coefficient (IC)
        - Std: Standard deviation of IC
        - RAIC: Risk-adjusted IC (Mean/Std)
        - Skew: Skewness of IC distribution
        - Kurtosis: Kurtosis of IC distribution
        - T Pval: p-value from t-test based on alternative
        - Wcx Pval: p-value from Wilcoxon signed-rank test based on alternative
        - Hit Rate: Percentage of periods with positive IC
        - HR Pval: p-value for hit rate based on alternative (greater/less than 50%, or !=)
        
    Notes
    -----
    - Assumes concurrent correlation; lag factors externally for predictive analysis to avoid look-ahead bias.
    - The function requires the `cs_spearmanr` function to calculate 
      correlations at each time step.
    - NaN values in the correlation time series are dropped before statistics are computed.
    - For inverted factors (expected negative IC), use 'less' alternative or negate factors.
    """
    # Input validation
    if not isinstance(factors, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError("Both factors and returns must be pandas DataFrames")
        
    if not factors.index.equals(returns.index):
        raise ValueError("Factors and returns must have the same index")
        
    if factors.empty or returns.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    # Calculate cross-sectional Spearman rank correlations at each time step
    cs_spearmanr_df = cs_spearmanr(factors, returns).dropna()
    
    if cs_spearmanr_df.empty:
        raise ValueError("No valid data points after computing correlations and removing NaNs")

    # Calculate statistics from the time series of correlations
    ic_series = cs_spearmanr_df
    sample_size = len(ic_series)
    
    mean_corr = ic_series.mean()
    std_corr = ic_series.std()
    
    # Avoid division by zero
    raic = mean_corr / std_corr if std_corr != 0 else np.nan
    t_stat = mean_corr / (std_corr / np.sqrt(sample_size)) if std_corr != 0 else np.nan
    
    # Calculate t-test p-value based on alternative
    if np.isnan(t_stat):
        t_pval = np.nan
    else:
        if alternative == 'two-sided':
            t_pval = 2 * (1 - t.cdf(abs(t_stat), sample_size - 1))
        elif alternative == 'greater':
            t_pval = 1 - t.cdf(t_stat, sample_size - 1)
        elif alternative == 'less':
            t_pval = t.cdf(t_stat, sample_size - 1)

    # Wilcoxon Signed-Rank Test
    if sample_size > 0 and not all(ic_series == 0):  # Check for non-zero ICs
        w_stat, wilcoxon_pval = wilcoxon(ic_series, alternative=alternative, zero_method="wilcox")
    else:
        w_stat, wilcoxon_pval = np.nan, np.nan
    
    ic_skew = skew(ic_series)
    ic_kurtosis = kurtosis(ic_series)

    n_positive = (ic_series > 0).sum()
    n_total = len(ic_series)
    
    # Adjust binomtest for directional: test p != 0.5 (two-sided), p > 0.5 (greater), p < 0.5 (less)
    if alternative == 'two-sided':
        sign_pval = binomtest(n_positive, n_total, p=0.5, alternative="two-sided").pvalue
    elif alternative == 'greater':
        sign_pval = binomtest(n_positive, n_total, p=0.5, alternative="greater").pvalue
    elif alternative == 'less':
        sign_pval = binomtest(n_positive, n_total, p=0.5, alternative="less").pvalue
    
    hit_rate = (ic_series > 0).mean()  # Percentage of positive ICs
    
    #Create dictionary to store results
    ic_stats_dict = {
        "IC Stats": [
            mean_corr, 
            std_corr,
            raic,
            ic_skew,
            ic_kurtosis,
            t_pval,
            wilcoxon_pval,
            hit_rate,
            sign_pval
        ]
    }

    # Create DataFrame with statistics
    col_names = ["Mean", "Std", "RAIC", "Skew", "Kurtosis", "T Pval", "Wcx Pval", "Hit Rate", "HR Pval"]
    ic_stats_df = pd.DataFrame.from_dict(ic_stats_dict, orient="index", columns=col_names).round(round_digits)
    
    return ic_stats_df


# ============== FACTOR INFORMATION DECAY ANALYSIS ============== #

def compute_forward_returns(returns: pd.DataFrame, forward_periods: int) -> pd.DataFrame:
    """
    Compute cumulative forward returns over a specified horizon for each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of simple returns (not log returns), indexed by date.
    forward_periods : int
        Number of periods ahead to compute the forward return.

    Returns
    -------
    pd.DataFrame
        DataFrame of forward returns, aligned with the original index.
    """
    # Compute cumulative product of (1 + returns)
    cumulative_growth = (returns + 1).cumprod()
    # Compute cumulative forward returns: (future value / current value) - 1
    forward_returns = (cumulative_growth.shift(-forward_periods) / cumulative_growth) - 1

    return forward_returns


def factor_decay(factors:pd.DataFrame, returns:pd.DataFrame, max_horizon:int) -> pd.DataFrame:

    # Validate inputs
    assert factors.shape == returns.shape, "Factors and returns must have same dimensions"
    assert max_horizon > 0, "max_horizon must be positive"

    ic_decay = []
    p_values = []
    
    for h in range(1, max_horizon + 1):
        # Compute forward returns for horizon h: cumulative returns from t+1 to t+h
        forward_rets = compute_forward_returns(returns, h)
        
        # Compute cross-sectional Spearman correlation at each time t
        ic_series = cs_spearmanr(factors, forward_rets)
        
        # Drop NaN values
        ic_series = ic_series.dropna()
        
        if len(ic_series) > 1:
            mean_ic = ic_series.mean()
            t_stat, p_val = ttest_1samp(ic_series, 0)
            ic_decay.append(mean_ic)
            p_values.append(p_val)
        else:
            ic_decay.append(np.nan)
            p_values.append(np.nan)
    
    horizons = range(1, max_horizon + 1)
    return pd.DataFrame({'IC': ic_decay, 'p_value': p_values}, index=horizons)

 # ============== THE END ============== #     