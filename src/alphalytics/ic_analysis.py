
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, stats, t, ttest_1samp


# ============== INFORMATION COEFFICIENT ANALYSIS ============== #

def cross_sectional_spearmanr(factors: pd.DataFrame, returns: pd.DataFrame, factor_lag=0) -> pd.DataFrame:
    """
    Compute cross-sectional Spearman rank correlation between factors and returns over time.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame with dates as index and securities as columns, containing factor values.
    returns : pd.DataFrame
        DataFrame with dates as index and securities as columns, containing return values.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and columns 'SpearmanR' (correlation coefficient)
        and 'P-Value' (statistical significance).
    """

    # Shift factors by the specified lag
    factors = factors.shift(factor_lag)
    # Ensure indices align and preprocess NaN dropping outside the loop
    common_dates = factors.index.intersection(returns.index)
    factors_aligned = factors.loc[common_dates].to_numpy()
    returns_aligned = returns.loc[common_dates].to_numpy()
    
    # Preallocate results array
    result = np.full((len(common_dates), 2), np.nan)
    
    # Loop over rows (dates)
    for i in range(len(common_dates)):
        # Get paired data, drop NaNs
        paired_data = np.vstack((returns_aligned[i], factors_aligned[i])).T
        mask = ~np.isnan(paired_data).any(axis=1)
        valid_data = paired_data[mask]
        
        if len(valid_data) >= 2:
            corr, p_value = spearmanr(valid_data[:, 0], valid_data[:, 1])
            result[i] = [corr, p_value]
    
    return pd.DataFrame(result, index=common_dates, columns=["SpearmanR", "P-Value"]).dropna()


def compute_spearman_stats(factors: pd.DataFrame, returns: pd.DataFrame, 
                          factor_lag: int = 0) -> pd.DataFrame:
    """
    Compute Spearman rank correlation statistics between lagged factors and returns.
    
    This function calculates various statistical measures based on the time-series of 
    cross-sectional Spearman rank correlations (Information Coefficients) between factors 
    and subsequent returns. These statistics help evaluate the predictive power and 
    consistency of factors for financial returns.
    
    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame containing factor values. Each column represents a factor and each row
        represents a time period. The index should be time-based.
    
    returns : pd.DataFrame
        DataFrame containing return values. Each column represents an asset's returns and 
        each row represents a time period. Must have the same index as `factors`.
    
    lag : int, optional
        Number of periods to lag the factors by. Default is 1.
        
    
    Returns
    -------
    pd.DataFrame or tuple
        If return_ts is False, returns a DataFrame containing the following statistics:
        - Mean: Average Information Coefficient (IC)
        - Std: Standard deviation of IC
        - RA IC: Risk-adjusted IC (Mean/Std)
        - T-Stat: T-statistic of IC
        - P-Value: Two-sided p-value
        - IC Skew: Skewness of IC distribution
        - IC Kurtosis: Kurtosis of IC distribution
        - Hit Rate: Percentage of periods with positive IC
        
    Return: Spearman Statistics DataFrame
    
    Notes
    -----
    - Factors are lagged to ensure predictive calculations avoid look-ahead bias.
    - The function requires the `cross_sectional_spearmanr` function to calculate 
      correlations at each time step.
    - NaN values in the correlation time series are dropped before statistics are computed.
    """
    # Input validation
    if not isinstance(factors, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError("Both factors and returns must be pandas DataFrames")
        
    if not factors.index.equals(returns.index):
        raise ValueError("Factors and returns must have the same index")
        
    if factors.empty or returns.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    # Calculate cross-sectional Spearman rank correlations at each time step
    ts_spearmanr_df = cross_sectional_spearmanr(factors, returns, factor_lag=factor_lag).dropna()
    
    if ts_spearmanr_df.empty:
        raise ValueError("No valid data points after computing correlations and removing NaNs")

    # Calculate statistics from the time series of correlations
    ic_series = ts_spearmanr_df["SpearmanR"]
    sample_size = len(ic_series)
    
    mean_corr = ic_series.mean()
    std_corr = ic_series.std()
    
    # Avoid division by zero
    raic = mean_corr / std_corr if std_corr != 0 else np.nan
    t_stat = mean_corr / (std_corr / np.sqrt(sample_size)) if std_corr != 0 else np.nan
    t_pval = 2 * (1 - t.cdf(abs(t_stat), sample_size - 1)) if not np.isnan(t_stat) else np.nan

    # Wilcoxon Signed-Rank Test
    if sample_size > 0 and not all(ic_series == 0):  # Check for non-zero ICs
        w_stat, wilcoxon_pval = stats.wilcoxon(ic_series, alternative="two-sided", zero_method="wilcox")
    else:
        w_stat, wilcoxon_pval = np.nan, np.nan
    
    ic_skew = stats.skew(ic_series)
    ic_kurtosis = stats.kurtosis(ic_series)

    n_positive = (ic_series > 0).sum()
    n_total = len(ic_series.dropna())
    sign_pval = stats.binomtest(n_positive, n_total, p=0.5, alternative="two-sided").pvalue
    hit_rate = (ic_series > 0).mean()  # Percentage of positive ICs
    
    #Create dictionary to store results
    spearman_stats_dict = {
        "IC Stats": [
            mean_corr, 
            std_corr,
            raic,
            ic_skew,
            ic_kurtosis,
            t_pval,
            wilcoxon_pval,
            sign_pval,
            hit_rate
        ]
    }

    # Create DataFrame with statistics
    col_names = ["Mean", "Std", "RA IC", "IC Skew", "IC Kurtosis", "T Pval", "Wilcoxon Pval", "Sign Pval", "Hit Rate"]
    spearman_stats_df = pd.DataFrame.from_dict(spearman_stats_dict, orient="index", columns=col_names).round(4)
    
    return spearman_stats_df


# ============== FACTOR INFORMATION DECAY ANALYSIS ============== #

def compute_forward_returns(returns:pd.DataFrame, forward_periods:int) -> pd.DataFrame:

    # Compute cumulative growth and forward returns
    cumulative_growth = (returns + 1).cumprod()
    # Compute forward returns
    forward_returns = (cumulative_growth.shift(-forward_periods) / cumulative_growth) - 1 

    return forward_returns


def factor_decay(factors:pd.DataFrame, returns:pd.DataFrame, max_horizon:int) -> pd.DataFrame:

    # Validate inputs
    assert factors.shape == returns.shape, "Factors and returns must have same dimensions"
    assert max_horizon > 0, "max_horizon must be positive"

    ic_decay = []
    p_values = []
    
    for h in range(1, max_horizon + 1):
        # Compute forward returns for horizon h: sum of log returns from t+1 to t+h
        forward_rets = compute_forward_returns(returns, h)
        
        # Compute cross-sectional Spearman correlation at each time t
        ic_series = factors.corrwith(forward_rets, axis=1, method='spearman')
        
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