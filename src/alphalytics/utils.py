
import pandas as pd
import numpy as np



# =============== CLEAN DATA ============== #

def detect_extreme_outliers(df:pd.DataFrame, iqr_multiplier:int=4, threshold_percentage=0.10) -> list:

    """
    Detects columns with extreme outliers in a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze for outliers
    iqr_multiplier : float, default 3.0
        Multiplier for IQR to identify extreme outliers (standard is 1.5, 3.0 for extreme)
    threshold_percentage : float, default 0.05
        Minimum percentage of extreme outliers required to flag a column
        
    Returns:
    --------
    list
        Names of columns containing extreme outliers
    """
    outlier_columns = []
    
    # Process only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Skip columns with all NaN values
        if df[col].isna().all():
            continue
        
        # Calculate Q1, Q3, and IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Define lower and upper bounds for extreme outliers
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        # Calculate percentage of outliers
        outlier_percentage = len(outliers) / len(df[col].dropna())
        
        # Add column to the list if it exceeds the threshold
        if outlier_percentage >= threshold_percentage:
            outlier_columns.append(col)
    
    return outlier_columns


def detect_internal_nan(df: pd.DataFrame) -> list[str]:
    """
    Detect columns in a DataFrame that have internal NaN values.

    An "internal NaN" means the column (treated as a Series) starts and ends with non-NaN values
    but has at least one NaN in between. This indicates interruptions in the data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to check. Each column should be numeric or support NaN checks.

    Returns
    -------
    list[str]
        A list of column names that have internal NaNs.

    Examples
    --------
    >>> test_dict = {"A": [0.01, 0.02, 0.03, -0.02, 0.03, 0.05], 
    ...              "B": [0.03, 0.01, -0.01, np.nan, np.nan, np.nan],
    ...              "C": [np.nan, np.nan, np.nan, 0.03, -0.01, 0.04],
    ...              "D": [0.01, np.nan, np.nan, np.nan, 0.05, 0.06],
    ...              "E": [0.01, np.nan, 0.03, np.nan, 0.05, 0.06]}
    >>> test_df = pd.DataFrame(test_dict, index=pd.date_range("2023-01-31", periods=6, freq="ME"))
    >>> detect_internal_nan_columns(test_df)
    ['D', 'E']

    Notes
    -----
    - Assumes the DataFrame columns are Series with at least 3 elements (shorter columns can't have "internal" NaNs).
    - Uses pd.isna() for NaN detection, which works for float, object, and other dtypes supporting NaNs.
    - Does not modify the input DataFrame.
    """
    
    def has_internal_nans(s: pd.Series) -> bool:
        # Helper function to check a single Series for internal NaNs
        if len(s) < 3:  # Too short to have interruptions "between" values
            return False
        
        has_any_nan = s.isna().any()  # Check for any NaNs at all
        first_not_nan = not pd.isna(s.iloc[0])  # First value is not NaN
        last_not_nan = not pd.isna(s.iloc[-1])  # Last value is not NaN
        
        # If all conditions are true, NaNs must be internal
        return has_any_nan and first_not_nan and last_not_nan
    
    # List to collect qualifying column names
    columns_with_internal_nans = []
    
    # Iterate over each column and apply the check
    for col in df.columns:
        if has_internal_nans(df[col]):
            columns_with_internal_nans.append(col)
    
    return columns_with_internal_nans


def fill_first_nan(series: pd.Series, value: float = 1.0) -> pd.Series:
    """
    Fill the first NaN value in a time series with a specified value.
    
    Parameters
    ----------
    series : pd.Series
        Time series data with datetime index
    value : float, default 1.0
        Value to fill the first NaN with
        
    Returns
    -------
    pd.Series
        Series with first NaN filled
        
    Raises
    ------
    TypeError
        If input is not a pandas Series
    ValueError
        If series is empty or has no NaN values
    """
    # Input validation
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if len(series) == 0:
        raise ValueError("Series is empty")
        
    if not series.isna().any():
        return series
    
    # Find first NaN date
    first_nan_idx = series.index[series.isna()][0]
    
    # Create copy to avoid modifying original
    result = series.copy()
    result.loc[first_nan_idx] = value
    
    return result

 # ============== THE END ============== #     