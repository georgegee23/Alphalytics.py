
import pandas as pd
import numpy as np



# =============== CLEAN DATA ============== #

def detect_extreme_outliers(df:pd.DataFrame, iqr_multiplier:int=4, threshold_percentage=0.10) -> list:

    """Detects columns that contain extreme outliers relative the column's values.

    Args:
        df: DataFrame to analyze for outliers.
        iqr_multiplier: Multiplier for IQR to identify extreme outliers
            (standard is 1.5, 3.0 for extreme).
        threshold_percentage: Minimum percentage of extreme outliers
            required to flag a column.

    Returns:
        List of column names containing extreme outliers.
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
    """Detect columns in a DataFrame that have internal NaN values.

    An "internal NaN" means the column (treated as a Series) starts and ends
    with non-NaN values but has at least one NaN in between. This indicates
    interruptions in the data.

    Args:
        df: The input DataFrame to check. Each column should be numeric
            or support NaN checks.

    Returns:
        A list of column names that have internal NaNs.

    Examples:
        >>> test_dict = {"A": [0.01, 0.02, 0.03, -0.02, 0.03, 0.05],
        ...              "B": [0.03, 0.01, -0.01, np.nan, np.nan, np.nan],
        ...              "C": [np.nan, np.nan, np.nan, 0.03, -0.01, 0.04],
        ...              "D": [0.01, np.nan, np.nan, np.nan, 0.05, 0.06],
        ...              "E": [0.01, np.nan, 0.03, np.nan, 0.05, 0.06]}
        >>> test_df = pd.DataFrame(test_dict, index=pd.date_range("2023-01-31", periods=6, freq="ME"))
        >>> detect_internal_nan(test_df)
        ['D', 'E']

    Note:
        - Assumes the DataFrame columns are Series with at least 3 elements
          (shorter columns can't have "internal" NaNs).
        - Uses pd.isna() for NaN detection, which works for float, object,
          and other dtypes supporting NaNs.
        - Does not modify the input DataFrame.
    """

    def has_internal_nans(s: pd.Series) -> bool:
        # Helper function to check a single Series for internal NaNs
        if len(s) < 3:  # Too short to have interruptions "between" values
            return False

        # Trim leading and trailing NaN values to find the valid data range
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        if first_valid is None or last_valid is None:
            return False  # All NaN

        trimmed = s.loc[first_valid:last_valid]

        # If the trimmed range has any NaN, those are internal gaps
        return trimmed.isna().any()

    # List to collect qualifying column names
    columns_with_internal_nans = []

    # Iterate over each column and apply the check
    for col in df.columns:
        if has_internal_nans(df[col]):
            columns_with_internal_nans.append(col)

    return columns_with_internal_nans


def fill_first_nan(series: pd.Series, value: float = 1.0) -> pd.Series:
    """Fill the first NaN value in a time series with a specified value.

    Args:
        series: Time series data with datetime index.
        value: Value to fill the first NaN with.

    Returns:
        Series with first NaN filled.

    Raises:
        TypeError: If input is not a pandas Series.
        ValueError: If series is empty or has no NaN values.
    """
    # Input validation
    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be a pd.Series, got {type(series).__name__}")

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


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the number of periods per year from a DatetimeIndex.

    Args:
        index: The datetime index to analyze.

    Returns:
        Estimated periods per year (252 for daily, 52 for weekly,
        12 for monthly, 4 for quarterly, 1 for annual).
    """
    if len(index) < 2:
        return 12  # default to monthly

    median_days = pd.Series(index).diff().dt.days.median()

    if median_days <= 3:
        return 252
    elif median_days <= 8:
        return 52
    elif median_days <= 35:
        return 12
    elif median_days <= 100:
        return 4
    else:
        return 1


__all__ = [
    "detect_extreme_outliers",
    "detect_internal_nan",
    "fill_first_nan",
]
