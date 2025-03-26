
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



 # ============== THE END ============== #     