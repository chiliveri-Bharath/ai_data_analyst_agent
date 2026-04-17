import pandas as pd
import numpy as np

def groupby_analysis(df, column):
    return df.groupby(column).mean(numeric_only=True)

def get_top_bottom(df, column, n=5, mode="both"):
    """Returns top, bottom, or both values for a column."""
    if column not in df.columns:
        return None
        
    # If numeric → use nlargest and nsmallest
    if np.issubdtype(df[column].dtype, np.number):
        res = {}
        if mode in ["top", "both"]:
            res["top"] = df.nlargest(n, column)
        if mode in ["bottom", "both"]:
            res["bottom"] = df.nsmallest(n, column)
        return res

    # If categorical → use value_counts
    else:
        counts = df[column].value_counts()
        res = {}
        if mode in ["top", "both"]:
            res["top"] = counts.head(n)
        if mode in ["bottom", "both"]:
            res["bottom"] = counts.tail(n)
        return res

def calculate_stat(df, column, stat_type):
    """Performs basic statistical aggregations (average, sum, min, max)."""
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return None
        
    if stat_type == "average": return df[column].mean()
    elif stat_type == "sum": return df[column].sum()
    elif stat_type == "min": return df[column].min()
    elif stat_type == "max": return df[column].max()
    return None

def get_value_counts(df, column):
    """Returns value counts for a column."""
    if column in df.columns:
        return df[column].value_counts()
    return None