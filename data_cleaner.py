import pandas as pd
import numpy as np
import re
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer

# 🔍 Missing values summary
def missing_summary(df):
    return df.isnull().sum()

# 📄 Dataset Summary (FULL)
def dataset_summary(df):
    summary = {}
    summary["Shape"] = df.shape
    summary["Data Types"] = df.dtypes
    summary["Missing Values"] = df.isnull().sum()
    summary["Duplicates"] = df.duplicated().sum()
    summary["Numeric Summary"] = df.describe()
    summary["Categorical Summary"] = df.describe(include='object') if not df.select_dtypes(include='object').empty else "No categorical columns"
    return summary

# 🧹 Fill missing values (SMART)
def fill_missing(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

# 🤖 KNN Imputer (Numerical)
def knn_impute(df, n_neighbors=5):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

# 🧩 Simple Imputer (Categorical)
def simple_impute(df, strategy='most_frequent'):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        imputer = SimpleImputer(strategy=strategy)
        df[cat_cols] = imputer.fit_transform(df[cat_cols])
    return df

# 🧹 Drop duplicates
def drop_duplicates(df):
    return df.drop_duplicates()

# ⚠️ Detect outliers (Indices and Bounds)
def detect_outliers(df, column, method='iqr', threshold=3):
    if column not in df.columns or not np.issubdtype(df[column].dtype, np.number):
        return None
    
    data = df[column].dropna()
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    else: # zscore
        z_scores = np.abs(stats.zscore(data))
        mean = data.mean()
        std = data.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return {
        "indices": outliers.index.tolist(),
        "lower_bound": lower,
        "upper_bound": upper,
        "count": len(outliers),
        "min": outliers[column].min() if len(outliers) > 0 else None,
        "max": outliers[column].max() if len(outliers) > 0 else None
    }

# ⚠️ Cap outliers (Winsorization)
def cap_outliers(df, column, method='iqr', threshold=3):
    status = detect_outliers(df, column, method, threshold)
    if not status: return df
    
    df = df.copy()
    lower, upper = status["lower_bound"], status["upper_bound"]
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df

# ⚠️ Remove outliers (Direct)
def remove_outliers(df, column, method='iqr', threshold=3):
    status = detect_outliers(df, column, method, threshold)
    if not status or status["count"] == 0: return df
    
    return df.drop(index=status["indices"])

# 🔤 Clean text columns
def clean_text(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

# 🔄 Convert datatypes automatically
def fix_datatypes(df):
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

# 📅 Convert to Datetime
def convert_to_datetime(df, column):
    df = df.copy()
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

# 🏷 Label Encoding
def encode_labels(df, column=None):
    """Encodes a single column or all categorical columns if none specified."""
    df = df.copy()
    if column and column != "all" and column in df.columns:
        df[column] = df[column].astype('category').cat.codes
    elif not column or column == "all":
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype('category').cat.codes
    return df

# 📏 Min-Max Scaling [0, 1]
def min_max_scale(df, column=None):
    """Scales a single column or all numerical columns to [0, 1]."""
    df = df.copy()
    if column and column != "all" and column in df.columns and np.issubdtype(df[column].dtype, np.number):
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif not column or column == "all":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# 📏 Standard Scaling (Mean=0, Std=1)
def standard_scale(df, column=None):
    """Scales a single column or all numerical columns to Mean=0, Std=1."""
    df = df.copy()
    if column and column != "all" and column in df.columns and np.issubdtype(df[column].dtype, np.number):
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    elif not column or column == "all":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df[col] = (df[col] - mean_val) / std_val
    return df

# 🧠 FULL AUTO CLEAN
def auto_clean(df):
    df = drop_duplicates(df)
    df = clean_text(df)
    df = fix_datatypes(df)
    df = fill_missing(df)
    for col in df.select_dtypes(include='number').columns:
        df = remove_outliers(df, col)
    return df

# 🗑 Remove Column
def remove_column(df, column):
    if column in df.columns:
        return df.drop(columns=[column])
    return df

# ✏ Rename Column
def rename_column(df, old_name, new_name):
    if old_name in df.columns:
        return df.rename(columns={old_name: new_name})
    return df

# 🔢 Sort Data
def sort_data(df, column, ascending=True):
    if column in df.columns:
        return df.sort_values(by=column, ascending=ascending)
    return df

# 📉 Drop NA Columns (Advanced)
def drop_na_columns(df, threshold=0.5):
    """Drops columns where missing value ratio exceeds threshold."""
    limit = len(df) * threshold
    return df.dropna(axis=1, thresh=len(df)-limit)

# 📉 Drop NA Rows
def drop_na_rows(df, column=None):
    """Drops rows with missing values globally or for a specific column."""
    if column and column in df.columns:
        return df.dropna(subset=[column])
    return df.dropna()

# 📏 Standardize Names (Advanced)
def standardize_names(df):
    """Converts column names to snake_case and removes special characters."""
    def clean_name(name):
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        return name.lower()
    
    df = df.copy()
    df.columns = [clean_name(col) for col in df.columns]
    return df

# ✂ Strip Whitespace (Advanced)
def strip_whitespace(df):
    """Strips leading/trailing whitespace from all string columns."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

# ➗ Split Column
def split_column(df, column, delimiter=','):
    """Splits a column by a delimiter into multiple columns."""
    df = df.copy()
    if column in df.columns:
        expanded = df[column].astype(str).str.split(delimiter, expand=True)
        expanded.columns = [f"{column}_{i+1}" for i in range(expanded.shape[1])]
        return pd.concat([df.drop(columns=[column]), expanded], axis=1)
    return df

# ➗ Calculate New Feature
def create_calculated_feature(df, col1, col2, op, new_name):
    """Performs common math between two columns and stores in a new column."""
    df = df.copy()
    if col1 in df.columns and col2 in df.columns:
        if op == '+': df[new_name] = df[col1] + df[col2]
        elif op == '-': df[new_name] = df[col1] - df[col2]
        elif op == '*': df[new_name] = df[col1] * df[col2]
        elif op == '/': df[new_name] = df[col1] / df[col2]
    return df

# 📅 Extract Date Features
def extract_date_parts(df, column):
    """Extracts year, month, day, and weekday from a date column."""
    df = df.copy()
    if column in df.columns:
        # Convert to datetime first if not already
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_weekday"] = df[column].dt.weekday
    return df

# 🏷️ One-Hot Encoding
def one_hot_encode(df, column=None):
    """Creates binary dummy variables for a single column or all categorical columns."""
    df = df.copy()
    if column and column != "all" and column in df.columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
    elif not column or column == "all":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            df = pd.get_dummies(df, columns=cat_cols)
    return df