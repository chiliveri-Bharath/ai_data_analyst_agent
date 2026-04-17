import pandas as pd

# 📊 Average
def get_average(df, column):
    return df[column].mean()

# 📊 Sum
def get_sum(df, column):
    return df[column].sum()

# 📊 Min / Max
def get_min(df, column):
    return df[column].min()

def get_max(df, column):
    return df[column].max()

# 📈 Rolling Mean
def rolling_mean(df, column):
    return df[column].rolling(window=3).mean()

# 📈 Cumulative Sum
def cumulative_sum(df, column):
    return df[column].cumsum()

# 🔍 Filter
def filter_data(df, column, value):
    return df[df[column] > value]

# 🧠 Unique Values
def unique_values(df, column):
    return df[column].unique()

# 📊 Data Types
def data_types(df):
    return df.dtypes

# 📊 Null Count
def null_count(df):
    return df.isnull().sum()