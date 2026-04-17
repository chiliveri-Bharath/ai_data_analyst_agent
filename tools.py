import pandas as pd

def get_schema(df):
    return str(df.dtypes)

def get_summary(df):
    return df.describe(include='all').to_string()

def get_columns(df):
    return list(df.columns)

def get_head(df):
    return df.head().to_string()