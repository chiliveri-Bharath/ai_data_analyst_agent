def generate_insights(df):

    insights = []

    # Missing values
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            insights.append(f"{col} has {val} missing values")

    # Numeric trends
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        mean = df[col].mean()
        max_val = df[col].max()
        min_val = df[col].min()

        insights.append(f"{col}: avg={mean:.2f}, max={max_val}, min={min_val}")

    return insights