import pandas as pd
import numpy as np
from visualizer import (
    histogram, scatter_plot, correlation_heatmap, 
    boxplot, line_chart, pie_chart, bar_chart
)

def generate_dashboard(df):
    """
    Intelligently generates a collection of 4-6 premium charts 
    to provide a comprehensive overview of the dataset.
    """
    dashboard_items = []
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Correlation Heatmap (Always useful for overview if >1 numeric)
    if len(numeric_cols) > 1:
        dashboard_items.append({
            "title": "Variable Correlations",
            "figure": correlation_heatmap(df),
            "description": "This heatmap shows how different numeric variables relate to each other. High values (close to 1 or -1) indicate strong relationships."
        })

    # 2. Main Numeric Distribution (Histogram)
    if len(numeric_cols) > 0:
        main_num = numeric_cols[0]
        dashboard_items.append({
            "title": f"Distribution of {main_num}",
            "figure": histogram(df, main_num),
            "description": f"The histogram shows the frequency distribution and spread of values for {main_num}."
        })

    # 3. Categorical Breakdown (Pie/Bar)
    if len(cat_cols) > 0:
        main_cat = cat_cols[0]
        # Use Pie for few categories, Bar for many
        if df[main_cat].nunique() <= 10:
            dashboard_items.append({
                "title": f"Proportion of {main_cat}",
                "figure": pie_chart(df, main_cat),
                "description": f"This pie chart shows the relative proportions of different categories in {main_cat}."
            })
        else:
            dashboard_items.append({
                "title": f"Top 10 Categories in {main_cat}",
                "figure": bar_chart(df, main_cat),
                "description": f"This bar chart highlights the most frequent categories within {main_cat}."
            })

    # 4. Outlier Analysis (Boxplot)
    if len(numeric_cols) > 0:
        # Pick a different numeric col if possible
        box_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        dashboard_items.append({
            "title": f"Outlier Check: {box_col}",
            "figure": boxplot(df, box_col),
            "description": f"This boxplot helps identify the median, quartiles, and potential outliers in the {box_col} data."
        })

    # 5. Relationship (Scatter)
    if len(numeric_cols) >= 2:
        dashboard_items.append({
            "title": f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}",
            "figure": scatter_plot(df, numeric_cols[0], numeric_cols[1]),
            "description": f"Visualizing the relationship between {numeric_cols[0]} and {numeric_cols[1]} to detect trends and patterns."
        })

    # 6. Trend over time (if datetime available)
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    if not date_cols:
        # Check if columns look like dates
        for col in df.select_dtypes(include=['object']).columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    temp_df = pd.to_datetime(df[col], errors='coerce')
                    if temp_df.notnull().sum() > len(df) * 0.5: # If half are dates
                        df[col] = temp_df
                        date_cols.append(col)
                        break
                except:
                    continue

    if date_cols and len(numeric_cols) > 0:
        # Simple trend chart (aggregate by date)
        trend_col = numeric_cols[0]
        date_col = date_cols[0]
        trend_df = df.groupby(date_col)[trend_col].mean().reset_index()
        dashboard_items.append({
            "title": f"Trend: Mean {trend_col} over Time",
            "figure": line_chart(trend_df, trend_col),
            "description": f"Observing how the average {trend_col} has evolved over the timeline recorded in {date_col}."
        })

    return dashboard_items