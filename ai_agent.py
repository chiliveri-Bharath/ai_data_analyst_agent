import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def get_client():
    """
    Returns an initialized Groq client. Re-checks the environment variable 
    to ensure changes in .env are picked up.
    """
    # Force reload environment variables to catch changes in .env
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return None
    return Groq(api_key=api_key)

# 🧠 UNIFIED AI AGENT
def process_query(query, df, chat_history=[]):
    """
    Analyzes the query, identifies tasks, and provides a conversational response.
    3. For analysis tasks likes 'topvalues' or 'bottomvalues', identify the number of results requested (e.g., 'top 10' -> n=10). Default to n=5 if not specified.
    4. For 'topvalues' or 'bottomvalues', always use the respective task name. If the user asks for 'top and bottom' or 'top 5 and bottom 5', use task='topvalues' and mode='both'.
    5. Always return a JSON object with at least 'task' and 'message'. Include 'col1', 'col2', 'n', 'mode' as needed.
    """
    client = get_client()
    if not client:
        return {"task": "chat", "message": "⚠️ Groq API Key is missing. Please check your .env file."}

    columns = df.columns.tolist()
    
    # System Prompt for Decision Making
    system_prompt = f"""
    You are an expert, friendly AI Data Analyst. Your goal is to help users clean, analyze, and visualize their data with clarity and insight.
    
    Data Columns: {columns}
    
    Available Tasks (Keywords):
    - Visualization: histogram, scatter, correlation, boxplot, linechart, pie, bar, dashboard
    - Cleaning/Transform: fillna, dropnarrows, duplicates, outliers, removecolumn, renamecolumn, sortdata, 
      standardizenames, stripwhitespace, dropnacolumns, splitcolumn, zscore, todatetime, encode, scale, normalize, knnimpute, simpleimpute, autoclean
    - Analysis: summary, groupby, average, sum, min, max, topvalues, bottomvalues, valuecounts
    - Feature Engineering: calculate (+, -, *, /), onehot, dateextract
    - General: chat (for general questions or unknown requests)
 
    Instructions for a Great User Experience:
    1. Mapping Tasks: Identify the task and columns (col1, col2). Use col1="all" if appropriate.
    2. Top/Bottom Logic:
       - Extract 'n' (e.g., "top 10" -> n=10, default=5).
       - Extract 'mode': "top", "bottom", or "both" (e.g., "top and bottom 5" -> mode="both").
    3. Dashboard Task:
       - Use task="dashboard" when the user asks for a dashboard, an overview, a summary of trends, or "show me everything."
    4. Conversational Style: 
       - Be friendly and proactive. 
       - ALWAYS explain WHAT you found and WHY it matters in the 'message' field.
       - DON'T just say "Here is the result." Instead, say something like "I've calculated the average salary for you. It looks like the typical compensation is around X, which might be useful for your budget planning."
    4. Proactive Suggestions: Occasionally suggest a logical next step (e.g., "Would you like to see a chart of this distribution next?").
    5. No Placeholder Text: Do not use technical jargon unless necessary.
    6. Returns: Return ONLY a JSON object.

    Example JSON:
    {{
        "task": "topvalues",
        "col1": "Sales",
        "n": 3,
        "mode": "both",
        "message": "I've pulled the top and bottom 3 sales performers for you. Looking at the extremes can help us identify both our biggest successes and the areas that might need some extra attention. Would you like to see a bar chart of these individuals?"
    }}
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add limited history for context
    for msg in chat_history[-5:]:
        # Use content only if it's text
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"] if isinstance(msg.get("content"), str) else "Data visualization/table displayed."
        messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": query})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        return {"task": "chat", "message": f"I had a bit of trouble processing that. Error: {str(e)}"}

# 🚀 AUTO INSIGHTS
def generate_auto_insights(df):
    client = get_client()
    if not client: return "API Key missing."
    summary = df.describe(include='all').to_dict()
    prompt = f"Analyze this dataset and give 2-3 professional insights in markdown: {summary}"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 🧠 EXPLAIN OUTLIERS
def explain_outliers(column, stats):
    client = get_client()
    if not client: return "API Key missing."
    
    prompt = f"""
    As an AI Data Analyst, explain why these outliers in the '{column}' column are significant.
    Statistics:
    - Count: {stats['count']}
    - Range of Outliers: [{stats['min']} to {stats['max']}]
    - Bounds: Lower={stats['lower_bound']:.2f}, Upper={stats['upper_bound']:.2f}
    
    Give a professional, concise 2-3 sentence explanation of what this might mean for the data (e.g., potential errors, extreme but valid cases, or data skew).
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 📊 EXPLAIN CHART
def explain_chart(chart_type, col1, col2=None, df=None):
    client = get_client()
    if not client: return "API Key missing."
    
    # Get basic stats for context
    stats_context = ""
    if df is not None:
        def get_col_stats(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                return f"Mean={df[col].mean():.2f}, Median={df[col].median():.2f}, Std={df[col].std():.2f}"
            else:
                return f"Unique Values={df[col].nunique()}, Most Common='{df[col].mode()[0] if not df[col].empty else 'N/A'}'"

        if col1 in df.columns:
            stats_context += f"\n- {col1} Stats: {get_col_stats(col1)}"
        if col2 and col2 in df.columns:
            stats_context += f"\n- {col2} Stats: {get_col_stats(col2)}"
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                correlation = df[[col1, col2]].corr().iloc[0,1]
                stats_context += f"\n- Correlation between {col1} and {col2}: {correlation:.2f}"

    prompt = f"""
    As an AI Data Analyst, explain the findings in this '{chart_type}' chart involving columns: '{col1}' {f'and {col2}' if col2 else ''}.
    
    Data Context: {stats_context}
    
    Provide a professional, concise 3-4 sentence explanation of the trends, distribution, or relationships shown in this chart.
    Focus on what a business user should take away from this visualization.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"