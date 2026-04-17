import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Define a premium color palette
PREMIUM_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
TEMPLATE = "plotly_white"

def apply_premium_style(fig):
    fig.update_layout(
        template=TEMPLATE,
        font_family="Inter, system-ui, sans-serif",
        title_font_size=20,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="closest",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# 📦 Boxplot
def boxplot(df, column):
    fig = px.box(df, y=column, title=f"Distribution of {column}", color_discrete_sequence=[PREMIUM_COLORS[3]])
    return apply_premium_style(fig)

# 📈 Line Chart
def line_chart(df, column):
    fig = px.line(df, y=column, title=f"{column} over Index", color_discrete_sequence=[PREMIUM_COLORS[0]], markers=True)
    return apply_premium_style(fig)

# 📊 Histogram
def histogram(df, column):
    fig = px.histogram(df, x=column, title=f"Frequency Distribution: {column}", color_discrete_sequence=[PREMIUM_COLORS[0]], marginal="box")
    return apply_premium_style(fig)

# 📊 Bar Chart
def bar_chart(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=f"Top Values in {column}", color=column, color_discrete_sequence=PREMIUM_COLORS)
    return apply_premium_style(fig)

# 🥧 Pie Chart
def pie_chart(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.pie(counts, names=column, values="count", title=f"Proportion: {column}", color_discrete_sequence=PREMIUM_COLORS, hole=0.4)
    return apply_premium_style(fig)

# 📈 Scatter Plot
def scatter_plot(df, x, y):
    title = f"Relationship: {x} vs {y}" if y else f"Scatter of {x}"
    fig = px.scatter(df, x=x, y=y, title=title, color_discrete_sequence=[PREMIUM_COLORS[1]], trendline="ols")
    return apply_premium_style(fig)

# 🔥 Correlation Heatmap
def correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
    return apply_premium_style(fig)