import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from ai_agent import process_query, generate_auto_insights, explain_outliers, explain_chart
from file_handler import load_file
from visualizer import histogram, scatter_plot, correlation_heatmap, boxplot, line_chart, pie_chart, bar_chart
from dashboard import generate_dashboard
from data_cleaner import (
    fill_missing, drop_duplicates,
    remove_outliers, missing_summary,
    detect_outliers, cap_outliers,
    create_calculated_feature, extract_date_parts, one_hot_encode,
    clean_text, fix_datatypes, auto_clean, dataset_summary,
    remove_column, rename_column, sort_data,
    drop_na_columns, standardize_names, strip_whitespace, split_column,
    convert_to_datetime, encode_labels, min_max_scale, standard_scale,
    knn_impute, simple_impute, drop_na_rows
)
from analysis import groupby_analysis, get_top_bottom, calculate_stat, get_value_counts

# Page Config
st.set_page_config(
    page_title="🤖 AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Vibrant Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #f0f4f8;
    }

    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }

    .stChatMessage, .stDataFrame, .stExpander {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
        margin-bottom: 1rem !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stButton>button:not([kind="tertiary"]):not([kind="secondary"]) {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #0EA5E9 100%);
        color: white !important;
        border: none !important;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .stButton>button[kind="secondary"] {
        padding: 0.1rem 0.3rem !important;
        font-size: 0.8rem !important;
        min-height: 0 !important;
        height: auto !important;
        background: transparent !important;
        border: none !important;
        color: #64748b !important;
    }

    .stButton>button[kind="secondary"]:hover {
        background: rgba(0,0,0,0.05) !important;
        color: #1e293b !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.2);
        opacity: 0.9;
    }

    h1 {
        background: linear-gradient(90deg, #1E293B 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    .stChatMessage {
        padding: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "pending_outliers" not in st.session_state:
        st.session_state.pending_outliers = None
    if "edit_message_idx" not in st.session_state:
        st.session_state.edit_message_idx = None
    if "show_raw_idx" not in st.session_state:
        st.session_state.show_raw_idx = None

def sidebar_summary():
    with st.sidebar:
        st.title("📊 Dataset Control")
        
        st.subheader("🔑 Configuration")
        if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
            new_key = st.text_input("Enter Groq API Key:", type="password", key="api_key_input")
            if new_key:
                os.environ["GROQ_API_KEY"] = new_key
                st.success("API Key applied!")
        else:
            st.success("API Key loaded from .env")

        if st.session_state.df is not None:
            df = st.session_state.df
            st.success("Dataset Loaded!")
            col1, col2 = st.columns(2)
            col1.metric("Rows", df.shape[0])
            col2.metric("Cols", df.shape[1])
            
            if st.sidebar.button("🗑️ Clear Dataset", key="clear_btn"):
                st.session_state.df = None
                st.session_state.chat_history = []
                st.rerun()
            
            if st.sidebar.button("✨ Auto Clean Data", key="autoclean_btn"):
                st.session_state.df = auto_clean(df)
                st.toast("Applied Full Auto Clean")
                st.rerun()

            with st.expander("🛠 Cleaning Tools"):
                if st.button("📏 Standardize Names", key="std_names"):
                    st.session_state.df = standardize_names(df)
                    st.rerun()
                if st.button("✂ Strip Whitespace", key="strip_ws"):
                    st.session_state.df = strip_whitespace(df)
                    st.rerun()

def main():
    initialize_session()
    sidebar_summary()
    
    st.title("🤖 AI Data Analyst Agent")
    st.info("Upload a dataset and ask questions to get visual and AI-driven insights.")

    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV/Excel)", type=["csv", "xlsx"], key="main_file_uploader")

    if uploaded_file is not None and st.session_state.df is None:
        st.session_state.df = load_file(uploaded_file)
        st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Raw Data", "📊 Dashboard"])

        with tab2:
            st.subheader("📄 Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("🚀 Generate Auto Insights", key="insights_btn"):
                with st.spinner("Analyzing..."):
                    insights = generate_auto_insights(df)
                    st.markdown(insights)

        with tab3:
            st.subheader("📊 Trends & Insights Dashboard")
            if st.button("🔄 Regenerate Dashboard", key="regen_dash"):
                with st.spinner("Generating Dashboard..."):
                    st.session_state.dashboard_items = generate_dashboard(df)
                    st.rerun()
            
            if "dashboard_items" in st.session_state and st.session_state.dashboard_items:
                cols = st.columns(2)
                for idx, item in enumerate(st.session_state.dashboard_items):
                    with cols[idx % 2]:
                        st.markdown(f"### {item['title']}")
                        st.plotly_chart(item['figure'], use_container_width=True, key=f"dash_fig_{idx}")
                        with st.expander("💡 Insight"):
                            st.write(item['description'])
            else:
                st.info("No dashboard generated yet. Ask the AI to 'show a dashboard' or click the button above.")

        with tab1:
            # Display history
            for i, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                    # Edit mode handle
                    if st.session_state.edit_message_idx == i:
                        new_content = st.text_area("Edit Message:", value=msg.get("content", ""), key=f"edit_area_{i}")
                        col_save, col_cancel = st.columns([1, 10])
                        if col_save.button("💾 Save", key=f"save_btn_{i}"):
                            st.session_state.chat_history[i]["content"] = new_content
                            st.session_state.edit_message_idx = None
                            st.rerun()
                        if col_cancel.button("❌ Cancel", key=f"cancel_btn_{i}"):
                            st.session_state.edit_message_idx = None
                            st.rerun()
                    else:
                        if st.session_state.show_raw_idx == i:
                            st.code(msg.get("content", ""), language="markdown")
                        else:
                            st.markdown(msg.get("content", ""))
                            
                        # Action Dropdown (restricted to user prompts)
                        if msg["role"] == "user":
                            _, menu_col = st.columns([9.5, 0.5])
                            with menu_col:
                                with st.popover("⋮", help="Message Options"):
                                    if st.button("✏️ Edit", key=f"edit_{i}", use_container_width=True):
                                        st.session_state.edit_message_idx = i
                                        st.session_state.show_raw_idx = None
                                        st.rerun()
                                    if st.button("📋 Copy ", key=f"copy_{i}", use_container_width=True):
                                        if st.session_state.show_raw_idx == i:
                                            st.session_state.show_raw_idx = None
                                        else:
                                            st.session_state.show_raw_idx = i
                                        st.rerun()
                                    if st.button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                                        st.session_state.chat_history.pop(i)
                                        # Check if the next message is the AI's response, and delete it too
                                        if i < len(st.session_state.chat_history) and st.session_state.chat_history[i]["role"] == "assistant":
                                            st.session_state.chat_history.pop(i)
                                        st.session_state.show_raw_idx = None
                                        st.rerun()
                    if "figure" in msg:
                        st.caption(f"📊 {msg.get('chart_name', 'Visualization')}")
                        st.plotly_chart(msg["figure"], use_container_width=True, key=f"fig_{i}")
                        
                        # Add Explain Chart functionality
                        if "explanation" in msg:
                            st.info(msg["explanation"])
                        elif "chart_type" in msg:
                            if st.button("🔍 Explain Chart", key=f"explain_btn_{i}"):
                                with st.spinner("Generating explanation..."):
                                    explanation = explain_chart(msg["chart_type"], msg.get("col1"), msg.get("col2"), st.session_state.df)
                                    msg["explanation"] = explanation # Update the history object directly
                                    st.rerun()
                    
                    if "dashboard_items" in msg:
                        for idx, item in enumerate(msg["dashboard_items"]):
                            st.subheader(f"📊 {item['title']}")
                            st.plotly_chart(item['figure'], use_container_width=True, key=f"fig_{i}_{idx}")
                            st.info(item['description'])

                    if "table" in msg: st.table(msg["table"])
                    if "top_table" in msg:
                        st.subheader(f"🔝 Top {msg.get('n', 5)} in {msg.get('col1', '')}")
                        st.table(msg["top_table"])
                    if "bottom_table" in msg:
                        st.subheader(f"🔽 Bottom {msg.get('n', 5)} in {msg.get('col1', '')}")
                        st.table(msg["bottom_table"])
                    if "missing_table" in msg:
                        with st.expander("📉 Detailed Missing Values"):
                            st.table(msg["missing_table"])
            
            # --- Interactive Outlier Handler ---
            if st.session_state.pending_outliers:
                po = st.session_state.pending_outliers
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"### ⚠️ Outlier Action Required: {po['column']}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Outliers", po["count"])
                    col_b.metric("Min outlier", f"{po['min']:.2f}" if po['min'] is not None else "N/A")
                    col_c.metric("Max outlier", f"{po['max']:.2f}" if po['max'] is not None else "N/A")

                    with st.spinner("Generating AI explanation..."):
                        if "explanation" not in po:
                            po["explanation"] = explain_outliers(po["column"], po)
                        st.info(po["explanation"])

                    st.write("How would you like to handle these?")
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    if btn_col1.button("🗑️ Remove Rows", key="outlier_remove"):
                        st.session_state.df = remove_outliers(st.session_state.df, po["column"], method=po["method"])
                        st.session_state.pending_outliers = None
                        st.toast(f"Removed {po['count']} rows from {po['column']}")
                        st.rerun()

                    if btn_col2.button("🛡️ Cap Values", key="outlier_cap"):
                        st.session_state.df = cap_outliers(st.session_state.df, po["column"], method=po["method"])
                        st.session_state.pending_outliers = None
                        st.toast(f"Capped outliers in {po['column']}")
                        st.rerun()

                    if btn_col3.button("✅ Keep All", key="outlier_keep"):
                        st.session_state.pending_outliers = None
                        st.toast("Keeping all points as-is.")
                        st.rerun()

        # --- chat input (Outside Tabs to stay pinned at bottom) ---
        if prompt := st.chat_input("Ask about your data, e.g., 'What is the sum of Salary?' or 'Show a boxplot of Age'", key="chat_input_box"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Show processing in Tab 1
            with tab1:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing..."):
                        res = process_query(prompt, df, st.session_state.chat_history)
                        task = res.get("task", "chat")
                        col1 = res.get("col1")
                        col2 = res.get("col2")
                        msg_text = res.get("message", "Here is what I found.")
                        
                        task_item = {"role": "assistant", "content": msg_text}
                        needs_rerun = False
                        
                        try:
                            if task in ["boxplot", "linechart", "histogram", "scatter", "scatterplot", "pie", "bar", "correlation"]:
                                task_item["chart_type"] = task
                                task_item["col1"] = col1
                                task_item["col2"] = col2

                            if task == "boxplot": 
                                task_item["figure"] = boxplot(df, col1)
                                task_item["chart_name"] = f"Boxplot of {col1}"
                            elif task == "linechart": 
                                task_item["figure"] = line_chart(df, col1)
                                task_item["chart_name"] = f"Line Chart of {col1}"
                            elif task == "histogram": 
                                task_item["figure"] = histogram(df, col1)
                                task_item["chart_name"] = f"Histogram of {col1}"
                            elif task in ["scatter", "scatterplot"]: 
                                task_item["figure"] = scatter_plot(df, col1, col2)
                                task_item["chart_name"] = f"Scatter Plot: {col1} vs {col2}"
                            elif task == "pie": 
                                task_item["figure"] = pie_chart(df, col1)
                                task_item["chart_name"] = f"Pie Chart of {col1}"
                            elif task == "bar": 
                                task_item["figure"] = bar_chart(df, col1)
                                task_item["chart_name"] = f"Bar Chart of {col1}"
                            elif task == "correlation": 
                                task_item["figure"] = correlation_heatmap(df)
                                task_item["chart_name"] = "Correlation Heatmap"
                            elif task == "summary":
                                d_summary = dataset_summary(df)
                                task_item["table"] = d_summary["Numeric Summary"]
                                msg_text += f"\n\n**Data Quality Info:**\n- 🔁 Duplicate Rows: `{d_summary['Duplicates']}`\n- 📉 Total Missing Values: `{d_summary['Missing Values'].sum()}`"
                                task_item["content"] = msg_text
                                if d_summary['Missing Values'].any():
                                    task_item["missing_table"] = d_summary["Missing Values"]
                            elif task == "groupby": task_item["table"] = groupby_analysis(df, col1)
                            elif task in ["topvalues", "bottomvalues"]:
                                n = res.get("n", 5)
                                mode = res.get("mode", "top" if task == "topvalues" else "bottom")
                                tb_res = get_top_bottom(df, col1, n=n, mode=mode)
                                if tb_res:
                                    task_item["n"] = n
                                    task_item["col1"] = col1
                                    if "top" in tb_res:
                                        task_item["top_table"] = tb_res["top"][[col1]] if isinstance(tb_res["top"], pd.DataFrame) else tb_res["top"]
                                    if "bottom" in tb_res:
                                        task_item["bottom_table"] = tb_res["bottom"][[col1]] if isinstance(tb_res["bottom"], pd.DataFrame) else tb_res["bottom"]
                                    # Append specific observation to the AI's descriptive message
                                    msg_text += f"\n\nShowing the **{mode} {n}** values for **{col1}** below."
                            elif task in ["average", "sum", "min", "max"]:
                                stat_val = calculate_stat(df, col1, task)
                                if stat_val is not None:
                                    msg_text += f"\n\n**Result:** The **{task}** of **{col1}** is `{stat_val:,.2f}`."
                            elif task == "valuecounts":
                                v_counts = get_value_counts(df, col1)
                                if v_counts is not None:
                                    task_item["table"] = v_counts
                            
                            # Mutating tasks
                            elif task == "calculate":
                                op = res.get("operation", "+")
                                new_name = res.get("new_col_name", f"{col1}_{op}_{col2}")
                                st.session_state.df = create_calculated_feature(df, col1, col2, op, new_name)
                                needs_rerun = True
                            elif task == "onehot":
                                st.session_state.df = one_hot_encode(df, col1)
                                needs_rerun = True
                            elif task == "dateextract":
                                st.session_state.df = extract_date_parts(df, col1)
                                needs_rerun = True
                            elif task == "fillna": st.session_state.df = fill_missing(df); needs_rerun = True
                            elif task == "duplicates": st.session_state.df = drop_duplicates(df); needs_rerun = True
                            elif task == "autoclean": st.session_state.df = auto_clean(df); needs_rerun = True
                            elif task == "outliers" or task == "zscore":
                                method = "zscore" if task == "zscore" else "iqr"
                                status = detect_outliers(df, col1, method=method)
                                if status and status["count"] > 0:
                                    status["column"] = col1
                                    status["method"] = method
                                    st.session_state.pending_outliers = status
                                    msg_text = f"🔍 I've detected {status['count']} outliers in **{col1}**. I'm preparing an explanation and handling options for you now."
                                else:
                                    msg_text = f"✅ No outliers detected in **{col1}** using the {method.upper()} method."
                            elif task == "knnimpute": st.session_state.df = knn_impute(df); needs_rerun = True
                            elif task == "simpleimpute": st.session_state.df = simple_impute(df); needs_rerun = True
                            elif task == "removecolumn": st.session_state.df = remove_column(df, col1); needs_rerun = True
                            elif task == "renamecolumn": st.session_state.df = rename_column(df, col1, col2 or "renamed"); needs_rerun = True
                            elif task == "encode": st.session_state.df = encode_labels(df, col1); needs_rerun = True
                            elif task == "todatetime": st.session_state.df = convert_to_datetime(df, col1); needs_rerun = True
                            elif task == "scale": st.session_state.df = standard_scale(df, col1); needs_rerun = True
                            elif task == "normalize": st.session_state.df = min_max_scale(df, col1); needs_rerun = True
                            elif task == "sortdata": st.session_state.df = sort_data(df, col1); needs_rerun = True
                            elif task == "standardizenames": st.session_state.df = standardize_names(df); needs_rerun = True
                            elif task == "stripwhitespace": st.session_state.df = strip_whitespace(df); needs_rerun = True
                            elif task == "dropnacolumns": st.session_state.df = drop_na_columns(df); needs_rerun = True
                            elif task == "dropnarrows": st.session_state.df = drop_na_rows(df, col1); needs_rerun = True
                            elif task == "dashboard":
                                items = generate_dashboard(df)
                                st.session_state.dashboard_items = items
                                task_item["dashboard_items"] = items
                                msg_text += "\n\nI've generated a comprehensive dashboard for you! You can also view it in the **Dashboard** tab for a better layout."
                            
                            task_item["content"] = msg_text
                            if needs_rerun:
                                task_item["table"] = st.session_state.df.head()
                            st.session_state.chat_history.append(task_item)
                            st.rerun() # Refresh to show in history loop
                        except Exception as e:
                            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()