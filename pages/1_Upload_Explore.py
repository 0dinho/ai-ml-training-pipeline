import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.utils.data_utils import (
    load_dataset,
    save_uploaded_file,
    detect_column_types,
    get_dataset_summary,
    compute_missing_values,
    compute_correlation_matrix,
    compute_class_balance,
    get_numerical_stats,
    detect_task_type,
)

TASK_OPTIONS = [
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "dimensionality_reduction",
    "anomaly_detection",
]
TASK_LABELS = {
    "binary_classification": "Binary Classification",
    "multiclass_classification": "Multiclass Classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "dimensionality_reduction": "Dimensionality Reduction",
    "anomaly_detection": "Anomaly Detection",
}
UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}

st.set_page_config(page_title="Upload & Explore", page_icon="📊", layout="wide")

# ── Plotly theme defaults ────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORWAY = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset Status")
    if "df" in st.session_state:
        st.write(f"**File:** {st.session_state.get('uploaded_filename', '—')}")
        shape = st.session_state["df"].shape
        st.write(f"**Shape:** {shape[0]} rows × {shape[1]} cols")
        if st.session_state.get("target_column"):
            st.write(f"**Target:** {st.session_state['target_column']}")
        if st.session_state.get("task_type"):
            st.write(f"**Task:** {st.session_state['task_type']}")
    else:
        st.info("No dataset loaded yet.")

    if st.button("Clear Dataset", type="secondary"):
        for key in [
            "df", "column_types", "dataset_summary", "target_column",
            "task_type", "uploaded_filename", "raw_file_path",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Page Title ───────────────────────────────────────────────────────────────
st.title("📊 Upload & Explore")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: File Upload
# ═══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"],
    help="Supported formats: CSV, Excel (.xlsx/.xls), JSON",
)

if uploaded_file is not None and (
    "uploaded_filename" not in st.session_state
    or st.session_state["uploaded_filename"] != uploaded_file.name
):
    try:
        df = load_dataset(uploaded_file, uploaded_file.name)
        uploaded_file.seek(0)
        raw_path = save_uploaded_file(uploaded_file, uploaded_file.name)
        col_types = detect_column_types(df)
        summary = get_dataset_summary(df, col_types)

        st.session_state["df"] = df
        st.session_state["column_types"] = col_types
        st.session_state["dataset_summary"] = summary
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["raw_file_path"] = raw_path
        # Reset target/task on new upload
        st.session_state.pop("target_column", None)
        st.session_state.pop("task_type", None)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

if "df" not in st.session_state:
    st.info("Upload a dataset to get started.")
    st.stop()

# ── Local references ─────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state["df"]
column_types: dict = st.session_state["column_types"]
summary: dict = st.session_state["dataset_summary"]

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{summary['rows']:,}")
col2.metric("Columns", summary["columns"])
col3.metric("Missing Values", f"{summary['total_missing']:,} ({summary['missing_percentage']}%)")
col4.metric("Duplicates", f"{summary['duplicate_rows']:,}")

# Column types table
col_info = pd.DataFrame({
    "Column": df.columns,
    "Detected Type": [column_types.get(c, "unknown") for c in df.columns],
    "Non-Null Count": [int(df[c].notna().sum()) for c in df.columns],
    "Unique Values": [int(df[c].nunique()) for c in df.columns],
})
st.dataframe(col_info, width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Data Preview with Pagination
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Data Preview")

PAGE_SIZE = 25
total_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, len(df))
st.caption(f"Showing rows {start + 1}–{end} of {len(df)}")
st.dataframe(df.iloc[start:end], width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: EDA Report
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Exploratory Data Analysis")

tab_dist, tab_missing, tab_corr, tab_target = st.tabs(
    ["Distributions", "Missing Values", "Correlations", "Target Analysis"]
)

numerical_cols = [c for c, t in column_types.items() if t == "numerical"]
categorical_cols = [c for c, t in column_types.items() if t == "categorical"]

# ── Distributions ─────────────────────────────────────────────────────────────
with tab_dist:
    if numerical_cols:
        st.subheader("Numerical Distributions")
        grid_cols = st.columns(2)
        for i, col_name in enumerate(numerical_cols):
            with grid_cols[i % 2]:
                fig = px.histogram(
                    df, x=col_name, marginal="box",
                    color_discrete_sequence=[COLORWAY[i % len(COLORWAY)]],
                )
                fig.update_layout(**PLOTLY_LAYOUT, title=col_name)
                st.plotly_chart(fig, width='stretch')

    if categorical_cols:
        st.subheader("Categorical Distributions")
        grid_cols = st.columns(2)
        for i, col_name in enumerate(categorical_cols):
            with grid_cols[i % 2]:
                counts = df[col_name].value_counts().head(20)
                fig = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    labels={"x": col_name, "y": "Count"},
                    color_discrete_sequence=[COLORWAY[i % len(COLORWAY)]],
                )
                fig.update_layout(**PLOTLY_LAYOUT, title=col_name)
                st.plotly_chart(fig, width='stretch')

    if not numerical_cols and not categorical_cols:
        st.info("No numerical or categorical columns to plot.")

# ── Missing Values ────────────────────────────────────────────────────────────
with tab_missing:
    missing_df = compute_missing_values(df)
    missing_with_values = missing_df[missing_df["missing_count"] > 0]

    if missing_with_values.empty:
        st.success("No missing values in the dataset!")
    else:
        fig = px.bar(
            missing_with_values, x="column", y="missing_percentage",
            labels={"column": "Column", "missing_percentage": "Missing %"},
            color_discrete_sequence=[COLORWAY[0]],
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Missing Values by Column (%)")
        st.plotly_chart(fig, width='stretch')

        # Heatmap of missing pattern (sample to 100 rows for performance)
        st.subheader("Missing Pattern Heatmap")
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        missing_mask = sample_df.isnull().astype(int)
        fig = px.imshow(
            missing_mask,
            labels=dict(x="Column", y="Row", color="Missing"),
            color_continuous_scale=["#1a1a2e", COLORWAY[0]],
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Missing Data Pattern (sample)")
        st.plotly_chart(fig, width='stretch')

# ── Correlations ──────────────────────────────────────────────────────────────
with tab_corr:
    corr_matrix = compute_correlation_matrix(df, column_types)
    if corr_matrix is None:
        st.info("Need at least 2 numerical columns to compute correlations.")
    else:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Correlation Matrix",
            height=max(400, 60 * len(corr_matrix)),
        )
        st.plotly_chart(fig, width='stretch')

# ── Target Analysis ───────────────────────────────────────────────────────────
with tab_target:
    task = st.session_state.get("task_type", "")
    target = st.session_state.get("target_column", "")

    if task in UNSUPERVISED_TASKS:
        st.info(
            f"**{TASK_LABELS.get(task, task)}** task selected — no target column required. "
            "The model will be trained on all features."
        )
        if task == "anomaly_detection":
            st.warning(
                "For anomaly detection, configure the contamination rate (expected "
                "fraction of anomalies) in the Training page."
            )
    elif not target:
        st.info("Select a target column below to see target analysis.")
    else:
        if task in ("binary_classification",):
            balance_df = compute_class_balance(df, target)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    balance_df, x="class", y="count",
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_layout(**PLOTLY_LAYOUT, title="Class Distribution")
                st.plotly_chart(fig, width='stretch')
            with c2:
                fig = px.pie(
                    balance_df, names="class", values="count",
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_layout(**PLOTLY_LAYOUT, title="Class Proportions")
                st.plotly_chart(fig, width='stretch')

        elif task == "multiclass_classification":
            balance_df = compute_class_balance(df, target)
            fig = px.bar(
                balance_df, x="class", y="count",
                color_discrete_sequence=COLORWAY,
            )
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Class Distribution ({balance_df.shape[0]} classes)")
            st.plotly_chart(fig, width='stretch')

        elif task == "regression":
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(
                    df, x=target, color_discrete_sequence=[COLORWAY[1]],
                )
                fig.update_layout(**PLOTLY_LAYOUT, title=f"Distribution of {target}")
                st.plotly_chart(fig, width='stretch')
            with c2:
                fig = px.box(
                    df, y=target, color_discrete_sequence=[COLORWAY[2]],
                )
                fig.update_layout(**PLOTLY_LAYOUT, title=f"Box Plot of {target}")
                st.plotly_chart(fig, width='stretch')

        else:
            # Legacy "classification" fallback
            balance_df = compute_class_balance(df, target)
            st.dataframe(balance_df, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Target & Task Configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Target & Task Configuration")

# Task type selector
current_task = st.session_state.get("task_type", "")
# Normalize legacy "classification"
if current_task == "classification":
    current_task = "binary_classification"

task_type = st.selectbox(
    "Task type",
    options=TASK_OPTIONS,
    format_func=lambda t: TASK_LABELS[t],
    index=TASK_OPTIONS.index(current_task) if current_task in TASK_OPTIONS else 0,
    help="Select the ML task. For clustering/dimensionality reduction/anomaly detection, no target column is needed.",
)

# Target column selector — only shown for supervised tasks
target_col = ""
if task_type not in UNSUPERVISED_TASKS:
    # Auto-suggest task type from target column
    suggested_target = st.session_state.get("target_column", df.columns[0])
    if suggested_target not in df.columns:
        suggested_target = df.columns[0]

    target_col = st.selectbox(
        "Select target column",
        options=df.columns.tolist(),
        index=(
            df.columns.tolist().index(suggested_target)
            if suggested_target in df.columns.tolist()
            else 0
        ),
    )

    # Auto-suggest task type when user selects a target
    auto_suggested = detect_task_type(df, target_col)
    if auto_suggested != task_type:
        st.caption(f"💡 Auto-detected: **{TASK_LABELS.get(auto_suggested, auto_suggested)}** based on target column characteristics.")

else:
    st.info(f"**{TASK_LABELS[task_type]}** does not require a target column.")

if st.button("Confirm Configuration", type="primary"):
    st.session_state["target_column"] = target_col
    st.session_state["task_type"] = task_type
    st.rerun()

configured = st.session_state.get("task_type")
if configured:
    _tgt = st.session_state.get("target_column", "")
    if _tgt:
        st.success(
            f"**Target:** `{_tgt}` | **Task:** `{TASK_LABELS.get(configured, configured)}`"
        )
    else:
        st.success(f"**Task:** `{TASK_LABELS.get(configured, configured)}`")
    st.page_link("pages/2_Preprocessing.py", label="Continue to Preprocessing →")
