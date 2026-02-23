import streamlit as st
import pandas as pd
import numpy as np

from src.pipelines.preprocessing import (
    build_preprocessing_pipeline,
    drop_rows_with_missing,
    extract_datetime_features,
    fit_and_transform,
    generate_smart_defaults,
    get_preprocessing_summary,
    save_pipeline,
    save_processed_data,
    save_schema,
    split_data,
    version_with_dvc,
)

st.set_page_config(page_title="Preprocessing", page_icon="🔧", layout="wide")

# ── Plotly theme (same as page 1) ────────────────────────────────────────────
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

# ── Guard clause ─────────────────────────────────────────────────────────────
st.title("🔧 Preprocessing")

if "df" not in st.session_state or "target_column" not in st.session_state:
    st.warning(
        "Please upload a dataset and select a target column on the "
        "**Upload & Explore** page first."
    )
    st.stop()

df: pd.DataFrame = st.session_state["df"]
column_types: dict = st.session_state["column_types"]
target_column: str = st.session_state["target_column"]
task_type: str = st.session_state.get("task_type", "classification")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Quick Start — Auto Preprocess
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Quick Start")

if st.button("⚡ Auto Preprocess", type="primary"):
    st.session_state["column_config"] = generate_smart_defaults(
        column_types, target_column,
    )
    st.rerun()

if "column_config" in st.session_state:
    st.success("Column configuration loaded. Review or adjust below.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Per-Column Configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Per-Column Configuration")

feature_columns = [c for c in df.columns if c != target_column]

if "column_config" not in st.session_state:
    st.info("Click **Auto Preprocess** above or configure each column manually.")
    st.session_state["column_config"] = {}

config: dict = st.session_state["column_config"]

for col in feature_columns:
    ctype = column_types.get(col, "text")
    existing = config.get(col, {})

    with st.expander(f"**{col}** — `{ctype}`", expanded=False):
        if ctype == "numerical":
            imp_options = ["mean", "median", "mode", "constant", "drop"]
            imp_default = existing.get("imputation", "median")
            imp_idx = imp_options.index(imp_default) if imp_default in imp_options else 1
            imputation = st.selectbox(
                "Imputation", imp_options, index=imp_idx, key=f"imp_{col}",
            )

            scl_options = ["standard", "minmax", "robust", "none"]
            scl_default = existing.get("scaling", "standard")
            scl_idx = scl_options.index(scl_default) if scl_default in scl_options else 0
            scaling = st.selectbox(
                "Scaling", scl_options, index=scl_idx, key=f"scl_{col}",
            )

            config[col] = {
                "type": ctype,
                "imputation": imputation,
                "scaling": scaling,
            }

        elif ctype == "categorical":
            imp_options = ["mode", "constant", "drop"]
            imp_default = existing.get("imputation", "mode")
            imp_idx = imp_options.index(imp_default) if imp_default in imp_options else 0
            imputation = st.selectbox(
                "Imputation", imp_options, index=imp_idx, key=f"imp_{col}",
            )

            enc_options = ["onehot", "label", "ordinal", "target"]
            enc_default = existing.get("encoding", "onehot")
            enc_idx = enc_options.index(enc_default) if enc_default in enc_options else 0
            encoding = st.selectbox(
                "Encoding", enc_options, index=enc_idx, key=f"enc_{col}",
            )

            config[col] = {
                "type": ctype,
                "imputation": imputation,
                "encoding": encoding,
            }

        elif ctype == "datetime":
            act_options = ["drop", "extract"]
            act_default = existing.get("action", "drop")
            act_idx = act_options.index(act_default) if act_default in act_options else 0
            action = st.selectbox(
                "Action", act_options, index=act_idx, key=f"act_{col}",
            )

            config[col] = {"type": ctype, "action": action}

        else:  # text
            st.write("Text columns can only be dropped.")
            config[col] = {"type": ctype, "action": "drop"}

st.session_state["column_config"] = config

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Data Split Configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Data Split Configuration")

col_s1, col_s2 = st.columns(2)
with col_s1:
    test_size = st.slider(
        "Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05,
        key="test_size_slider",
    )
with col_s2:
    val_size = st.slider(
        "Validation size", min_value=0.05, max_value=0.5, value=0.1, step=0.05,
        key="val_size_slider",
    )

train_size = round(1.0 - test_size - val_size, 2)

if train_size <= 0:
    st.error("Train size must be positive. Reduce test and/or validation sizes.")
    st.stop()

# Visual proportion bar
col_p1, col_p2, col_p3 = st.columns([train_size, val_size, test_size])
n_rows = len(df)
with col_p1:
    st.metric("Train", f"{train_size:.0%}", f"{int(n_rows * train_size)} rows")
with col_p2:
    st.metric("Validation", f"{val_size:.0%}", f"{int(n_rows * val_size)} rows")
with col_p3:
    st.metric("Test", f"{test_size:.0%}", f"{int(n_rows * test_size)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Run Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Run Preprocessing Pipeline")

if not config:
    st.info("Configure columns above before running the pipeline.")

if st.button("🚀 Run Preprocessing Pipeline", type="primary", disabled=not config):
    with st.spinner("Running preprocessing pipeline..."):
        try:
            working_df = df.copy()

            # --- Step 1: Datetime feature extraction ---
            datetime_extract_cols = [
                col for col, cfg in config.items()
                if cfg["type"] == "datetime" and cfg.get("action") == "extract"
            ]
            if datetime_extract_cols:
                working_df = extract_datetime_features(working_df, datetime_extract_cols)
                # Update column types for new derived columns
                for col in datetime_extract_cols:
                    for suffix in ["_year", "_month", "_day", "_dayofweek"]:
                        new_col = f"{col}{suffix}"
                        column_types[new_col] = "numerical"
                        config[new_col] = {
                            "type": "numerical",
                            "imputation": "median",
                            "scaling": "standard",
                        }

            # --- Step 2: Identify columns with imputation=drop and remove NaN rows ---
            drop_imp_cols = [
                col for col, cfg in config.items()
                if cfg.get("imputation") == "drop" and col in working_df.columns
            ]
            if drop_imp_cols:
                working_df = drop_rows_with_missing(working_df, drop_imp_cols)

            # --- Step 3: Split data ---
            splits = split_data(
                working_df,
                target=target_column,
                test_size=test_size,
                val_size=val_size,
                random_state=42,
                task_type=task_type,
            )

            # --- Step 4: Build pipeline ---
            # Filter config to only include columns present in X_train
            active_config = {
                col: cfg for col, cfg in config.items()
                if col in splits["X_train"].columns
            }
            pipeline, kept_cols, dropped_cols = build_preprocessing_pipeline(
                active_config, column_types,
            )

            # --- Step 5: Fit & transform ---
            X_train_t, X_val_t, X_test_t, feature_names = fit_and_transform(
                pipeline,
                splits["X_train"],
                splits["X_val"],
                splits["X_test"],
                y_train=splits["y_train"],
            )

            # --- Step 6: Save pipeline ---
            pipeline_path = save_pipeline(pipeline)

            # --- Step 6b: Save schema for the inference API ---
            save_schema(
                feature_columns=[c for c in df.columns if c != target_column],
                column_types=column_types,
                column_config=active_config,
                task_type=task_type,
                target_column=target_column,
                feature_names=feature_names,
            )

            # --- Step 7: Save processed data ---
            data_paths = save_processed_data(
                X_train_t, X_val_t, X_test_t,
                splits["y_train"], splits["y_val"], splits["y_test"],
                feature_names,
            )

            # --- Step 8: DVC versioning (best-effort) ---
            all_paths = [pipeline_path] + list(data_paths.values())
            dvc_ok = version_with_dvc(all_paths)

            # --- Step 9: Store in session state ---
            st.session_state["X_train"] = X_train_t
            st.session_state["X_val"] = X_val_t
            st.session_state["X_test"] = X_test_t
            st.session_state["y_train"] = splits["y_train"]
            st.session_state["y_val"] = splits["y_val"]
            st.session_state["y_test"] = splits["y_test"]
            st.session_state["preprocessing_pipeline"] = pipeline
            st.session_state["feature_names"] = feature_names
            st.session_state["dvc_status"] = dvc_ok

            split_sizes = {
                "train": len(splits["X_train"]),
                "val": len(splits["X_val"]),
                "test": len(splits["X_test"]),
            }
            st.session_state["preprocessing_summary"] = get_preprocessing_summary(
                active_config, dropped_cols, kept_cols, feature_names, split_sizes,
            )

            st.success("Preprocessing pipeline completed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Preprocessing failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Before / After Preview (appears after pipeline run)
# ═══════════════════════════════════════════════════════════════════════════════
if "preprocessing_summary" in st.session_state:
    st.header("Results")

    summary = st.session_state["preprocessing_summary"]

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features Before", len(config))
    m2.metric("Features After", summary["features_after"])
    m3.metric("Columns Dropped", summary["columns_dropped"])
    m4.metric(
        "Rows Kept",
        f"{summary['split_sizes']['train'] + summary['split_sizes']['val'] + summary['split_sizes']['test']:,}",
    )

    # Before / After tabs
    tab_before, tab_after = st.tabs(["Raw Data", "Processed Data"])

    with tab_before:
        st.dataframe(df.head(20), use_container_width=True)

    with tab_after:
        X_train_t = st.session_state["X_train"]
        feature_names = st.session_state["feature_names"]
        processed_preview = pd.DataFrame(
            X_train_t[:20], columns=feature_names,
        )
        st.dataframe(processed_preview, use_container_width=True)

    # Split sizes and DVC status
    st.subheader("Split Details")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Train", f"{summary['split_sizes']['train']:,}")
    sc2.metric("Validation", f"{summary['split_sizes']['val']:,}")
    sc3.metric("Test", f"{summary['split_sizes']['test']:,}")

    dvc_ok = st.session_state.get("dvc_status", False)
    sc4.metric("DVC Versioned", "Yes" if dvc_ok else "No")

    if summary["dropped_column_names"]:
        st.caption(f"Dropped columns: {', '.join(summary['dropped_column_names'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Navigation
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/1_Upload_Explore.py", label="← Back to Upload & Explore")
with nav2:
    if "preprocessing_summary" in st.session_state:
        st.page_link("pages/3_Training.py", label="Continue to Training →")
