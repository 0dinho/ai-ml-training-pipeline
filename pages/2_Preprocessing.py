import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

st.set_page_config(page_title="Cleaning & Preprocessing", page_icon="🔧", layout="wide")

UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}

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
st.title("🔧 Cleaning & Preprocessing")

if "df" not in st.session_state:
    st.warning(
        "Please upload a dataset on the **Upload & Explore** page first."
    )
    st.stop()

if "task_type" not in st.session_state:
    st.warning("Please select a task type on the **Upload & Explore** page first.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
column_types: dict = st.session_state["column_types"]
task_type: str = st.session_state.get("task_type", "binary_classification")

# Normalize legacy task type
if task_type == "classification":
    task_type = "binary_classification"
    st.session_state["task_type"] = task_type

target_column: str = st.session_state.get("target_column", "")
is_unsupervised = task_type in UNSUPERVISED_TASKS

if not is_unsupervised and not target_column:
    st.warning(
        "Please select a target column on the **Upload & Explore** page first."
    )
    st.stop()

if is_unsupervised:
    st.info(
        f"**{task_type.replace('_', ' ').title()}** task — using all features "
        "(no target column, no train/val/test split)."
    )

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
# Section 2: Data Quality Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Data Quality Diagnostics")

with st.expander("📊 Data Quality Diagnostics", expanded=False):
    st.caption(
        "Automated checks on the current dataset: missing values, duplicates, "
        "near-constant columns, and outliers. Use these insights to guide your "
        "cleaning decisions below."
    )

    # ── a) Missing Values Bar Chart ───────────────────────────────────────────
    st.subheader("Missing Values")
    missing_ratio = df.isnull().mean()
    missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

    if missing_ratio.empty:
        st.success("No missing values found in any column.")
    else:
        bar_colors = []
        for ratio in missing_ratio.values:
            if ratio > 0.20:
                bar_colors.append("#FF6B6B")
            elif ratio > 0.05:
                bar_colors.append("#FFEAA7")
            else:
                bar_colors.append("#4ECDC4")

        fig_missing = go.Figure(go.Bar(
            x=missing_ratio.values * 100,
            y=missing_ratio.index.tolist(),
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in missing_ratio.values * 100],
            textposition="outside",
        ))
        fig_missing.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Missing %",
            yaxis_title="Column",
            height=max(250, len(missing_ratio) * 28),
            margin=dict(l=10, r=40, t=20, b=30),
        )
        st.plotly_chart(fig_missing, width='stretch')
        high_missing_count = int((missing_ratio > 0.20).sum())
        st.metric("Columns with >20% missing", high_missing_count)

    # ── b) Duplicate Rows ─────────────────────────────────────────────────────
    st.subheader("Duplicate Rows")
    n_dupes = int(df.duplicated().sum())
    st.metric("Duplicate rows", n_dupes)
    if n_dupes > 0:
        if st.button("Drop Duplicates", key="drop_dupes_btn"):
            st.session_state["df"] = st.session_state["df"].drop_duplicates().reset_index(drop=True)
            st.rerun()
    else:
        st.success("No duplicate rows detected.")

    # ── c) Constant / Near-Constant Columns ───────────────────────────────────
    st.subheader("Constant / Near-Constant Columns")
    st.caption("Numerical columns with <3 unique values or top value >95%. Categorical with 1 unique value.")
    near_constant_cols = []
    for col in df.columns:
        ctype = column_types.get(col, "text")
        if ctype == "numerical":
            n_unique = df[col].nunique()
            if n_unique < 3:
                near_constant_cols.append((col, f"only {n_unique} unique value(s)"))
            elif df[col].value_counts(normalize=True).iloc[0] > 0.95:
                freq = df[col].value_counts(normalize=True).iloc[0]
                near_constant_cols.append((col, f"top value covers {freq:.1%} of rows"))
        elif ctype == "categorical" and df[col].nunique() == 1:
            near_constant_cols.append((col, "only 1 unique value"))
    if near_constant_cols:
        for col_name, reason in near_constant_cols:
            st.warning(f"**{col_name}**: {reason}")
    else:
        st.success("No constant or near-constant columns detected.")

    # ── d) Outlier Detection Summary (IQR) ────────────────────────────────────
    st.subheader("Outlier Detection (IQR Method)")
    st.caption("Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. Only columns with outliers shown.")
    num_cols_diag = [
        col for col in df.columns
        if column_types.get(col, "text") == "numerical"
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    outlier_rows = []
    for col in num_cols_diag:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if n_outliers > 0:
            outlier_rows.append({
                "Column": col,
                "Outlier Count": n_outliers,
                "Outlier %": f"{n_outliers / len(series) * 100:.2f}%",
            })
    if outlier_rows:
        st.dataframe(pd.DataFrame(outlier_rows), width='stretch', hide_index=True)
    else:
        st.success("No outliers detected in any numerical column.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Advanced Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Advanced Cleaning")

with st.expander("Low-Variance Filter", expanded=False):
    st.caption("Remove numerical features whose variance falls below a threshold.")
    var_threshold = st.slider("Variance threshold", 0.0, 1.0, 0.01, 0.01, key="var_threshold_slider")
    num_cols_var = [
        col for col in df.columns
        if column_types.get(col, "text") == "numerical"
        and pd.api.types.is_numeric_dtype(df[col])
        and col != target_column
    ]
    low_var_preview = [col for col in num_cols_var if df[col].dropna().var() < var_threshold]
    if low_var_preview:
        st.write(f"**Would drop** ({len(low_var_preview)}): {', '.join(low_var_preview)}")
    else:
        st.info("No columns fall below the current threshold.")
    if st.button("Apply Low-Variance Filter", key="apply_var_filter_btn"):
        try:
            from sklearn.feature_selection import VarianceThreshold
            if num_cols_var:
                selector = VarianceThreshold(threshold=var_threshold)
                tmp = df[num_cols_var].fillna(df[num_cols_var].median())
                selector.fit(tmp)
                dropped = [c for c, keep in zip(num_cols_var, selector.get_support()) if not keep]
                if dropped:
                    st.session_state["df"] = st.session_state["df"].drop(columns=dropped)
                    for c in dropped:
                        st.session_state["column_types"].pop(c, None)
                        st.session_state.get("column_config", {}).pop(c, None)
                    st.success(f"Dropped {len(dropped)} column(s): {', '.join(dropped)}")
                    st.rerun()
                else:
                    st.info("No columns to drop with the current threshold.")
        except Exception as e:
            st.error(f"Low-variance filter failed: {e}")

with st.expander("High-Correlation Filter", expanded=False):
    st.caption("Remove one column from each pair of highly correlated numerical features.")
    corr_threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.95, 0.01, key="corr_threshold_slider")
    num_cols_corr = [
        col for col in df.columns
        if column_types.get(col, "text") == "numerical"
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(num_cols_corr) >= 2:
        heatmap_cols = num_cols_corr[:20]
        corr_matrix = df[heatmap_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu", zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}", showscale=True,
        ))
        fig_corr.update_layout(
            **PLOTLY_LAYOUT,
            height=max(350, len(heatmap_cols) * 30),
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_corr, width='stretch')
        feat_corr_cols = [c for c in num_cols_corr if c != target_column]
        if len(feat_corr_cols) >= 2:
            corr_abs = df[feat_corr_cols].corr().abs()
            upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
            high_corr_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
            if high_corr_drop:
                st.write(f"**Would drop** ({len(high_corr_drop)}): {', '.join(high_corr_drop)}")
            else:
                st.info("No column pairs exceed the current threshold.")
        if st.button("Apply High-Correlation Filter", key="apply_corr_filter_btn"):
            try:
                fc = [c for c in num_cols_corr if c != target_column]
                if len(fc) >= 2:
                    ca = df[fc].corr().abs()
                    ut = ca.where(np.triu(np.ones(ca.shape), k=1).astype(bool))
                    to_drop = [c for c in ut.columns if any(ut[c] > corr_threshold)]
                    if to_drop:
                        st.session_state["df"] = st.session_state["df"].drop(columns=to_drop)
                        for c in to_drop:
                            st.session_state["column_types"].pop(c, None)
                            st.session_state.get("column_config", {}).pop(c, None)
                        st.success(f"Dropped {len(to_drop)} column(s): {', '.join(to_drop)}")
                        st.rerun()
                    else:
                        st.info("No columns to drop with the current threshold.")
            except Exception as e:
                st.error(f"High-correlation filter failed: {e}")
    else:
        st.info("Need at least 2 numerical columns to compute correlations.")

with st.expander("Value Range Validation", expanded=False):
    st.caption(
        "Set acceptable min/max bounds for numerical columns. "
        "Rows with values outside the bounds will be clipped to the boundary value."
    )
    num_cols_range = [
        col for col in df.columns
        if column_types.get(col, "text") == "numerical"
        and pd.api.types.is_numeric_dtype(df[col])
        and col != target_column
    ]
    if not num_cols_range:
        st.info("No numerical feature columns found.")
    else:
        range_configs: dict[str, tuple] = {}
        for col in num_cols_range:
            col_min = float(df[col].min()) if not df[col].isna().all() else 0.0
            col_max = float(df[col].max()) if not df[col].isna().all() else 1.0
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.write(f"`{col}` (data range: {col_min:.3g} – {col_max:.3g})")
            with col2:
                user_min = st.number_input("Min", value=col_min, key=f"rng_min_{col}")
            with col3:
                user_max = st.number_input("Max", value=col_max, key=f"rng_max_{col}")
            with col4:
                if st.checkbox("Enable", key=f"rng_en_{col}"):
                    range_configs[col] = (user_min, user_max)

        if range_configs:
            n_violations = sum(
                int(((df[col] < lo) | (df[col] > hi)).sum())
                for col, (lo, hi) in range_configs.items()
            )
            st.caption(f"{n_violations} total out-of-range values across {len(range_configs)} column(s).")

        if st.button("Apply Value Range Clipping", key="apply_range_btn"):
            if not range_configs:
                st.info("Enable at least one column range above.")
            else:
                working = st.session_state["df"].copy()
                for col, (lo, hi) in range_configs.items():
                    working[col] = working[col].clip(lower=lo, upper=hi)
                st.session_state["df"] = working
                st.success(f"Clipped {len(range_configs)} column(s) to configured bounds.")
                st.rerun()

with st.expander("Regex Pattern Validation", expanded=False):
    st.caption(
        "Flag rows where a categorical / text column does not match a regex pattern. "
        "Non-matching rows are replaced with NaN (to be handled by imputation)."
    )
    cat_cols_regex = [
        col for col in df.columns
        if column_types.get(col, "text") in ("categorical", "text")
        and col != target_column
    ]
    if not cat_cols_regex:
        st.info("No categorical or text columns found.")
    else:
        regex_configs: dict[str, str] = {}
        for col in cat_cols_regex:
            r1, r2, r3 = st.columns([3, 4, 1])
            with r1:
                st.write(f"`{col}`")
            with r2:
                pattern = st.text_input("Regex pattern", value="", key=f"regex_{col}",
                                        placeholder=r"e.g. ^\d{4}$ or ^[A-Z]")
            with r3:
                if pattern and st.checkbox("Enable", key=f"regex_en_{col}"):
                    regex_configs[col] = pattern

        if regex_configs:
            for col, pat in regex_configs.items():
                try:
                    non_match = (~df[col].astype(str).str.match(pat, na=False)).sum()
                    st.caption(f"`{col}`: {non_match} non-matching rows for pattern `{pat}`")
                except Exception as e:
                    st.warning(f"`{col}`: invalid regex — {e}")

        if st.button("Apply Regex Validation (replace non-matches with NaN)", key="apply_regex_btn"):
            if not regex_configs:
                st.info("Enable at least one column with a pattern above.")
            else:
                working = st.session_state["df"].copy()
                total_replaced = 0
                for col, pat in regex_configs.items():
                    try:
                        mask = ~working[col].astype(str).str.match(pat, na=False)
                        total_replaced += int(mask.sum())
                        working.loc[mask, col] = np.nan
                    except Exception as e:
                        st.warning(f"Regex failed for `{col}`: {e}")
                st.session_state["df"] = working
                st.success(f"Replaced {total_replaced} non-matching values with NaN.")
                st.rerun()

with st.expander("KNN Imputation", expanded=False):
    st.caption(
        "Select **knn** as the imputation strategy for any numerical column below. "
        "KNNImputer (sklearn) will be applied before the pipeline is built."
    )
    st.info("Set Imputation → **knn** in the Per-Column Configuration below to enable it per column.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Per-Column Configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Per-Column Configuration")

feature_columns = [c for c in df.columns if c != target_column] if target_column else list(df.columns)

if "column_config" not in st.session_state:
    st.info("Click **Auto Preprocess** above or configure each column manually.")
    st.session_state["column_config"] = {}

config: dict = st.session_state["column_config"]

for col in feature_columns:
    ctype = column_types.get(col, "text")
    existing = config.get(col, {})

    with st.expander(f"**{col}** — `{ctype}`", expanded=False):
        if ctype == "numerical":
            imp_options = ["mean", "median", "mode", "constant", "knn", "drop"]
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

if is_unsupervised:
    st.info(
        "Unsupervised tasks use the **full dataset** for fitting — "
        "no train/validation/test split is applied."
    )
    test_size = 0.2  # stored but not used for unsupervised
    val_size = 0.1
else:
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

            # --- Step 0: KNN Imputation (applied before pipeline build) ---
            knn_cols = [
                col for col, cfg in config.items()
                if cfg.get("imputation") == "knn"
                and col in working_df.columns
                and pd.api.types.is_numeric_dtype(working_df[col])
            ]
            if knn_cols:
                from sklearn.impute import KNNImputer
                knn_imputer = KNNImputer(n_neighbors=5)
                working_df[knn_cols] = knn_imputer.fit_transform(working_df[knn_cols])
                # Remap knn → median so the pipeline builder handles remaining imputation
                for col in knn_cols:
                    config[col] = dict(config[col], imputation="median")
                st.session_state["column_config"] = config

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
                target=target_column if not is_unsupervised else None,
                test_size=test_size,
                val_size=val_size,
                random_state=42,
                task_type=task_type,
            )

            # --- Step 4: Build pipeline ---
            # Filter config to only include columns present in X_train
            x_train_cols = splits["X_train"].columns
            active_config = {
                col: cfg for col, cfg in config.items()
                if col in x_train_cols
            }
            pipeline, kept_cols, dropped_cols = build_preprocessing_pipeline(
                active_config, column_types,
            )

            # --- Step 5: Fit & transform ---
            X_train_t, X_val_t, X_test_t, feature_names = fit_and_transform(
                pipeline,
                splits["X_train"],
                splits["X_val"],  # None for unsupervised
                splits["X_test"],  # None for unsupervised
                y_train=splits["y_train"],
            )

            # --- Step 6: Save pipeline ---
            pipeline_path = save_pipeline(pipeline)

            # --- Step 6b: Save schema for the inference API ---
            save_schema(
                feature_columns=[c for c in df.columns if c != target_column] if target_column else list(df.columns),
                column_types=column_types,
                column_config=active_config,
                task_type=task_type,
                target_column=target_column or "",
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
                "val": len(splits["X_val"]) if splits["X_val"] is not None else 0,
                "test": len(splits["X_test"]) if splits["X_test"] is not None else 0,
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
        st.dataframe(df.head(20), width='stretch')

    with tab_after:
        X_train_t = st.session_state["X_train"]
        feature_names = st.session_state["feature_names"]
        processed_preview = pd.DataFrame(
            X_train_t[:20], columns=feature_names,
        )
        st.dataframe(processed_preview, width='stretch')

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
        st.page_link("pages/3_Feature_Engineering.py", label="Continue to Feature Engineering →")
