"""Phase 6 — Prediction & Inference."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from src.pipelines.preprocessing import extract_datetime_features
from src.pipelines.training import TrainingResult

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
MLFLOW_URI = "http://localhost:5000"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Status")
    if "training_results" in st.session_state:
        s = st.session_state["training_summary"]
        st.success(f"{s['models_trained']} model(s) trained")
        st.write(f"**Best:** {s['best_model']}")
    else:
        st.info("No models trained yet.")
    if "promoted_model" in st.session_state:
        st.write(f"**Promoted:** `{st.session_state['promoted_model']['name']}`")
    if st.session_state.get("task_type"):
        st.write(f"**Task:** {st.session_state['task_type']}")

# ── Guard clause ───────────────────────────────────────────────────────────────
st.title("🎯 Prediction & Inference")

if "preprocessing_pipeline" not in st.session_state:
    st.warning("Please complete the **Preprocessing** and **Training** steps first.")
    st.stop()

if "trained_models" not in st.session_state:
    st.warning("Please train at least one model on the **Training** page first.")
    st.stop()

# ── Shared state ───────────────────────────────────────────────────────────────
task_type: str = st.session_state.get("task_type", "classification")
preprocessing_pipeline = st.session_state["preprocessing_pipeline"]
column_config: dict = st.session_state.get("column_config", {})
column_types: dict = st.session_state.get("column_types", {})
target_column: str = st.session_state.get("target_column", "")
df_original: pd.DataFrame = st.session_state["df"]
results: list[TrainingResult] = st.session_state["training_results"]
summary: dict = st.session_state["training_summary"]

# Datetime columns that were extracted (not dropped) during preprocessing
datetime_extract_cols: list[str] = [
    col for col, cfg in column_config.items()
    if cfg.get("type") == "datetime" and cfg.get("action") == "extract"
    and col in df_original.columns
]

# Original feature columns (what the user should provide)
feature_cols: list[str] = [c for c in df_original.columns if c != target_column]

# Columns to show in the form (skip purely dropped ones)
_drop_types = {"text"}
_form_cols: list[str] = [
    c for c in feature_cols
    if column_types.get(c) not in _drop_types
    and not (
        column_types.get(c) == "datetime" and column_config.get(c, {}).get("action") == "drop"
    )
]


# ── Preprocessing helper ───────────────────────────────────────────────────────
def preprocess_input(df_new: pd.DataFrame) -> np.ndarray | None:
    """Apply the same preprocessing as during training to new raw data."""
    df_proc = df_new.copy()
    if datetime_extract_cols:
        # Keep only cols present in the new data
        to_extract = [c for c in datetime_extract_cols if c in df_proc.columns]
        if to_extract:
            df_proc = extract_datetime_features(df_proc, to_extract)
    try:
        return preprocessing_pipeline.transform(df_proc)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None


def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (predictions, probabilities|None)."""
    y_pred = model.predict(X)
    y_proba = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            pass
    return y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model Selection
# ══════════════════════════════════════════════════════════════════════════════
st.header("Model Selection")

model_source = st.radio(
    "Load model from",
    ["Session (trained models)", "MLflow Registry"],
    horizontal=True,
    key="model_source",
)

active_model = None
active_model_name = ""

if model_source == "Session (trained models)":
    trained_models: dict = st.session_state["trained_models"]
    model_options = [r.model_name for r in results]
    default_idx = next(
        (i for i, r in enumerate(results) if r.model_name == summary["best_model"]), 0
    )
    chosen_name = st.selectbox(
        "Choose model", model_options, index=default_idx, key="chosen_model"
    )
    chosen_result = next(r for r in results if r.model_name == chosen_name)
    active_model = chosen_result.model
    active_model_name = chosen_name

    with st.expander("Model details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Parameters")
            st.json(chosen_result.params)
        with c2:
            st.caption("Validation Metrics")
            for k, v in chosen_result.metrics.items():
                st.write(f"- **{k}:** {v:.4f}")
            if chosen_result.cv_mean is not None:
                st.write(f"- **CV mean:** {chosen_result.cv_mean:.4f} ± {chosen_result.cv_std:.4f}")

else:  # MLflow Registry
    def _list_registry_models() -> list[str] | None:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()
            return [m.name for m in client.search_registered_models()]
        except Exception:
            return None

    registry_models = _list_registry_models()

    if registry_models is None:
        st.warning(
            "MLflow server not reachable at `http://localhost:5000`. "
            "Start it with `mlflow ui`."
        )
        st.stop()
    elif not registry_models:
        st.info("No registered models found. Promote a model from the **Results** page first.")
        st.stop()
    else:
        c_reg1, c_reg2, c_reg3 = st.columns([2, 1, 1])
        with c_reg1:
            reg_name = st.selectbox("Registered model", registry_models, key="reg_model_name")
        with c_reg2:
            reg_stage = st.selectbox("Version / Stage", ["latest", "1", "2", "3"], key="reg_stage")
        with c_reg3:
            st.write("")
            st.write("")
            if st.button("Load", type="secondary"):
                with st.spinner("Loading from registry..."):
                    try:
                        import mlflow.sklearn
                        loaded = mlflow.sklearn.load_model(f"models:/{reg_name}/{reg_stage}")
                        st.session_state["registry_model"] = loaded
                        st.session_state["registry_model_name"] = reg_name
                        st.success(f"Loaded **{reg_name}** ({reg_stage}).")
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        if "registry_model" in st.session_state:
            active_model = st.session_state["registry_model"]
            active_model_name = st.session_state.get("registry_model_name", "Registry Model")
            st.info(f"Active model: **{active_model_name}**")
        else:
            st.info("Load a model from the registry above to continue.")
            st.stop()

if active_model is None:
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Prediction Modes
# ══════════════════════════════════════════════════════════════════════════════
tab_batch, tab_single = st.tabs(["Batch Prediction (CSV)", "Single Prediction (Form)"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB A: Batch Prediction
# ──────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Batch Prediction")

    # Template download
    template_cols = [c for c in feature_cols]
    template_df = df_original[template_cols].head(3).copy()
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV Template",
        data=template_csv,
        file_name="prediction_template.csv",
        mime="text/csv",
        help="Download a sample CSV showing the expected columns and format.",
    )

    st.caption(
        f"Upload a CSV with the same feature columns as the training data "
        f"(**{len(template_cols)} columns**, target `{target_column}` excluded)."
    )

    uploaded = st.file_uploader(
        "Upload prediction CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_new = None

        if df_new is not None:
            # Drop target column if accidentally included
            if target_column in df_new.columns:
                df_new = df_new.drop(columns=[target_column])
                st.caption(f"Column `{target_column}` was detected and removed from the input.")

            st.write(f"**Uploaded:** {len(df_new):,} rows × {df_new.shape[1]} columns")
            st.dataframe(df_new.head(10), use_container_width=True)

            # Check for expected columns
            missing_cols = [c for c in feature_cols if c not in df_new.columns
                            and column_types.get(c) not in ("text",)
                            and not (column_types.get(c) == "datetime"
                                     and column_config.get(c, {}).get("action") == "drop")]
            if missing_cols:
                st.warning(f"Missing columns in uploaded file: `{missing_cols}`")

            if st.button("▶ Run Batch Prediction", type="primary", key="run_batch"):
                with st.spinner("Preprocessing and predicting..."):
                    X_new = preprocess_input(df_new)

                if X_new is not None:
                    y_pred, y_proba = predict(active_model, X_new)

                    result_df = df_new.copy()
                    result_df["prediction"] = y_pred

                    if task_type == "classification" and y_proba is not None:
                        classes = getattr(active_model, "classes_", None)
                        if classes is None:
                            # Try to infer from training labels
                            try:
                                classes = np.unique(
                                    np.asarray(st.session_state.get("y_train", []))
                                )
                            except Exception:
                                classes = list(range(y_proba.shape[1]))
                        for i, cls in enumerate(classes):
                            result_df[f"prob_{cls}"] = y_proba[:, i].round(4)
                        result_df["confidence"] = y_proba.max(axis=1).round(4)

                    st.session_state["batch_result_df"] = result_df
                    st.success(f"Predicted {len(result_df):,} rows.")

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        # Summary stats
        if task_type == "classification":
            pred_counts = result_df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["Class", "Count"]
            pred_counts["Percentage"] = (pred_counts["Count"] / len(result_df) * 100).round(1)
            st.caption("Prediction distribution:")
            st.dataframe(pred_counts, use_container_width=True, hide_index=True)
        else:
            pred_series = result_df["prediction"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{pred_series.mean():.4f}")
            m2.metric("Std", f"{pred_series.std():.4f}")
            m3.metric("Min", f"{pred_series.min():.4f}")
            m4.metric("Max", f"{pred_series.max():.4f}")

        # Download
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB B: Single Prediction
# ──────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.subheader("Single Prediction")
    st.caption("Fill in the feature values below and click **Predict**.")

    form_values: dict = {}

    # Lay out 3 inputs per row
    form_cols_list = _form_cols
    n_cols = 3
    rows = [form_cols_list[i:i + n_cols] for i in range(0, len(form_cols_list), n_cols)]

    for row_cols in rows:
        ui_cols = st.columns(len(row_cols))
        for ui_col, col_name in zip(ui_cols, row_cols):
            ctype = column_types.get(col_name, "text")
            with ui_col:
                if ctype == "numerical":
                    col_data = df_original[col_name].dropna()
                    mean_val = float(col_data.mean()) if len(col_data) > 0 else 0.0
                    min_val = float(col_data.min()) if len(col_data) > 0 else 0.0
                    max_val = float(col_data.max()) if len(col_data) > 0 else 1.0
                    form_values[col_name] = st.number_input(
                        col_name,
                        value=round(mean_val, 4),
                        min_value=min_val - abs(min_val),
                        max_value=max_val + abs(max_val),
                        key=f"form_{col_name}",
                        help=f"Range: [{min_val:.2f}, {max_val:.2f}]  Mean: {mean_val:.2f}",
                    )
                elif ctype == "categorical":
                    options = sorted(df_original[col_name].dropna().unique().tolist(), key=str)
                    form_values[col_name] = st.selectbox(
                        col_name,
                        options=options,
                        key=f"form_{col_name}",
                    )
                elif ctype == "datetime":
                    import datetime
                    form_values[col_name] = st.date_input(
                        col_name,
                        value=datetime.date.today(),
                        key=f"form_{col_name}",
                    )
                else:
                    form_values[col_name] = st.text_input(col_name, key=f"form_{col_name}")

    if st.button("▶ Predict", type="primary", key="run_single"):
        # Build a single-row DataFrame with all original feature columns
        row_data: dict = {}
        for col in feature_cols:
            if col in form_values:
                row_data[col] = [form_values[col]]
            else:
                # Column not shown in form (dropped) — fill with NaN
                row_data[col] = [np.nan]

        df_row = pd.DataFrame(row_data)

        with st.spinner("Predicting..."):
            X_row = preprocess_input(df_row)

        if X_row is not None:
            y_pred, y_proba = predict(active_model, X_row)
            pred_val = y_pred[0]

            st.divider()
            if task_type == "classification":
                st.subheader("Prediction Result")
                result_col, conf_col = st.columns(2)
                with result_col:
                    st.metric("Predicted Class", str(pred_val))
                if y_proba is not None:
                    confidence = float(y_proba.max())
                    with conf_col:
                        st.metric("Confidence", f"{confidence:.1%}")

                    # Probability bar per class
                    classes = getattr(active_model, "classes_", None)
                    if classes is None:
                        try:
                            classes = np.unique(
                                np.asarray(st.session_state.get("y_train", []))
                            )
                        except Exception:
                            classes = list(range(y_proba.shape[1]))

                    import plotly.graph_objects as go
                    proba_fig = go.Figure(
                        go.Bar(
                            x=[str(c) for c in classes],
                            y=y_proba[0],
                            marker_color="#4ECDC4",
                            text=[f"{p:.1%}" for p in y_proba[0]],
                            textposition="outside",
                        )
                    )
                    proba_fig.update_layout(
                        title="Class Probabilities",
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(proba_fig, use_container_width=True)
            else:
                st.subheader("Prediction Result")
                st.metric("Predicted Value", f"{pred_val:.4f}")

            # Show the inputs used
            with st.expander("Input values used", expanded=False):
                input_display = {k: v for k, v in form_values.items()}
                st.json(input_display)

# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/4_Results.py", label="← Back to Results")
with nav2:
    st.page_link("pages/6_Monitoring.py", label="Continue to Monitoring →")
