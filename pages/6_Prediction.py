"""Phase 6 — Prediction & Inference (all 6 task types)."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st

from src.pipelines.preprocessing import extract_datetime_features
from src.pipelines.training import TrainingResult, normalize_task_type, is_supervised

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
MLFLOW_URI = "http://localhost:5001"
UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}

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
task_type: str = normalize_task_type(st.session_state.get("task_type", "binary_classification"))
supervised = is_supervised(task_type)
preprocessing_pipeline = st.session_state["preprocessing_pipeline"]
column_config: dict = st.session_state.get("column_config", {})
column_types: dict = st.session_state.get("column_types", {})
target_column: str = st.session_state.get("target_column", "")
df_original: pd.DataFrame = st.session_state["df"]
results: list[TrainingResult] = st.session_state["training_results"]
summary: dict = st.session_state["training_summary"]
result_adapters: dict = st.session_state.get("training_adapters", {})

# Datetime columns that were extracted (not dropped)
datetime_extract_cols: list[str] = [
    col for col, cfg in column_config.items()
    if cfg.get("type") == "datetime" and cfg.get("action") == "extract"
    and col in df_original.columns
]

# Original feature columns
feature_cols: list[str] = [c for c in df_original.columns if c != target_column]

# Columns to show in the form
_drop_types = {"text"}
_form_cols: list[str] = [
    c for c in feature_cols
    if column_types.get(c) not in _drop_types
    and not (
        column_types.get(c) == "datetime"
        and column_config.get(c, {}).get("action") == "drop"
    )
]


# ── Preprocessing helper ───────────────────────────────────────────────────────
def preprocess_input(df_new: pd.DataFrame) -> np.ndarray | None:
    df_proc = df_new.copy()
    if datetime_extract_cols:
        to_extract = [c for c in datetime_extract_cols if c in df_proc.columns]
        if to_extract:
            df_proc = extract_datetime_features(df_proc, to_extract)
    try:
        return preprocessing_pipeline.transform(df_proc)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None


def _run_adapter_prediction(adapter, X: np.ndarray) -> dict:
    """Dispatch prediction through a ModelAdapter and return a result dict."""
    out: dict = {}
    if adapter.is_reduction():
        try:
            out["reduced_coords"] = adapter.transform(X)
        except NotImplementedError:
            out["tsne_warning"] = True
            stored = st.session_state.get("reduction_output")
            out["reduced_coords"] = stored
    elif adapter.is_anomaly():
        preds = adapter.predict(X)
        scores = adapter.decision_scores(X)
        out["predictions"] = preds
        out["anomaly_scores"] = scores
    elif adapter.is_clustering():
        out["predictions"] = adapter.predict(X)
    else:
        # Supervised
        out["predictions"] = adapter.predict(X)
        out["probabilities"] = adapter.predict_proba(X)
    return out


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
active_adapter = None
active_model_name = ""

if model_source == "Session (trained models)":
    model_options = [r.model_name for r in results]
    default_idx = next(
        (i for i, r in enumerate(results) if r.model_name == summary["best_model"]), 0
    )
    chosen_name = st.selectbox(
        "Choose model", model_options, index=default_idx, key="chosen_model"
    )
    chosen_result = next(r for r in results if r.model_name == chosen_name)
    active_model = chosen_result.model
    active_adapter = result_adapters.get(chosen_result.model_type)
    active_model_name = chosen_name

    with st.expander("Model details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Parameters")
            st.json(chosen_result.params)
        with c2:
            st.caption("Metrics")
            for k, v in chosen_result.metrics.items():
                if isinstance(v, float):
                    st.write(f"- **{k}:** {v:.4f}")
                else:
                    st.write(f"- **{k}:** {v}")
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
        st.warning("MLflow server not reachable at `http://localhost:5001`.")
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
    template_cols = list(feature_cols)
    template_df = df_original[template_cols].head(3).copy()
    st.download_button(
        "⬇ Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    st.caption(
        f"Upload a CSV with the same feature columns as training "
        f"(**{len(template_cols)} columns**, target `{target_column or '—'}` excluded)."
    )

    if task_type == "dimensionality_reduction":
        st.info("⚠️ t-SNE models cannot project new data — other reduction models (PCA, UMAP, TruncatedSVD) work normally.")

    uploaded = st.file_uploader("Upload prediction CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_new = None

        if df_new is not None:
            if target_column and target_column in df_new.columns:
                df_new = df_new.drop(columns=[target_column])
                st.caption(f"Column `{target_column}` was removed from input.")

            st.write(f"**Uploaded:** {len(df_new):,} rows × {df_new.shape[1]} columns")
            st.dataframe(df_new.head(10), width='stretch')

            missing_cols = [
                c for c in feature_cols
                if c not in df_new.columns
                and column_types.get(c) not in ("text",)
                and not (column_types.get(c) == "datetime"
                         and column_config.get(c, {}).get("action") == "drop")
            ]
            if missing_cols:
                st.warning(f"Missing columns: `{missing_cols}`")

            # ── Regression prediction-interval options ────────────────────────
            _batch_show_ci = False
            _batch_ci_level = 95
            if task_type == "regression":
                with st.expander("Prediction Interval Options", expanded=False):
                    _batch_show_ci = st.checkbox(
                        "Add prediction interval columns (ci_low, ci_high)",
                        value=True,
                        key="batch_ci_enable",
                    )
                    if _batch_show_ci:
                        _batch_ci_level = st.slider(
                            "Confidence level (%)", 80, 99, 95, key="batch_ci_level"
                        )
                        st.caption(
                            "Interval estimated from training residuals: "
                            "`ci = prediction ± quantile(y_train − ŷ_train)`"
                        )

            if st.button("▶ Run Batch Prediction", type="primary", key="run_batch"):
                with st.spinner("Preprocessing and predicting..."):
                    X_new = preprocess_input(df_new)

                if X_new is not None:
                    result_df = df_new.copy()

                    if active_adapter is not None:
                        pred_out = _run_adapter_prediction(active_adapter, X_new)
                    else:
                        # Fallback to raw model
                        pred_out = {
                            "predictions": active_model.predict(X_new),
                            "probabilities": None,
                        }
                        if supervised and hasattr(active_model, "predict_proba"):
                            try:
                                pred_out["probabilities"] = active_model.predict_proba(X_new)
                            except Exception:
                                pass

                    # Build output columns by task type
                    if task_type == "dimensionality_reduction":
                        coords = pred_out.get("reduced_coords")
                        if pred_out.get("tsne_warning"):
                            st.warning("t-SNE cannot project new data points. Showing training projection instead.")
                        if coords is not None:
                            for i in range(coords.shape[1]):
                                result_df[f"coord_{i}"] = coords[:, i].round(6) if len(coords) == len(df_new) else np.nan

                    elif task_type == "anomaly_detection":
                        preds = pred_out.get("predictions")
                        scores = pred_out.get("anomaly_scores")
                        if preds is not None:
                            result_df["is_anomaly"] = (preds == 1).astype(bool)
                            result_df["prediction"] = preds
                        if scores is not None:
                            result_df["anomaly_score"] = scores.round(6)

                    elif task_type == "clustering":
                        result_df["cluster_id"] = pred_out.get("predictions")

                    else:
                        # Supervised
                        y_pred = pred_out.get("predictions")
                        y_proba = pred_out.get("probabilities")
                        result_df["prediction"] = y_pred

                        # Regression prediction interval (residual-based)
                        if task_type == "regression" and _batch_show_ci and y_pred is not None:
                            _X_tr = st.session_state.get("X_train")
                            _y_tr = st.session_state.get("y_train")
                            if _X_tr is not None and _y_tr is not None:
                                try:
                                    _m = active_adapter if active_adapter is not None else active_model
                                    _tr_preds = _m.predict(_X_tr)
                                    _residuals = (
                                        np.asarray(_y_tr, dtype=float)
                                        - np.asarray(_tr_preds, dtype=float)
                                    )
                                    _tail = (100 - _batch_ci_level) / 200.0
                                    _q_lo = float(np.quantile(_residuals, _tail))
                                    _q_hi = float(np.quantile(_residuals, 1.0 - _tail))
                                    _y_arr = np.asarray(y_pred, dtype=float)
                                    result_df["ci_low"] = (_y_arr + _q_lo).round(4)
                                    result_df["ci_high"] = (_y_arr + _q_hi).round(4)
                                except Exception:
                                    pass

                        if task_type in ("binary_classification", "multiclass_classification") and y_proba is not None:
                            classes = getattr(active_model, "classes_", None)
                            if classes is None and active_adapter is not None:
                                classes = active_adapter.classes_ or list(range(y_proba.shape[1]))
                            if classes is None:
                                classes = list(range(y_proba.shape[1]))
                            for i, cls in enumerate(classes):
                                result_df[f"prob_{cls}"] = y_proba[:, i].round(4)
                            result_df["confidence"] = y_proba.max(axis=1).round(4)

                    st.session_state["batch_result_df"] = result_df
                    st.success(f"Predicted {len(result_df):,} rows.")

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        st.subheader("Prediction Results")
        st.dataframe(result_df, width='stretch')

        # Summary stats
        if task_type in ("binary_classification", "multiclass_classification"):
            pred_counts = result_df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["Class", "Count"]
            pred_counts["Percentage"] = (pred_counts["Count"] / len(result_df) * 100).round(1)
            st.caption("Prediction distribution:")
            st.dataframe(pred_counts, width='stretch', hide_index=True)

        elif task_type == "regression" and "prediction" in result_df.columns:
            pred_series = result_df["prediction"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean", f"{pred_series.mean():.4f}")
            m2.metric("Std", f"{pred_series.std():.4f}")
            m3.metric("Min", f"{pred_series.min():.4f}")
            m4.metric("Max", f"{pred_series.max():.4f}")

        elif task_type == "clustering" and "cluster_id" in result_df.columns:
            cluster_counts = result_df["cluster_id"].value_counts().sort_index()
            st.caption("Cluster distribution:")
            st.bar_chart(cluster_counts)

        elif task_type == "anomaly_detection" and "is_anomaly" in result_df.columns:
            n_anom = int(result_df["is_anomaly"].sum())
            m1, m2 = st.columns(2)
            m1.metric("Anomalies", n_anom)
            m2.metric("Anomaly Rate", f"{100 * n_anom / len(result_df):.1f}%")

        # Download
        st.download_button(
            "⬇ Download Predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────────────
# TAB B: Single Prediction
# ──────────────────────────────────────────────────────────────────────────────
with tab_single:
    st.subheader("Single Prediction")
    st.caption("Fill in the feature values below and click **Predict**.")

    if task_type == "dimensionality_reduction":
        st.info("Single prediction returns reduced coordinates (new point projection).")

    form_values: dict = {}
    n_cols = 3
    rows = [_form_cols[i:i + n_cols] for i in range(0, len(_form_cols), n_cols)]

    for row_items in rows:
        ui_cols = st.columns(len(row_items))
        for ui_col, col_name in zip(ui_cols, row_items):
            ctype = column_types.get(col_name, "text")
            with ui_col:
                if ctype == "numerical":
                    col_data = df_original[col_name].dropna()
                    mean_val = float(col_data.mean()) if len(col_data) > 0 else 0.0
                    min_val  = float(col_data.min())  if len(col_data) > 0 else 0.0
                    max_val  = float(col_data.max())  if len(col_data) > 0 else 1.0
                    form_values[col_name] = st.number_input(
                        col_name, value=round(mean_val, 4),
                        min_value=min_val - abs(min_val),
                        max_value=max_val + abs(max_val),
                        key=f"form_{col_name}",
                        help=f"Range: [{min_val:.2f}, {max_val:.2f}]  Mean: {mean_val:.2f}",
                    )
                elif ctype == "categorical":
                    options = sorted(df_original[col_name].dropna().unique().tolist(), key=str)
                    form_values[col_name] = st.selectbox(col_name, options, key=f"form_{col_name}")
                elif ctype == "datetime":
                    import datetime
                    form_values[col_name] = st.date_input(col_name, value=datetime.date.today(), key=f"form_{col_name}")
                else:
                    form_values[col_name] = st.text_input(col_name, key=f"form_{col_name}")

    # ── Regression prediction-interval options ────────────────────────────────
    _single_show_ci = False
    _single_ci_level = 95
    if task_type == "regression":
        _single_show_ci = st.checkbox(
            "Show prediction interval", value=True, key="single_ci_enable"
        )
        if _single_show_ci:
            _single_ci_level = st.slider(
                "Confidence level (%)", 80, 99, 95, key="single_ci_level"
            )

    if st.button("▶ Predict", type="primary", key="run_single"):
        row_data: dict = {}
        for col in feature_cols:
            row_data[col] = [form_values[col]] if col in form_values else [np.nan]
        df_row = pd.DataFrame(row_data)

        with st.spinner("Predicting..."):
            X_row = preprocess_input(df_row)

        if X_row is not None:
            st.divider()
            st.subheader("Prediction Result")

            if active_adapter is not None:
                pred_out = _run_adapter_prediction(active_adapter, X_row)
            else:
                pred_out = {"predictions": active_model.predict(X_row), "probabilities": None}
                if supervised and hasattr(active_model, "predict_proba"):
                    try:
                        pred_out["probabilities"] = active_model.predict_proba(X_row)
                    except Exception:
                        pass

            # Task-type-specific display
            if task_type == "dimensionality_reduction":
                coords = pred_out.get("reduced_coords")
                if pred_out.get("tsne_warning"):
                    st.warning("t-SNE cannot project new data points.")
                elif coords is not None:
                    coord_row = coords[0] if len(coords) > 0 else coords
                    cols = st.columns(min(len(coord_row), 5))
                    for i, c in enumerate(coord_row):
                        cols[i % len(cols)].metric(f"Dim {i}", f"{c:.4f}")

            elif task_type == "anomaly_detection":
                preds = pred_out.get("predictions")
                scores = pred_out.get("anomaly_scores")
                pred_val = int(preds[0]) if preds is not None else None
                score_val = float(scores[0]) if scores is not None else None
                c1, c2 = st.columns(2)
                if pred_val is not None:
                    label = "ANOMALY" if pred_val == 1 else "Normal"
                    c1.metric("Classification", label)
                if score_val is not None:
                    c2.metric("Anomaly Score", f"{score_val:.4f}")

            elif task_type == "clustering":
                pred_val = int(pred_out.get("predictions")[0])
                st.metric("Assigned Cluster", f"Cluster {pred_val}")

            elif task_type in ("binary_classification", "multiclass_classification"):
                y_pred = pred_out.get("predictions")
                y_proba = pred_out.get("probabilities")
                pred_val = y_pred[0] if y_pred is not None else None
                c1, c2 = st.columns(2)
                c1.metric("Predicted Class", str(pred_val))
                if y_proba is not None:
                    confidence = float(y_proba.max())
                    c2.metric("Confidence", f"{confidence:.1%}")

                    classes = None
                    if active_adapter is not None and active_adapter.classes_:
                        classes = active_adapter.classes_
                    elif hasattr(active_model, "classes_"):
                        classes = active_model.classes_
                    if classes is None:
                        classes = list(range(y_proba.shape[1]))

                    import plotly.graph_objects as go
                    proba_fig = go.Figure(go.Bar(
                        x=[str(c) for c in classes], y=y_proba[0],
                        marker_color="#4ECDC4",
                        text=[f"{p:.1%}" for p in y_proba[0]],
                        textposition="outside",
                    ))
                    proba_fig.update_layout(
                        title="Class Probabilities", xaxis_title="Class",
                        yaxis_title="Probability", yaxis=dict(range=[0, 1]),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(proba_fig, width='stretch')

            else:  # regression
                y_pred = pred_out.get("predictions")
                pred_val = float(y_pred[0]) if y_pred is not None else None

                if pred_val is not None and _single_show_ci:
                    _X_tr = st.session_state.get("X_train")
                    _y_tr = st.session_state.get("y_train")
                    _ci_lo = _ci_hi = None
                    if _X_tr is not None and _y_tr is not None:
                        try:
                            _m = active_adapter if active_adapter is not None else active_model
                            _tr_preds = _m.predict(_X_tr)
                            _residuals = (
                                np.asarray(_y_tr, dtype=float)
                                - np.asarray(_tr_preds, dtype=float)
                            )
                            _tail = (100 - _single_ci_level) / 200.0
                            _ci_lo = pred_val + float(np.quantile(_residuals, _tail))
                            _ci_hi = pred_val + float(np.quantile(_residuals, 1.0 - _tail))
                        except Exception:
                            pass
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        f"CI Lower ({_single_ci_level}%)",
                        f"{_ci_lo:.4f}" if _ci_lo is not None else "—",
                    )
                    c2.metric("Predicted Value", f"{pred_val:.4f}")
                    c3.metric(
                        f"CI Upper ({_single_ci_level}%)",
                        f"{_ci_hi:.4f}" if _ci_hi is not None else "—",
                    )
                    if _ci_lo is not None:
                        st.caption(
                            f"**{_single_ci_level}% prediction interval** estimated "
                            "from training residuals."
                        )
                else:
                    st.metric(
                        "Predicted Value",
                        f"{pred_val:.4f}" if pred_val is not None else "—",
                    )

            with st.expander("Input values used", expanded=False):
                st.json({k: str(v) for k, v in form_values.items()})

# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/5_Results.py", label="← Back to Results")
with nav2:
    st.page_link("pages/7_Monitoring.py", label="Continue to Monitoring →")
