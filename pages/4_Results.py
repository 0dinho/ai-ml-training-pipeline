"""Phase 5 — Results & Metrics Dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve

from src.pipelines.training import TrainingResult, compute_metrics

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
MLFLOW_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "automl-experiments"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Status")
    if "training_results" in st.session_state:
        s = st.session_state["training_summary"]
        st.success(f"{s['models_trained']} model(s) trained")
        st.write(f"**Best:** {s['best_model']}")
        st.write(f"**{s['primary_metric']}:** {s['best_score']:.4f}")
    else:
        st.info("No models trained yet.")
    if st.session_state.get("task_type"):
        st.write(f"**Task:** {st.session_state['task_type']}")

# ── Guard clause ───────────────────────────────────────────────────────────────
st.title("📈 Results & Metrics")

if "training_results" not in st.session_state:
    st.warning("Please train at least one model on the **Training** page first.")
    st.stop()

results: list[TrainingResult] = st.session_state["training_results"]
summary: dict = st.session_state["training_summary"]
task_type: str = st.session_state.get("task_type", "classification")
feature_names: list[str] = st.session_state.get("feature_names", [])

X_train: np.ndarray = st.session_state["X_train"]
y_train = np.asarray(st.session_state["y_train"])
X_val: np.ndarray = st.session_state["X_val"]
y_val = np.asarray(st.session_state["y_val"])
X_test: np.ndarray = st.session_state["X_test"]
y_test = np.asarray(st.session_state["y_test"])

_colors = {r.model_type: COLORS[i % len(COLORS)] for i, r in enumerate(results)}

# ── Pre-compute test-set predictions ──────────────────────────────────────────
test_preds: dict = {}
for r in results:
    y_pred = r.model.predict(X_test)
    y_proba = None
    if task_type == "classification" and hasattr(r.model, "predict_proba"):
        try:
            y_proba = r.model.predict_proba(X_test)
        except Exception:
            pass
    test_preds[r.model_type] = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": compute_metrics(y_test, y_pred, task_type, y_proba),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Metrics Summary
# ══════════════════════════════════════════════════════════════════════════════
st.header("Metrics Summary")

if task_type == "classification":
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    col_labels = {
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "AUC-ROC",
    }
else:
    metric_cols = ["mse", "rmse", "mae", "r2"]
    col_labels = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}


def _build_metrics_df(source: str) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.metrics if source == "val" else test_preds[r.model_type]["metrics"]
        row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
        for mc in metric_cols:
            v = m.get(mc)
            row[col_labels[mc]] = f"{v:.4f}" if v is not None else "—"
        if r.cv_mean is not None:
            row["CV Mean ± Std"] = f"{r.cv_mean:.4f} ± {r.cv_std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _highlight_best(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    best = summary["best_model"]

    def _apply(row):
        if row["Model"] == best:
            return ["background-color: rgba(78,205,196,0.15)"] * len(row)
        return [""] * len(row)

    return df.style.apply(_apply, axis=1)


tab_val, tab_test = st.tabs(["Validation Set", "Test Set"])

with tab_val:
    st.caption("Metrics computed on the validation set during training.")
    st.dataframe(_highlight_best(_build_metrics_df("val")), use_container_width=True, hide_index=True)

with tab_test:
    st.caption("Final evaluation on the held-out test set.")
    st.dataframe(_highlight_best(_build_metrics_df("test")), use_container_width=True, hide_index=True)

# Best model callout (test set)
best_result = next(r for r in results if r.model_name == summary["best_model"])
primary = summary["primary_metric"]
best_test_score = test_preds[best_result.model_type]["metrics"].get(primary, 0)
st.success(
    f"**Best model (test set):** {summary['best_model']} — "
    f"{primary}: {best_test_score:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
st.header("Visualizations")

if task_type == "classification":
    tab_cm, tab_roc, tab_fi, tab_lc = st.tabs(
        ["Confusion Matrix", "ROC Curve", "Feature Importance", "Learning Curves"]
    )
else:
    tab_res, tab_avp, tab_fi, tab_lc = st.tabs(
        ["Residual Plot", "Actual vs Predicted", "Feature Importance", "Learning Curves"]
    )

# ── Classification-specific plots ─────────────────────────────────────────────
if task_type == "classification":
    classes = np.unique(y_test)

    # Confusion matrix
    with tab_cm:
        n_cols = min(len(results), 3)
        cols_cm = st.columns(n_cols)
        for idx, r in enumerate(results):
            y_pred = test_preds[r.model_type]["y_pred"]
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=[str(c) for c in classes],
                    y=[str(c) for c in classes],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig.update_layout(
                title=r.model_name,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                **PLOTLY_LAYOUT,
            )
            with cols_cm[idx % n_cols]:
                st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    with tab_roc:
        n_classes = len(classes)
        if n_classes == 2:
            fig = go.Figure()
            any_roc = False
            for r in results:
                tp = test_preds[r.model_type]
                if tp["y_proba"] is not None:
                    fpr, tpr, _ = roc_curve(y_test, tp["y_proba"][:, 1])
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{r.model_name} (AUC={roc_auc:.4f})",
                            line=dict(color=_colors[r.model_type], width=2),
                        )
                    )
                    any_roc = True
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random Classifier",
                    line=dict(color="gray", width=1, dash="dash"),
                )
            )
            fig.update_layout(
                title="ROC Curves — Binary Classification",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **PLOTLY_LAYOUT,
            )
            if any_roc:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC curve requires probability estimates (predict_proba). Not available for the trained model(s).")
        else:
            # Multi-class OVR
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=classes)
            for r in results:
                tp = test_preds[r.model_type]
                if tp["y_proba"] is None:
                    st.info(f"No probability estimates for {r.model_name}.")
                    continue
                fig = go.Figure()
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], tp["y_proba"][:, i])
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"Class {cls} (AUC={roc_auc:.3f})",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Baseline", line=dict(color="gray", dash="dash"),
                    )
                )
                fig.update_layout(
                    title=f"ROC Curve (OVR) — {r.model_name}",
                    xaxis_title="FPR", yaxis_title="TPR",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Regression-specific plots ──────────────────────────────────────────────────
else:
    # Residual plot
    with tab_res:
        n_cols = min(len(results), 3)
        cols_res = st.columns(n_cols)
        for idx, r in enumerate(results):
            y_pred = test_preds[r.model_type]["y_pred"]
            residuals = y_test - y_pred
            fig = go.Figure(
                go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(color=_colors[r.model_type], opacity=0.6, size=5),
                    name=r.model_name,
                )
            )
            fig.add_hline(y=0, line_color="red", line_dash="dash")
            fig.update_layout(
                title=r.model_name,
                xaxis_title="Predicted", yaxis_title="Residuals",
                **PLOTLY_LAYOUT,
            )
            with cols_res[idx % n_cols]:
                st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    with tab_avp:
        n_cols = min(len(results), 3)
        cols_avp = st.columns(n_cols)
        for idx, r in enumerate(results):
            y_pred = test_preds[r.model_type]["y_pred"]
            mn = float(min(y_test.min(), y_pred.min()))
            mx = float(max(y_test.max(), y_pred.max()))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=y_test, y=y_pred, mode="markers",
                    marker=dict(color=_colors[r.model_type], opacity=0.6, size=5),
                    name="Predictions",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="red", dash="dash"), name="Perfect fit",
                )
            )
            fig.update_layout(
                title=r.model_name,
                xaxis_title="Actual", yaxis_title="Predicted",
                **PLOTLY_LAYOUT,
            )
            with cols_avp[idx % n_cols]:
                st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance (both task types) ──────────────────────────────────────
with tab_fi:
    n_features = X_train.shape[1]
    feat_names = feature_names if feature_names else [f"feature_{i}" for i in range(n_features)]
    top_n = st.slider("Top N features", 5, min(30, n_features), min(15, n_features), key="top_n_fi")

    has_any = False
    for r in results:
        importances = getattr(r.model, "feature_importances_", None)
        if importances is None:
            continue
        has_any = True
        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_names = [feat_names[i] if i < len(feat_names) else f"feature_{i}" for i in top_idx]
        top_vals = importances[top_idx]

        fig = go.Figure(
            go.Bar(
                x=top_vals, y=top_names, orientation="h",
                marker_color=_colors[r.model_type],
            )
        )
        fig.update_layout(
            title=f"Feature Importance — {r.model_name}",
            xaxis_title="Importance",
            yaxis=dict(autorange="reversed"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not has_any:
        st.info(
            "Feature importance is not natively available for the trained model(s). "
            "Tree-based models (Random Forest, XGBoost) expose `feature_importances_`; "
            "Neural Networks do not."
        )

# ── Learning Curves (both task types) ─────────────────────────────────────────
with tab_lc:
    st.caption(
        "Learning curves show model performance vs. training set size. "
        "Each curve requires multiple re-fits — this may take a while for large datasets or neural networks."
    )
    if st.button("Compute Learning Curves", type="secondary"):
        from sklearn.base import clone

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        scoring = "accuracy" if task_type == "classification" else "r2"

        for r in results:
            with st.spinner(f"Computing learning curve for {r.model_name}..."):
                try:
                    cloned = clone(r.model)
                    train_sizes, train_scores, val_scores = learning_curve(
                        cloned, X_combined, y_combined,
                        cv=3,
                        scoring=scoring,
                        train_sizes=np.linspace(0.1, 1.0, 8),
                        n_jobs=-1,
                    )
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=train_sizes,
                            y=train_scores.mean(axis=1),
                            mode="lines+markers",
                            name="Train",
                            line=dict(color=_colors[r.model_type]),
                            error_y=dict(type="data", array=train_scores.std(axis=1), visible=True),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=train_sizes,
                            y=val_scores.mean(axis=1),
                            mode="lines+markers",
                            name="Cross-val",
                            line=dict(color=_colors[r.model_type], dash="dash"),
                            error_y=dict(type="data", array=val_scores.std(axis=1), visible=True),
                        )
                    )
                    fig.update_layout(
                        title=f"Learning Curve — {r.model_name}",
                        xaxis_title="Training Examples",
                        yaxis_title=scoring.capitalize(),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute learning curve for {r.model_name}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — MLflow Experiment History
# ══════════════════════════════════════════════════════════════════════════════
st.header("MLflow Experiment History")


def _load_mlflow_runs() -> pd.DataFrame | None:
    """Return DataFrame of runs or None if MLflow is unreachable."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=100,
        )
        if not runs:
            return pd.DataFrame()
        rows = []
        for run in runs:
            end = run.info.end_time
            start = run.info.start_time
            duration = f"{(end - start) / 1000:.1f}" if end else "—"
            row: dict = {
                "Run ID": run.info.run_id[:8],
                "Run Name": run.info.run_name or "—",
                "Status": run.info.status,
                "Start Time": pd.to_datetime(start, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "Duration (s)": duration,
            }
            for metric in ["accuracy", "f1", "roc_auc", "r2", "rmse", "mae", "cv_mean"]:
                v = run.data.metrics.get(metric)
                row[metric] = f"{v:.4f}" if v is not None else "—"
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return None


mlflow_df = _load_mlflow_runs()

if mlflow_df is None:
    st.warning(
        "MLflow tracking server is not reachable at `http://localhost:5001`. "
        "Start it with `mlflow ui` to view experiment history."
    )
elif mlflow_df.empty:
    st.info("No MLflow runs found for the current experiment.")
else:
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        name_filter = st.text_input("Filter by run name", key="mlflow_filter")
    with fc2:
        status_filter = st.selectbox(
            "Status", ["All", "FINISHED", "RUNNING", "FAILED"], key="mlflow_status"
        )

    filtered = mlflow_df.copy()
    if name_filter:
        filtered = filtered[
            filtered["Run Name"].str.contains(name_filter, case=False, na=False)
        ]
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(mlflow_df)} runs.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Promote Model to Registry
# ══════════════════════════════════════════════════════════════════════════════
st.header("Promote Model to Registry")

col_r1, col_r2 = st.columns([2, 1])
with col_r1:
    registry_name = st.text_input(
        "Registry model name",
        value=f"automl-{best_result.model_type}",
        key="registry_name",
    )
with col_r2:
    promote_choice = st.selectbox(
        "Model to promote",
        options=[r.model_name for r in results],
        index=next(i for i, r in enumerate(results) if r.model_name == summary["best_model"]),
        key="promote_model_select",
    )

selected: TrainingResult = next(r for r in results if r.model_name == promote_choice)

st.write(
    f"Promoting **{selected.model_name}** — "
    f"test {primary}: **{test_preds[selected.model_type]['metrics'].get(primary, 0):.4f}**"
)

if st.button("🏆 Promote to MLflow Registry", type="primary"):
    with st.spinner("Registering model..."):
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)

            if selected.mlflow_run_id:
                with mlflow.start_run(run_id=selected.mlflow_run_id):
                    mlflow.sklearn.log_model(
                        selected.model, "model",
                        registered_model_name=registry_name,
                    )
                msg = (
                    f"Model logged to existing run `{selected.mlflow_run_id[:8]}` "
                    f"and registered as **{registry_name}**."
                )
            else:
                with mlflow.start_run(run_name=f"{selected.model_name} (Registry)") as run:
                    mlflow.log_params({k: str(v) for k, v in selected.params.items()})
                    mlflow.log_metrics(selected.metrics)
                    mlflow.sklearn.log_model(
                        selected.model, "model",
                        registered_model_name=registry_name,
                    )
                msg = (
                    f"Model logged in new run `{run.info.run_id[:8]}` "
                    f"and registered as **{registry_name}**."
                )

            st.success(msg)
            st.session_state["promoted_model"] = {
                "name": registry_name,
                "model_type": selected.model_type,
                "metrics": selected.metrics,
            }

        except Exception as e:
            st.error(
                f"Registration failed: {e}\n\n"
                "Make sure the MLflow server is running: `mlflow ui --port 5001`"
            )

if "promoted_model" in st.session_state:
    pm = st.session_state["promoted_model"]
    st.info(f"**Currently promoted:** `{pm['name']}` ({pm['model_type']})")

# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("pages/3_Training.py", label="← Back to Training")
with nav2:
    st.page_link("pages/5_Prediction.py", label="Continue to Prediction →")
