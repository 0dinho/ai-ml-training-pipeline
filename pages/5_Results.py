"""Phase 5 — Results & Metrics Dashboard (all 6 task types)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve

from src.pipelines.training import (
    TrainingResult,
    compute_metrics,
    normalize_task_type,
    is_supervised,
)

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
MLFLOW_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "automl-experiments"

UNSUPERVISED_TASKS = {"clustering", "dimensionality_reduction", "anomaly_detection"}

# Tree-based model types that support SHAP TreeExplainer
_TREE_BASED_TYPES = {
    "random_forest", "xgboost", "decision_tree", "gradient_boosting",
    "extra_trees", "gradient_boosting_regressor",
}

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
task_type: str = normalize_task_type(st.session_state.get("task_type", "binary_classification"))
supervised = is_supervised(task_type)
feature_names: list[str] = st.session_state.get(
    "feature_names_fe" if st.session_state.get("feature_engineering_applied") else "feature_names",
    []
)

# Resolve feature arrays
if st.session_state.get("feature_engineering_applied") and "X_train_fe" in st.session_state:
    X_train: np.ndarray = st.session_state["X_train_fe"]
    X_val_raw = st.session_state.get("X_val_fe")
    X_test_raw = st.session_state.get("X_test_fe")
else:
    X_train = st.session_state["X_train"]
    X_val_raw = st.session_state.get("X_val")
    X_test_raw = st.session_state.get("X_test")

X_val: np.ndarray | None = X_val_raw
X_test: np.ndarray | None = X_test_raw

# y arrays
_y_tr = st.session_state.get("y_train")
_y_v  = st.session_state.get("y_val")
_y_te = st.session_state.get("y_test")
y_train = np.asarray(_y_tr) if _y_tr is not None else None
y_val   = np.asarray(_y_v)  if _y_v  is not None else None
y_test  = np.asarray(_y_te) if _y_te is not None else None

# Adapters (created in Training page)
result_adapters: dict = st.session_state.get("training_adapters", {})

_colors = {r.model_type: COLORS[i % len(COLORS)] for i, r in enumerate(results)}

# ── Pre-compute test-set predictions (supervised only) ─────────────────────────
test_preds: dict = {}
if supervised and X_test is not None and y_test is not None:
    for r in results:
        adapter = result_adapters.get(r.model_type)
        if adapter is not None:
            y_pred = adapter.predict(X_test)
            y_proba = adapter.predict_proba(X_test)
        else:
            y_pred = r.model.predict(X_test)
            y_proba = None
            if hasattr(r.model, "predict_proba"):
                try:
                    y_proba = r.model.predict_proba(X_test)
                except Exception:
                    pass
        test_preds[r.model_type] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": compute_metrics(y_test, y_pred, task_type, y_proba),
        }


# ── SHAP helper ────────────────────────────────────────────────────────────────
def _render_shap_tab(r: "TrainingResult", X_data: np.ndarray, feat_names: list[str]) -> None:
    """Render SHAP summary and beeswarm plots for a single model result.

    Only attempts SHAP for tree-based model types because KernelExplainer
    is prohibitively slow for interactive use.
    """
    if r.model_type not in _TREE_BASED_TYPES:
        st.info(
            f"SHAP analysis is only available for tree-based models "
            f"(random_forest, xgboost, decision_tree, gradient_boosting). "
            f"**{r.model_name}** uses `{r.model_type}`, which is not supported."
        )
        return

    try:
        import shap  # noqa: F401
    except ImportError:
        st.info("SHAP is not installed. Run `pip install shap` to enable SHAP analysis.")
        return

    # Resolve the raw sklearn/xgb estimator from the adapter or result
    adapter = result_adapters.get(r.model_type)
    if adapter is not None:
        # adapter.model holds the fitted estimator directly
        raw_model = adapter.model
    else:
        raw_model = r.model

    # Limit data to 200 rows for performance
    n_shap = min(200, X_data.shape[0])
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(X_data.shape[0], size=n_shap, replace=False)
    X_sample = X_data[idx_sample]

    shap_values = None
    explainer_type = None

    with st.spinner(f"Computing SHAP values for {r.model_name}..."):
        try:
            import shap as shap_lib
            # TreeExplainer — fastest, works for sklearn trees + xgboost
            explainer = shap_lib.TreeExplainer(raw_model)
            sv = explainer.shap_values(X_sample)
            # For multi-output (e.g. multiclass RF), sv is a list — take class 1 or average
            if isinstance(sv, list):
                if len(sv) == 2:
                    shap_values = sv[1]
                else:
                    shap_values = np.mean(np.abs(np.array(sv)), axis=0)
            else:
                shap_values = sv
            explainer_type = "TreeExplainer"
        except Exception as tree_err:
            try:
                import shap as shap_lib
                # LinearExplainer fallback
                explainer = shap_lib.LinearExplainer(raw_model, X_sample)
                sv = explainer.shap_values(X_sample)
                shap_values = sv if not isinstance(sv, list) else sv[1]
                explainer_type = "LinearExplainer"
            except Exception:
                st.info(
                    f"Could not compute SHAP values for **{r.model_name}**: {tree_err}. "
                    "The model may not be supported by TreeExplainer or LinearExplainer."
                )
                return

    if shap_values is None:
        st.info(f"SHAP values could not be computed for {r.model_name}.")
        return

    st.caption(f"SHAP analysis via **{explainer_type}** — sample of {n_shap} rows.")

    n_feat = X_sample.shape[1]
    fn = feat_names if feat_names else [f"feature_{i}" for i in range(n_feat)]
    fn_raw = fn[:n_feat] if len(fn) >= n_feat else fn + [f"feature_{i}" for i in range(len(fn), n_feat)]
    fn_arr = np.array([str(x) for x in fn_raw])

    # Mean |SHAP| per feature — bar chart (top 15)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top15 = np.argsort(mean_abs_shap)[-15:][::-1]

    fig_bar = go.Figure(go.Bar(
        x=mean_abs_shap[top15],
        y=fn_arr[top15],
        orientation="h",
        marker_color=_colors.get(r.model_type, COLORS[0]),
    ))
    fig_bar.update_layout(
        title=f"SHAP Feature Importance (mean |SHAP|) — {r.model_name}",
        xaxis_title="Mean |SHAP value|",
        yaxis=dict(autorange="reversed"),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bar, width='stretch')

    # Beeswarm-style dot plot — top 15 features
    st.markdown("**Beeswarm dot plot** (x = SHAP value, color = feature value)")
    top15_feats = top15  # already sorted by importance, descending

    # Normalise feature values to [0, 1] per feature for coloring
    X_norm = X_sample[:, top15_feats].astype(float)
    col_mins = X_norm.min(axis=0, keepdims=True)
    col_maxs = X_norm.max(axis=0, keepdims=True)
    denom = np.where(col_maxs - col_mins == 0, 1.0, col_maxs - col_mins)
    X_norm = (X_norm - col_mins) / denom  # shape: (n_shap, 15)

    fig_bee = go.Figure()
    for fi, feat_idx in enumerate(top15_feats):
        feat_label = fn_arr[feat_idx]
        sv_col = shap_values[:, feat_idx]
        fv_col = X_norm[:, fi]

        # Add jitter on y-axis to avoid overplotting
        y_jitter = fi + np.random.RandomState(feat_idx).uniform(-0.35, 0.35, size=len(sv_col))

        fig_bee.add_trace(go.Scatter(
            x=sv_col,
            y=y_jitter,
            mode="markers",
            name=feat_label,
            marker=dict(
                color=fv_col,
                colorscale=[[0, "#4575b4"], [0.5, "#ffffbf"], [1, "#d73027"]],
                size=4,
                opacity=0.7,
                showscale=(fi == 0),
                colorbar=dict(
                    title="Feature value<br>(blue=low, red=high)",
                    thickness=12,
                    len=0.5,
                ) if fi == 0 else None,
            ),
            showlegend=False,
        ))

    fig_bee.add_vline(x=0, line_color="gray", line_dash="dash", line_width=1)
    fig_bee.update_layout(
        title=f"SHAP Beeswarm — {r.model_name} (top 15 features)",
        xaxis_title="SHAP value",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top15_feats))),
            ticktext=[fn_arr[fi] for fi in top15_feats],
        ),
        height=max(400, len(top15_feats) * 28),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bee, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch to task-specific sections
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION (binary + multiclass)
# ─────────────────────────────────────────────────────────────────────────────
if task_type in ("binary_classification", "multiclass_classification"):
    # ── Metrics Summary ──────────────────────────────────────────────────────
    st.header("Metrics Summary")

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    col_labels = {
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "AUC-ROC", "mcc": "MCC",
    }

    def _build_metrics_df(source: str) -> pd.DataFrame:
        rows = []
        for r in results:
            m = r.metrics if source == "val" else test_preds.get(r.model_type, {}).get("metrics", {})
            row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
            for mc in metric_cols:
                v = m.get(mc)
                row[col_labels[mc]] = f"{v:.4f}" if v is not None else "—"
            if r.cv_mean is not None:
                row["CV Mean ± Std"] = f"{r.cv_mean:.4f} ± {r.cv_std:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def _highlight_best(df: pd.DataFrame):
        best = summary["best_model"]
        def _apply(row):
            if row["Model"] == best:
                return ["background-color: rgba(78,205,196,0.15)"] * len(row)
            return [""] * len(row)
        return df.style.apply(_apply, axis=1)

    tab_val, tab_test = st.tabs(["Validation Set", "Test Set"])
    with tab_val:
        st.caption("Metrics on the validation set during training.")
        st.dataframe(_build_metrics_df("val"), width='stretch', hide_index=True)
    with tab_test:
        if test_preds:
            st.caption("Final evaluation on the held-out test set.")
            st.dataframe(_highlight_best(_build_metrics_df("test")), width='stretch', hide_index=True)
        else:
            st.info("No test predictions available.")

    # Best model callout
    best_result = next((r for r in results if r.model_name == summary["best_model"]), results[0])
    primary = summary["primary_metric"]
    if test_preds:
        best_test_score = test_preds[best_result.model_type]["metrics"].get(primary, 0)
        st.success(f"**Best model (test set):** {summary['best_model']} — {primary}: {best_test_score:.4f}")

    # ── Visualizations ───────────────────────────────────────────────────────
    st.header("Visualizations")
    if test_preds and y_test is not None:
        tab_cm, tab_roc, tab_pr, tab_fi, tab_lc, tab_shap = st.tabs(
            ["Confusion Matrix", "ROC Curve", "Precision-Recall", "Feature Importance", "Learning Curves", "SHAP Analysis"]
        )
        classes = np.unique(y_test)

        # Confusion Matrix
        with tab_cm:
            n_cols = min(len(results), 3)
            cols_cm = st.columns(n_cols)
            for idx, r in enumerate(results):
                y_pred = test_preds[r.model_type]["y_pred"]
                cm = confusion_matrix(y_test, y_pred, labels=classes)
                fig = go.Figure(go.Heatmap(
                    z=cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                    colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True,
                ))
                fig.update_layout(title=r.model_name, xaxis_title="Predicted",
                                  yaxis_title="Actual", **PLOTLY_LAYOUT)
                with cols_cm[idx % n_cols]:
                    st.plotly_chart(fig, width='stretch')

        # ROC Curve
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
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{r.model_name} (AUC={roc_auc:.4f})",
                            line=dict(color=_colors[r.model_type], width=2),
                        ))
                        any_roc = True
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random Classifier", line=dict(color="gray", dash="dash"),
                ))
                fig.update_layout(title="ROC Curves", xaxis_title="FPR",
                                  yaxis_title="TPR", **PLOTLY_LAYOUT)
                if any_roc:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("ROC curve requires probability estimates.")
            else:
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
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"Class {cls} (AUC={auc(fpr, tpr):.3f})",
                        ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Baseline", line=dict(color="gray", dash="dash"),
                    ))
                    fig.update_layout(title=f"ROC (OVR) — {r.model_name}",
                                      xaxis_title="FPR", yaxis_title="TPR", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width='stretch')

        # Precision-Recall Curve (binary only)
        with tab_pr:
            if n_classes == 2:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                fig = go.Figure()
                for r in results:
                    tp = test_preds[r.model_type]
                    if tp["y_proba"] is not None:
                        prec, rec, _ = precision_recall_curve(y_test, tp["y_proba"][:, 1])
                        ap = average_precision_score(y_test, tp["y_proba"][:, 1])
                        fig.add_trace(go.Scatter(
                            x=rec, y=prec, mode="lines",
                            name=f"{r.model_name} (AP={ap:.4f})",
                            line=dict(color=_colors[r.model_type], width=2),
                        ))
                fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                                  yaxis_title="Precision", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Precision-Recall curve is shown for binary classification only.")

        # Feature Importance
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
                fig = go.Figure(go.Bar(
                    x=importances[top_idx], y=top_names, orientation="h",
                    marker_color=_colors[r.model_type],
                ))
                fig.update_layout(title=f"Feature Importance — {r.model_name}",
                                  xaxis_title="Importance",
                                  yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
            if not has_any:
                st.info("Feature importance not available for the trained model(s).")

        # Learning Curves
        with tab_lc:
            st.caption("Learning curves — requires multiple re-fits.")
            if y_val is not None and st.button("Compute Learning Curves", type="secondary"):
                from sklearn.base import clone
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.concatenate([y_train, y_val])
                for r in results:
                    with st.spinner(f"Computing for {r.model_name}..."):
                        try:
                            train_sizes, tr_sc, val_sc = learning_curve(
                                clone(r.model), X_combined, y_combined,
                                cv=3, scoring="accuracy",
                                train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
                            )
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=train_sizes, y=tr_sc.mean(axis=1),
                                mode="lines+markers", name="Train",
                                line=dict(color=_colors[r.model_type]),
                                error_y=dict(type="data", array=tr_sc.std(axis=1), visible=True),
                            ))
                            fig.add_trace(go.Scatter(
                                x=train_sizes, y=val_sc.mean(axis=1),
                                mode="lines+markers", name="Cross-val",
                                line=dict(color=_colors[r.model_type], dash="dash"),
                                error_y=dict(type="data", array=val_sc.std(axis=1), visible=True),
                            ))
                            fig.update_layout(title=f"Learning Curve — {r.model_name}",
                                              xaxis_title="Training Examples",
                                              yaxis_title="Accuracy", **PLOTLY_LAYOUT)
                            st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.warning(f"Could not compute curve for {r.model_name}: {e}")

        # SHAP Analysis
        with tab_shap:
            st.caption(
                "SHAP (SHapley Additive exPlanations) explains model predictions. "
                "Only tree-based models are supported for interactive performance."
            )
            n_features_shap = X_train.shape[1]
            feat_names_shap = feature_names if feature_names else [f"feature_{i}" for i in range(n_features_shap)]
            # Use test data for SHAP if available, otherwise training data
            X_shap_source = X_test if X_test is not None else X_train
            for r in results:
                st.markdown(f"---\n#### {r.model_name}")
                _render_shap_tab(r, X_shap_source, feat_names_shap)

# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
elif task_type == "regression":
    st.header("Metrics Summary")

    metric_cols = ["mse", "rmse", "mae", "r2"]
    col_labels = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}

    def _build_reg_df(source: str) -> pd.DataFrame:
        rows = []
        for r in results:
            m = r.metrics if source == "val" else test_preds.get(r.model_type, {}).get("metrics", {})
            row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
            for mc in metric_cols:
                v = m.get(mc)
                row[col_labels[mc]] = f"{v:.4f}" if v is not None else "—"
            if r.cv_mean is not None:
                row["CV Mean ± Std"] = f"{r.cv_mean:.4f} ± {r.cv_std:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    tab_val, tab_test = st.tabs(["Validation Set", "Test Set"])
    with tab_val:
        st.dataframe(_build_reg_df("val"), width='stretch', hide_index=True)
    with tab_test:
        if test_preds:
            st.dataframe(_build_reg_df("test"), width='stretch', hide_index=True)
        else:
            st.info("No test predictions available.")

    best_result = next((r for r in results if r.model_name == summary["best_model"]), results[0])
    primary = summary["primary_metric"]
    if test_preds:
        best_score = test_preds[best_result.model_type]["metrics"].get(primary, 0)
        st.success(f"**Best model (test):** {summary['best_model']} — {primary}: {best_score:.4f}")

    st.header("Visualizations")
    if test_preds and y_test is not None:
        tab_res, tab_avp, tab_fi, tab_lc, tab_shap = st.tabs(
            ["Residual Plot", "Actual vs Predicted", "Feature Importance", "Learning Curves", "SHAP Analysis"]
        )
        with tab_res:
            n_cols = min(len(results), 3)
            cols_res = st.columns(n_cols)
            for idx, r in enumerate(results):
                y_pred = test_preds[r.model_type]["y_pred"]
                residuals = y_test - y_pred
                fig = go.Figure(go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(color=_colors[r.model_type], opacity=0.6, size=5),
                ))
                fig.add_hline(y=0, line_color="red", line_dash="dash")
                fig.update_layout(title=r.model_name, xaxis_title="Predicted",
                                  yaxis_title="Residuals", **PLOTLY_LAYOUT)
                with cols_res[idx % n_cols]:
                    st.plotly_chart(fig, width='stretch')

        with tab_avp:
            n_cols = min(len(results), 3)
            cols_avp = st.columns(n_cols)
            for idx, r in enumerate(results):
                y_pred = test_preds[r.model_type]["y_pred"]
                mn = float(min(y_test.min(), y_pred.min()))
                mx = float(max(y_test.max(), y_pred.max()))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_pred, mode="markers",
                    marker=dict(color=_colors[r.model_type], opacity=0.6, size=5),
                    name="Predictions",
                ))
                fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="red", dash="dash"), name="Perfect fit",
                ))
                fig.update_layout(title=r.model_name, xaxis_title="Actual",
                                  yaxis_title="Predicted", **PLOTLY_LAYOUT)
                with cols_avp[idx % n_cols]:
                    st.plotly_chart(fig, width='stretch')

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
                fig = go.Figure(go.Bar(
                    x=importances[top_idx], y=top_names, orientation="h",
                    marker_color=_colors[r.model_type],
                ))
                fig.update_layout(title=f"Feature Importance — {r.model_name}",
                                  yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
            if not has_any:
                st.info("Feature importance not available for the trained model(s).")

        with tab_lc:
            st.caption("Learning curves — requires multiple re-fits.")
            if y_val is not None and st.button("Compute Learning Curves", type="secondary"):
                from sklearn.base import clone
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.concatenate([y_train, y_val])
                for r in results:
                    with st.spinner(f"Computing for {r.model_name}..."):
                        try:
                            train_sizes, tr_sc, val_sc = learning_curve(
                                clone(r.model), X_combined, y_combined,
                                cv=3, scoring="r2",
                                train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
                            )
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=train_sizes, y=tr_sc.mean(axis=1),
                                mode="lines+markers", name="Train",
                                line=dict(color=_colors[r.model_type]),
                                error_y=dict(type="data", array=tr_sc.std(axis=1), visible=True),
                            ))
                            fig.add_trace(go.Scatter(
                                x=train_sizes, y=val_sc.mean(axis=1),
                                mode="lines+markers", name="Cross-val",
                                line=dict(color=_colors[r.model_type], dash="dash"),
                                error_y=dict(type="data", array=val_sc.std(axis=1), visible=True),
                            ))
                            fig.update_layout(title=f"Learning Curve — {r.model_name}",
                                              xaxis_title="Training Examples",
                                              yaxis_title="R²", **PLOTLY_LAYOUT)
                            st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.warning(f"Could not compute curve for {r.model_name}: {e}")

        # SHAP Analysis
        with tab_shap:
            st.caption(
                "SHAP (SHapley Additive exPlanations) explains model predictions. "
                "Only tree-based models are supported for interactive performance."
            )
            n_features_shap = X_train.shape[1]
            feat_names_shap = feature_names if feature_names else [f"feature_{i}" for i in range(n_features_shap)]
            X_shap_source = X_test if X_test is not None else X_train
            for r in results:
                st.markdown(f"---\n#### {r.model_name}")
                _render_shap_tab(r, X_shap_source, feat_names_shap)

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif task_type == "clustering":
    st.header("Clustering Metrics")

    rows = []
    for r in results:
        row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
        for k in ["silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters"]:
            v = r.metrics.get(k)
            if v is not None:
                row[k.replace("_", " ").title()] = f"{v:.4f}" if isinstance(v, float) else str(int(v))
            else:
                row[k.replace("_", " ").title()] = "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    best_result = next((r for r in results if r.model_name == summary["best_model"]), results[0])
    primary = summary["primary_metric"]
    st.success(
        f"**Best model:** {summary['best_model']} — "
        f"{primary}: {summary['best_score']:.4f}"
    )

    st.header("Visualizations")
    tab_scatter, tab_metrics, tab_dendrogram, tab_elbow = st.tabs(
        ["Cluster Scatter (2D PCA)", "Metric Comparison", "Dendrogram", "Elbow & Silhouette Curves"]
    )

    with tab_scatter:
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is None:
                continue
            try:
                labels = adapter.predict(X_train)
                # 2D PCA projection
                from sklearn.decomposition import PCA as _PCA
                pca_2d = _PCA(n_components=2)
                coords = pca_2d.fit_transform(X_train)
                fig = go.Figure()
                unique_labels = np.unique(labels)
                for i, lbl in enumerate(unique_labels):
                    mask = labels == lbl
                    color = COLORS[int(lbl) % len(COLORS)] if lbl >= 0 else "#888888"
                    name = f"Cluster {lbl}" if lbl >= 0 else "Noise (DBSCAN)"
                    fig.add_trace(go.Scatter(
                        x=coords[mask, 0], y=coords[mask, 1],
                        mode="markers", name=name,
                        marker=dict(color=color, size=5, opacity=0.7),
                    ))

                # ── Centroid overlays ──────────────────────────────────────
                # Attempt to get centroids from the model, then project via the same PCA
                clustering_model = adapter.model  # ClusteringAdapter or raw estimator
                inner_model = getattr(clustering_model, "_inner", None)
                centroids_raw = None

                # KMeans: cluster_centers_ on the inner sklearn KMeans
                if inner_model is not None and hasattr(inner_model, "cluster_centers_"):
                    centroids_raw = inner_model.cluster_centers_
                elif hasattr(clustering_model, "cluster_centers_"):
                    centroids_raw = clustering_model.cluster_centers_
                else:
                    # For all other algorithms: compute centroid as mean per cluster
                    valid_labels = unique_labels[unique_labels >= 0]
                    if len(valid_labels) > 0:
                        centroids_raw = np.array(
                            [X_train[labels == lbl].mean(axis=0) for lbl in valid_labels]
                        )
                        # Map cluster labels to their centroid indices
                        centroid_labels = valid_labels
                    else:
                        centroids_raw = None

                if centroids_raw is not None:
                    # Determine which cluster labels correspond to these centroids
                    if inner_model is not None and hasattr(inner_model, "cluster_centers_"):
                        centroid_labels = np.arange(len(centroids_raw))
                    elif hasattr(clustering_model, "cluster_centers_"):
                        centroid_labels = np.arange(len(centroids_raw))
                    else:
                        # already set above in the else branch
                        pass

                    try:
                        centroids_2d = pca_2d.transform(centroids_raw)
                        for ci, lbl in enumerate(centroid_labels):
                            color = COLORS[int(lbl) % len(COLORS)] if lbl >= 0 else "#888888"
                            fig.add_trace(go.Scatter(
                                x=[centroids_2d[ci, 0]],
                                y=[centroids_2d[ci, 1]],
                                mode="markers",
                                name=f"Centroid {lbl}",
                                marker=dict(
                                    color=color,
                                    size=15,
                                    symbol="star",
                                    line=dict(color="white", width=1),
                                ),
                                showlegend=True,
                            ))
                    except Exception:
                        pass  # silently skip centroid overlay if transform fails

                fig.update_layout(
                    title=f"Cluster Scatter — {r.model_name} (PCA 2D)",
                    xaxis_title="PC1", yaxis_title="PC2",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Could not render scatter for {r.model_name}: {e}")

    with tab_metrics:
        metric_names = ["silhouette", "calinski_harabasz"]
        for metric in metric_names:
            vals = [(r.model_name, r.metrics.get(metric, 0)) for r in results]
            vals = [(n, v) for n, v in vals if v is not None]
            if vals:
                names, values = zip(*vals)
                fig = go.Figure(go.Bar(x=list(names), y=list(values),
                                       marker_color=COLORS[:len(names)]))
                fig.update_layout(title=f"{metric.replace('_', ' ').title()} by Model",
                                  **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')

    # ── Dendrogram tab ─────────────────────────────────────────────────────
    with tab_dendrogram:
        st.caption(
            "Dendrogram visualization is only available for **Agglomerative Clustering** models. "
            "Limited to a random subsample of 100 rows for performance."
        )

        has_agglomerative = False
        for r in results:
            if "agglomerative" not in r.model_type.lower():
                continue

            has_agglomerative = True
            adapter = result_adapters.get(r.model_type)

            try:
                from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
            except ImportError:
                st.info("scipy is not installed. Run `pip install scipy` to enable dendrogram.")
                break

            # Subsample X_train to max 100 rows
            n_dendro = min(100, X_train.shape[0])
            rng_d = np.random.RandomState(0)
            idx_d = rng_d.choice(X_train.shape[0], size=n_dendro, replace=False)
            X_dendro = X_train[idx_d]

            # Resolve linkage method from the inner AgglomerativeClustering model
            linkage_method = "ward"
            if adapter is not None:
                clustering_adapter = adapter.model  # ClusteringAdapter
                inner_agg = getattr(clustering_adapter, "_inner", None)
                if inner_agg is not None:
                    linkage_method = getattr(inner_agg, "linkage", "ward")
                elif hasattr(clustering_adapter, "linkage"):
                    linkage_method = clustering_adapter.linkage
            elif hasattr(r.model, "linkage"):
                linkage_method = r.model.linkage

            with st.spinner(f"Computing dendrogram for {r.model_name}..."):
                try:
                    Z = linkage(X_dendro, method=linkage_method)
                    ddata = scipy_dendrogram(Z, no_plot=True)

                    # Build Plotly dendrogram from icoord/dcoord
                    icoord = ddata["icoord"]  # list of [x0, x1, x2, x3] for each merge
                    dcoord = ddata["dcoord"]  # list of [y0, y1, y2, y3] for each merge

                    # Color lines by height level (normalised to [0, 1])
                    all_heights = [max(dc) for dc in dcoord]
                    max_h = max(all_heights) if all_heights else 1.0
                    min_h = min(all_heights) if all_heights else 0.0
                    h_range = max_h - min_h if max_h != min_h else 1.0

                    fig_dend = go.Figure()
                    n_merges = len(icoord)
                    for mi in range(n_merges):
                        xs = icoord[mi]
                        ys = dcoord[mi]
                        norm_h = (max(ys) - min_h) / h_range
                        # Interpolate color: low=blue, mid=green, high=red
                        r_c = int(norm_h * 220)
                        g_c = int((1 - abs(norm_h - 0.5) * 2) * 180)
                        b_c = int((1 - norm_h) * 220)
                        line_color = f"rgb({r_c},{g_c},{b_c})"
                        fig_dend.add_trace(go.Scatter(
                            x=xs, y=ys,
                            mode="lines",
                            line=dict(color=line_color, width=1.5),
                            showlegend=False,
                            hoverinfo="skip",
                        ))

                    fig_dend.update_layout(
                        title=f"Dendrogram — {r.model_name} (linkage: {linkage_method}, n={n_dendro})",
                        xaxis=dict(showticklabels=False, title="Samples"),
                        yaxis_title="Merge distance",
                        height=500,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_dend, width='stretch')
                    st.caption(
                        f"Linkage method: **{linkage_method}**. "
                        f"Showing {n_dendro} randomly sampled data points."
                    )
                except Exception as e:
                    st.warning(f"Could not render dendrogram for {r.model_name}: {e}")

        if not has_agglomerative:
            st.info(
                "No Agglomerative Clustering model found in the current results. "
                "Train an Agglomerative model to see its dendrogram here."
            )

    # ── Elbow & Silhouette Curves tab ──────────────────────────────────────
    with tab_elbow:
        st.caption(
            "Fits **KMeans** for K in [2, 10] to compute inertia and silhouette scores. "
            "This helps choose the right number of clusters regardless of which model was trained."
        )
        st.info(
            "Uses KMeans to find optimal K. "
            "Check the scatter tab to see your trained model's clusters."
        )

        # Subsample for speed
        n_elbow_max = min(5000, X_train.shape[0])
        rng_e = np.random.RandomState(1)
        idx_e = rng_e.choice(X_train.shape[0], size=n_elbow_max, replace=False)
        X_elbow = X_train[idx_e]

        if st.button("Compute Elbow & Silhouette", type="secondary", key="elbow_btn"):
            from sklearn.cluster import KMeans as _KMeans
            from sklearn.metrics import silhouette_score as _silhouette_score

            k_range = list(range(2, 11))
            inertias = []
            silhouettes = []

            with st.spinner("Fitting KMeans for K = 2 to 10..."):
                for k in k_range:
                    km = _KMeans(n_clusters=k, random_state=42, n_init="auto")
                    km.fit(X_elbow)
                    inertias.append(km.inertia_)
                    sil = _silhouette_score(X_elbow, km.labels_)
                    silhouettes.append(sil)

            # Detect elbow via max second derivative of inertia
            if len(inertias) >= 3:
                inertia_arr = np.array(inertias)
                second_deriv = np.diff(np.diff(inertia_arr))
                elbow_idx = int(np.argmax(second_deriv)) + 1  # offset by 2 (two diffs)
                best_k_elbow = k_range[elbow_idx]
            else:
                best_k_elbow = k_range[0]

            best_k_silhouette = k_range[int(np.argmax(silhouettes))]

            col_e1, col_e2 = st.columns(2)

            with col_e1:
                fig_inertia = go.Figure()
                fig_inertia.add_trace(go.Scatter(
                    x=k_range, y=inertias,
                    mode="lines+markers",
                    name="Inertia",
                    line=dict(color=COLORS[0], width=2),
                    marker=dict(size=8),
                ))
                fig_inertia.add_vline(
                    x=best_k_elbow,
                    line_color=COLORS[2], line_dash="dash",
                    annotation_text=f"Elbow K={best_k_elbow}",
                    annotation_position="top right",
                )
                fig_inertia.update_layout(
                    title="Elbow Plot (Inertia vs K)",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Inertia (within-cluster sum of squares)",
                    xaxis=dict(tickmode="linear", dtick=1),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_inertia, width='stretch')

            with col_e2:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(
                    x=k_range, y=silhouettes,
                    mode="lines+markers",
                    name="Silhouette Score",
                    line=dict(color=COLORS[1], width=2),
                    marker=dict(size=8),
                ))
                fig_sil.add_vline(
                    x=best_k_silhouette,
                    line_color=COLORS[3], line_dash="dash",
                    annotation_text=f"Best K={best_k_silhouette}",
                    annotation_position="top right",
                )
                fig_sil.update_layout(
                    title="Silhouette Score vs K",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Silhouette Score",
                    xaxis=dict(tickmode="linear", dtick=1),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_sil, width='stretch')

            st.caption(
                f"Best K by silhouette: **{best_k_silhouette}** "
                f"(score: {silhouettes[best_k_silhouette - 2]:.4f})  |  "
                f"Elbow K: **{best_k_elbow}**  |  "
                f"Sample size used: {n_elbow_max} rows"
            )

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif task_type == "anomaly_detection":
    st.header("Anomaly Detection Metrics")

    rows = []
    for r in results:
        row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
        for k in ["n_anomalies", "anomaly_ratio"]:
            v = r.metrics.get(k)
            if v is not None:
                row[k.replace("_", " ").title()] = f"{v:.4f}" if isinstance(v, float) else str(int(v))
            else:
                row[k.replace("_", " ").title()] = "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.success(
        f"**Best model:** {summary['best_model']} — "
        f"{summary['primary_metric']}: {summary['best_score']:.4f}"
    )

    st.header("Visualizations")
    tab_hist, tab_table = st.tabs(["Anomaly Score Histogram", "Top Anomalies"])

    with tab_hist:
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is None:
                continue
            try:
                scores = adapter.decision_scores(X_train)
                if scores is not None:
                    preds = adapter.predict(X_train)
                    colors_arr = ["#FF6B6B" if p == 1 else "#4ECDC4" for p in preds]
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=scores,
                        nbinsx=50,
                        name="Anomaly Scores",
                        marker_color="#4ECDC4",
                        opacity=0.7,
                    ))
                    fig.update_layout(
                        title=f"Anomaly Score Distribution — {r.model_name}",
                        xaxis_title="Anomaly Score (higher = more anomalous)",
                        yaxis_title="Count",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, width='stretch')
                    n_anomalies = int(preds.sum())
                    st.caption(
                        f"{r.model_name}: {n_anomalies} anomalies detected "
                        f"({100 * n_anomalies / len(preds):.1f}%)"
                    )
            except Exception as e:
                st.warning(f"Could not render histogram for {r.model_name}: {e}")

    with tab_table:
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is None:
                continue
            try:
                scores = adapter.decision_scores(X_train)
                if scores is not None:
                    top_n = min(20, len(scores))
                    top_idx = np.argsort(scores)[-top_n:][::-1]
                    fn = feature_names if feature_names else [f"f{i}" for i in range(X_train.shape[1])]
                    df_top = pd.DataFrame(X_train[top_idx], columns=fn)
                    df_top.insert(0, "Anomaly Score", scores[top_idx].round(4))
                    df_top.insert(0, "Row Index", top_idx)
                    st.write(f"**{r.model_name} — Top {top_n} Anomalies:**")
                    st.dataframe(df_top, width='stretch', hide_index=True)
            except Exception as e:
                st.warning(f"Could not render table for {r.model_name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
elif task_type == "dimensionality_reduction":
    st.header("Dimensionality Reduction Metrics")

    rows = []
    for r in results:
        row: dict = {"Model": r.model_name, "Time (s)": f"{r.training_time:.2f}"}
        for k in ["explained_variance", "n_components"]:
            v = r.metrics.get(k)
            if v is not None:
                row[k.replace("_", " ").title()] = f"{v:.4f}" if isinstance(v, float) else str(int(v))
            else:
                row[k.replace("_", " ").title()] = "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.success(
        f"**Best model:** {summary['best_model']} — "
        f"{summary['primary_metric']}: {summary['best_score']:.4f}"
    )

    st.header("Visualizations")
    tab_scatter2d, tab_scatter3d, tab_scree = st.tabs(["2D Scatter", "3D Scatter", "Scree Plot"])

    with tab_scatter2d:
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is None:
                continue
            try:
                coords = adapter.transform(X_train)
                if coords.shape[1] >= 2:
                    fig = go.Figure(go.Scatter(
                        x=coords[:, 0], y=coords[:, 1], mode="markers",
                        marker=dict(color=COLORS[0], size=5, opacity=0.7),
                        name=r.model_name,
                    ))
                    fig.update_layout(
                        title=f"2D Projection — {r.model_name}",
                        xaxis_title="Component 1", yaxis_title="Component 2",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, width='stretch')
            except NotImplementedError:
                st.info(f"t-SNE: 2D projection shown from training fit. New data projection not supported.")
                # Try stored reduction output
                stored = st.session_state.get("reduction_output")
                if stored is not None and stored.shape[1] >= 2:
                    fig = go.Figure(go.Scatter(
                        x=stored[:, 0], y=stored[:, 1], mode="markers",
                        marker=dict(color=COLORS[0], size=5, opacity=0.7),
                    ))
                    fig.update_layout(title=f"2D Projection — {r.model_name} (training data)",
                                      xaxis_title="Dim 1", yaxis_title="Dim 2", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Could not render scatter for {r.model_name}: {e}")

    with tab_scatter3d:
        for r in results:
            adapter = result_adapters.get(r.model_type)
            if adapter is None:
                continue
            try:
                coords = adapter.transform(X_train)
                if coords.shape[1] >= 3:
                    fig = go.Figure(go.Scatter3d(
                        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                        mode="markers",
                        marker=dict(color=COLORS[1], size=3, opacity=0.7),
                    ))
                    fig.update_layout(
                        title=f"3D Projection — {r.model_name}",
                        scene=dict(xaxis_title="C1", yaxis_title="C2", zaxis_title="C3"),
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info(f"{r.model_name}: only 2 components — 3D plot unavailable.")
            except (NotImplementedError, Exception) as e:
                st.info(f"3D scatter not available for {r.model_name}: {e}")

    with tab_scree:
        for r in results:
            model = r.model
            evr = getattr(model, "explained_variance_ratio_", None)
            if evr is None:
                # Try inner model
                inner = getattr(model, "_inner", None) or getattr(model, "model", None)
                if inner is not None:
                    evr = getattr(inner, "explained_variance_ratio_", None)
            if evr is not None:
                fig = go.Figure()
                components = list(range(1, len(evr) + 1))
                fig.add_trace(go.Bar(
                    x=components, y=evr,
                    name="Individual variance", marker_color=COLORS[0],
                ))
                fig.add_trace(go.Scatter(
                    x=components, y=np.cumsum(evr), mode="lines+markers",
                    name="Cumulative", line=dict(color=COLORS[1]),
                ))
                fig.update_layout(
                    title=f"Scree Plot — {r.model_name}",
                    xaxis_title="Component", yaxis_title="Explained Variance Ratio",
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info(f"Explained variance ratio not available for {r.model_name} ({r.model_type}).")

else:
    st.warning(f"Unknown task type: `{task_type}`")

# ══════════════════════════════════════════════════════════════════════════════
# MLflow Experiment History (all task types)
# ══════════════════════════════════════════════════════════════════════════════
st.header("MLflow Experiment History")


def _load_mlflow_runs() -> pd.DataFrame | None:
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
            end   = run.info.end_time
            start = run.info.start_time
            duration = f"{(end - start) / 1000:.1f}" if end else "—"
            row: dict = {
                "Run ID":      run.info.run_id[:8],
                "Run Name":    run.info.run_name or "—",
                "Status":      run.info.status,
                "Task Type":   run.data.tags.get("task_type", "—"),
                "Start Time":  pd.to_datetime(start, unit="ms").strftime("%Y-%m-%d %H:%M"),
                "Duration (s)":duration,
            }
            for metric in ["accuracy", "f1", "roc_auc", "r2", "rmse", "mae",
                           "silhouette", "anomaly_ratio", "explained_variance", "cv_mean"]:
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
        filtered = filtered[filtered["Run Name"].str.contains(name_filter, case=False, na=False)]
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    st.dataframe(filtered, width='stretch', hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(mlflow_df)} runs.")

# ══════════════════════════════════════════════════════════════════════════════
# Promote Model to Registry (supervised only)
# ══════════════════════════════════════════════════════════════════════════════
if supervised:
    st.header("Promote Model to Registry")

    best_result = next((r for r in results if r.model_name == summary["best_model"]), results[0])
    primary = summary["primary_metric"]

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

    if test_preds and selected.model_type in test_preds:
        score = test_preds[selected.model_type]["metrics"].get(primary, 0)
        st.write(f"Promoting **{selected.model_name}** — test {primary}: **{score:.4f}**")

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
                        f"Logged to existing run `{selected.mlflow_run_id[:8]}` "
                        f"and registered as **{registry_name}**."
                    )
                else:
                    with mlflow.start_run(run_name=f"{selected.model_name} (Registry)") as run:
                        mlflow.log_params({k: str(v) for k, v in selected.params.items()})
                        mlflow.log_metrics({k: float(v) for k, v in selected.metrics.items() if isinstance(v, (int, float))})
                        mlflow.sklearn.log_model(
                            selected.model, "model",
                            registered_model_name=registry_name,
                        )
                    msg = (
                        f"Logged in new run `{run.info.run_id[:8]}` "
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
    st.page_link("pages/4_Training.py", label="← Back to Training")
with nav2:
    st.page_link("pages/6_Prediction.py", label="Continue to Prediction →")
