"""Phase 8 — Monitoring & Drift Detection (Streamlit Page 6).

Two independent sections:
  1. Live API / Prometheus metrics  — requires the FastAPI server + Prometheus
  2. Data drift detection           — runs locally; no external services needed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.monitoring.drift import PSI_NONE, PSI_SLIGHT, detect_drift, drift_summary

st.set_page_config(page_title="Monitoring", page_icon="📡", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"

# ── Prometheus helpers ─────────────────────────────────────────────────────────

def _prom_query(promql: str) -> list | None:
    """Instant PromQL query. Returns result list or None on failure."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=3,
        )
        data = r.json()
        if r.status_code == 200 and data.get("status") == "success":
            return data["data"]["result"]
    except Exception:
        pass
    return None


def _scalar(result: list) -> float | None:
    if result and result[0].get("value"):
        return float(result[0]["value"][1])
    return None


def _vector(result: list) -> dict[str, float]:
    """Return {label_value: float} for a vector result."""
    out: dict[str, float] = {}
    for item in result:
        val = item.get("value", [None, None])[1]
        labels = item.get("metric", {})
        # Use first non-empty label value as key
        key = next((v for v in labels.values() if v), str(labels))
        if val is not None:
            out[key] = float(val)
    return out


def _prom_available() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Status")
    if "training_results" in st.session_state:
        s = st.session_state["training_summary"]
        st.success(f"{s['models_trained']} model(s) trained")
        st.write(f"**Best:** {s['best_model']}")
    else:
        st.info("No models trained yet.")

    st.divider()
    st.caption("**External services**")
    health = _api_health()
    st.write(f"API: {'🟢 up' if health else '🔴 down'}")
    prom_up = _prom_available()
    st.write(f"Prometheus: {'🟢 up' if prom_up else '🔴 down'}")
    st.write(f"[Grafana dashboard →]({GRAFANA_URL})")

# ── Page title ─────────────────────────────────────────────────────────────────
st.title("📡 Monitoring & Drift Detection")

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — API Health
# ══════════════════════════════════════════════════════════════════════════════
st.header("API Health")

if health:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", health.get("status", "—").upper())
    c2.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
    c3.metric("Pipeline loaded", "Yes" if health.get("pipeline_loaded") else "No")
    c4.metric("Task type", health.get("task_type") or "—")
else:
    st.warning(
        f"FastAPI server not reachable at `{API_URL}`. "
        "Start it with: `uvicorn api.main:app --port 8000`"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Prometheus Live Metrics
# ══════════════════════════════════════════════════════════════════════════════
st.header("Live Prometheus Metrics")

if not prom_up:
    st.warning(
        f"Prometheus not reachable at `{PROMETHEUS_URL}`. "
        "Start it alongside the API, or use Docker Compose (Phase 10). "
        "Drift detection below works without Prometheus."
    )
else:
    # ── Top-level counters ────────────────────────────────────────────────────
    total_preds = _scalar(_prom_query("sum(api_predictions_total)") or [])
    total_reqs = _scalar(_prom_query("sum(api_requests_total)") or [])
    req_rate = _scalar(
        _prom_query(
            'sum(rate(api_requests_total{status="200"}[5m]))'
        ) or []
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Predictions", f"{int(total_preds):,}" if total_preds else "—")
    m2.metric("Total Requests", f"{int(total_reqs):,}" if total_reqs else "—")
    m3.metric("Request Rate (5 m)", f"{req_rate:.3f} req/s" if req_rate else "—")

    st.divider()

    col_lat, col_cls = st.columns(2)

    # ── Latency percentiles ────────────────────────────────────────────────────
    with col_lat:
        st.subheader("Request Latency")
        base = "sum(rate(api_request_latency_seconds_bucket[5m])) by (le)"
        p50 = _scalar(_prom_query(f"histogram_quantile(0.50, {base})") or [])
        p95 = _scalar(_prom_query(f"histogram_quantile(0.95, {base})") or [])
        p99 = _scalar(_prom_query(f"histogram_quantile(0.99, {base})") or [])

        if any(v is not None for v in [p50, p95, p99]):
            fig = go.Figure(
                go.Bar(
                    x=["p50", "p95", "p99"],
                    y=[
                        round(p50 * 1000, 2) if p50 else 0,
                        round(p95 * 1000, 2) if p95 else 0,
                        round(p99 * 1000, 2) if p99 else 0,
                    ],
                    marker_color=["#4ECDC4", "#45B7D1", "#FF6B6B"],
                    text=[
                        f"{p50*1000:.1f} ms" if p50 else "—",
                        f"{p95*1000:.1f} ms" if p95 else "—",
                        f"{p99*1000:.1f} ms" if p99 else "—",
                    ],
                    textposition="outside",
                )
            )
            fig.update_layout(
                yaxis_title="Latency (ms)", **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data yet — send some requests to the API.")

    # ── Prediction class distribution ──────────────────────────────────────────
    with col_cls:
        st.subheader("Prediction Distribution")
        cls_result = _prom_query("api_prediction_class_total") or []
        cls_dist = _vector(cls_result)

        if cls_dist:
            classes = list(cls_dist.keys())
            counts = list(cls_dist.values())
            fig = go.Figure(
                go.Bar(
                    x=classes,
                    y=counts,
                    marker_color=COLORS[: len(classes)],
                    text=[f"{int(c):,}" for c in counts],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Predicted Class",
                yaxis_title="Count",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction class data yet.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Data Drift Detection
# ══════════════════════════════════════════════════════════════════════════════
st.header("Data Drift Detection")

# ── Reference data ─────────────────────────────────────────────────────────────
target_column: str = st.session_state.get("target_column", "")
column_types: dict = st.session_state.get("column_types", {})
feature_columns: list[str] = []

df_ref: pd.DataFrame | None = None

if "df" in st.session_state and target_column:
    df_orig: pd.DataFrame = st.session_state["df"]
    feature_columns = [c for c in df_orig.columns if c != target_column]
    df_ref = df_orig[feature_columns].copy()
    st.info(
        f"Using training data as reference — "
        f"**{len(df_ref):,} rows × {len(feature_columns)} features**."
    )
else:
    st.caption(
        "No training session found. Upload a reference CSV to use as the baseline."
    )
    ref_upload = st.file_uploader(
        "Reference dataset (training data)", type=["csv"], key="ref_upload"
    )
    if ref_upload:
        df_ref = pd.read_csv(ref_upload)
        if target_column and target_column in df_ref.columns:
            df_ref = df_ref.drop(columns=[target_column])
        feature_columns = list(df_ref.columns)
        # Infer column types from the uploaded reference
        for col in feature_columns:
            if column_types.get(col) is None:
                column_types[col] = (
                    "numerical" if pd.api.types.is_numeric_dtype(df_ref[col])
                    else "categorical"
                )
        st.success(f"Reference loaded: {len(df_ref):,} rows, {len(feature_columns)} columns.")

st.subheader("Upload New / Production Data")
new_upload = st.file_uploader(
    "New data CSV (same feature columns as training)", type=["csv"], key="new_upload"
)

if df_ref is not None and new_upload is not None:
    df_new = pd.read_csv(new_upload)
    if target_column and target_column in df_new.columns:
        df_new = df_new.drop(columns=[target_column])

    st.write(
        f"**New data:** {len(df_new):,} rows × {df_new.shape[1]} columns  |  "
        f"**Reference:** {len(df_ref):,} rows"
    )

    # Only test columns present in both
    common_cols = [c for c in feature_columns if c in df_new.columns]
    missing = [c for c in feature_columns if c not in df_new.columns]
    if missing:
        st.warning(f"Missing columns in new data (will be skipped): `{missing}`")

    if st.button("▶ Run Drift Detection", type="primary"):
        with st.spinner("Running statistical tests…"):
            results = detect_drift(df_ref, df_new, column_types, common_cols)
        st.session_state["drift_results"] = results
        st.session_state["drift_ref"] = df_ref
        st.session_state["drift_new"] = df_new

if "drift_results" in st.session_state:
    results = st.session_state["drift_results"]
    df_ref_saved = st.session_state["drift_ref"]
    df_new_saved = st.session_state["drift_new"]
    summary = drift_summary(results)

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader("Drift Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Columns tested", summary["tested_columns"])
    s2.metric("Drifted columns", summary["drifted_columns"])
    s3.metric(
        "Drift rate",
        f"{summary['drift_rate']:.0%}",
        delta=None,
    )
    s4.metric("Max PSI", f"{summary['max_psi']:.4f}")

    if summary["drifted_columns"] > 0:
        st.error(
            f"⚠ Drift detected in {summary['drifted_columns']} column(s): "
            f"`{'`, `'.join(summary['drifted_column_names'])}`  \n"
            "Consider retraining the model on fresh data."
        )
    else:
        st.success("No significant drift detected. Model inputs appear stable.")

    # ── Results table ──────────────────────────────────────────────────────────
    st.subheader("Per-Column Results")

    rows = []
    for r in results:
        if "error" in r:
            rows.append({"Column": r["column"], "Type": r["type"],
                         "PSI": "—", "KS p-val": "—", "χ² p-val": "—",
                         "Drift": r["error"]})
            continue
        row = {
            "Column": r["column"],
            "Type": r["type"],
            "PSI": f"{r['psi']:.4f}",
            "KS p-val": f"{r['ks_pvalue']:.4f}" if "ks_pvalue" in r else "—",
            "χ² p-val": f"{r['chi2_pvalue']:.4f}" if r.get("chi2_pvalue") is not None else "—",
            "Drift": r["drift_label"],
        }
        rows.append(row)

    table_df = pd.DataFrame(rows)

    def _row_color(row):
        label = row.get("Drift", "")
        if "Significant" in label:
            return ["background-color: rgba(255,107,107,0.15)"] * len(row)
        if "Slight" in label:
            return ["background-color: rgba(255,200,100,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        table_df.style.apply(_row_color, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # PSI colour legend
    st.caption(
        f"PSI < {PSI_NONE}: no drift  |  "
        f"{PSI_NONE} ≤ PSI < {PSI_SLIGHT}: slight drift  |  "
        f"PSI ≥ {PSI_SLIGHT}: significant drift  |  "
        f"KS / χ² significant at α = 0.05"
    )

    # ── Distribution overlays for drifted columns ──────────────────────────────
    drifted_results = [r for r in results if r.get("drifted") and "error" not in r]

    if drifted_results:
        st.subheader("Distribution Comparison — Drifted Columns")
        n_cols = min(len(drifted_results), 2)
        grid = st.columns(n_cols)

        for idx, r in enumerate(drifted_results):
            col = r["column"]
            ctype = r["type"]
            with grid[idx % n_cols]:
                fig = go.Figure()
                if ctype == "numerical":
                    ref_vals = df_ref_saved[col].dropna().values
                    new_vals = df_new_saved[col].dropna().values
                    fig.add_trace(go.Histogram(
                        x=ref_vals, name="Reference",
                        opacity=0.6, marker_color="#4ECDC4",
                        histnorm="probability",
                    ))
                    fig.add_trace(go.Histogram(
                        x=new_vals, name="New data",
                        opacity=0.6, marker_color="#FF6B6B",
                        histnorm="probability",
                    ))
                    fig.update_layout(
                        barmode="overlay",
                        title=f"{col}  (PSI={r['psi']:.3f})",
                        xaxis_title=col,
                        yaxis_title="Density",
                        **PLOTLY_LAYOUT,
                    )
                else:
                    ref_vc = df_ref_saved[col].value_counts(normalize=True)
                    new_vc = df_new_saved[col].value_counts(normalize=True)
                    cats = sorted(set(ref_vc.index) | set(new_vc.index), key=str)
                    fig.add_trace(go.Bar(
                        x=cats, y=[ref_vc.get(c, 0) for c in cats],
                        name="Reference", marker_color="#4ECDC4", opacity=0.8,
                    ))
                    fig.add_trace(go.Bar(
                        x=cats, y=[new_vc.get(c, 0) for c in cats],
                        name="New data", marker_color="#FF6B6B", opacity=0.8,
                    ))
                    fig.update_layout(
                        barmode="group",
                        title=f"{col}  (PSI={r['psi']:.3f})",
                        xaxis_title=col,
                        yaxis_title="Proportion",
                        **PLOTLY_LAYOUT,
                    )
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Automated Retraining
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("4 — Automated Retraining")

import json  # noqa: E402 (deferred to avoid top-level import cost)
import os    # noqa: E402
from src.pipelines.retraining import MODEL_DISPLAY_NAMES, run_retraining  # noqa: E402

# ── Scheduler status banner ────────────────────────────────────────────────
STATUS_FILE = "artifacts/scheduler_status.json"
if os.path.exists(STATUS_FILE):
    try:
        with open(STATUS_FILE) as _sf:
            _sched = json.load(_sf)
        _sched_status = _sched.get("status", "unknown")
        _sched_color = {
            "idle": "✅", "running": "⏳", "error": "🔴",
            "stopped": "⛔", "starting": "🔄",
        }.get(_sched_status, "ℹ️")
        _sched_cols = st.columns([2, 2, 2, 4])
        _sched_cols[0].metric("Scheduler", f"{_sched_color} {_sched_status.upper()}")
        _sched_cols[1].metric(
            "Schedule",
            f"Every {_sched.get('interval_hours', '?')}h"
            if _sched.get("schedule") == "interval"
            else (_sched.get("cron") or "—"),
        )
        _sched_cols[2].metric("Auto-promote", "Yes" if _sched.get("auto_promote") else "No")
        _sched_cols[3].caption(f"Updated: {_sched.get('updated_at', '—')}")
        if _sched_status == "error" and _sched.get("error"):
            st.error(f"Scheduler error: {_sched['error']}")
    except Exception:
        st.info("Scheduler status file could not be read.")
else:
    st.info(
        "No scheduler is currently running. "
        "Start `python retraining_scheduler.py` to enable scheduled retraining."
    )

st.caption(
    "Trigger a manual retraining cycle below, or configure the scheduler for "
    "fully automated retraining whenever drift is detected."
)

# ── Drift alert ────────────────────────────────────────────────────────────
_drift = st.session_state.get("drift_results")
if _drift:
    _summary = drift_summary(_drift)
    if _summary["drift_rate"] > 0:
        st.warning(
            f"⚠️  Drift detected in **{_summary['drifted_columns']}** of "
            f"{_summary['tested_columns']} columns "
            f"({_summary['drift_rate']:.0%} drift rate). "
            "Consider retraining."
        )
    else:
        st.success("No significant drift detected in the latest comparison.")

# ── Retraining controls ────────────────────────────────────────────────────
_model_options = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
_task_type_default = st.session_state.get("task_type", "classification")

with st.form("retrain_form"):
    _rc1, _rc2 = st.columns(2)
    with _rc1:
        _selected_model_name = st.selectbox(
            "Model to retrain",
            options=list(_model_options.keys()),
            help="Select the algorithm to retrain from scratch.",
        )
        _task_type = st.selectbox(
            "Task type",
            ["classification", "regression"],
            index=0 if _task_type_default == "classification" else 1,
        )
    with _rc2:
        _cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        _auto_promote = st.toggle(
            "Auto-promote if improved",
            value=False,
            help="Register the new model in the MLflow Model Registry "
                 "if it outperforms the current one.",
        )
        _registry_name = st.text_input(
            "Registry model name",
            value="automl-model",
            disabled=not _auto_promote,
        )

    _submitted = st.form_submit_button(
        "🔄 Retrain Now",
        type="primary",
        use_container_width=True,
    )

if _submitted:
    _model_type = _model_options[_selected_model_name]
    _log_lines: list[str] = []
    _log_box = st.empty()

    def _ui_log(msg: str) -> None:
        _log_lines.append(msg)
        _log_box.code("\n".join(_log_lines[-30:]))  # keep last 30 lines

    with st.spinner(f"Retraining {_selected_model_name}…"):
        try:
            _retrain_result = run_retraining(
                model_type=_model_type,
                task_type=_task_type,
                cv_folds=_cv_folds,
                registry_name=_registry_name if _auto_promote else None,
                auto_promote=_auto_promote,
                log_callback=_ui_log,
            )
            # Persist in session history
            if "retraining_history" not in st.session_state:
                st.session_state["retraining_history"] = []
            st.session_state["retraining_history"].append(_retrain_result.as_dict())
            st.session_state["last_retraining"] = _retrain_result
        except Exception as _exc:
            st.error(f"Retraining failed: {_exc}")
            _retrain_result = None

    if _retrain_result is not None:
        _pk = _retrain_result.primary_metric
        _new_score = _retrain_result.new_test_metrics.get(_pk, 0.0)
        _old_score = (
            _retrain_result.old_test_metrics.get(_pk, 0.0)
            if _retrain_result.old_test_metrics else None
        )
        if _retrain_result.improved:
            st.success(
                f"✅ Retraining complete — **{_pk}** improved from "
                f"{f'{_old_score:.4f}' if _old_score is not None else 'N/A'} "
                f"→ **{_new_score:.4f}** (Δ {_retrain_result.metric_delta:+.4f})"
                + (" — model promoted to registry." if _retrain_result.promoted else ".")
            )
        else:
            st.warning(
                f"Retraining finished but no improvement — "
                f"{_pk}: new={_new_score:.4f}, "
                f"old={f'{_old_score:.4f}' if _old_score is not None else 'N/A'} "
                f"(Δ {_retrain_result.metric_delta:+.4f}). Existing model kept."
            )

        # Metric comparison cards
        _m1, _m2, _m3 = st.columns(3)
        _m1.metric(
            f"New {_pk} (test)",
            f"{_new_score:.4f}",
            delta=f"{_retrain_result.metric_delta:+.4f}" if _old_score is not None else None,
        )
        _m2.metric(
            f"Previous {_pk}",
            f"{_old_score:.4f}" if _old_score is not None else "—",
        )
        _m3.metric("Promoted", "Yes" if _retrain_result.promoted else "No")

        if _retrain_result.mlflow_run_id:
            st.caption(f"MLflow run: `{_retrain_result.mlflow_run_id}`")
        if _retrain_result.artifact_path:
            st.caption(f"Artifact: `{_retrain_result.artifact_path}`")
        if _retrain_result.promotion_error:
            st.error(f"Promotion error: {_retrain_result.promotion_error}")

# ── Retraining history table ───────────────────────────────────────────────
_history = st.session_state.get("retraining_history", [])
if _history:
    st.subheader("Retraining History")
    _hist_df = pd.DataFrame(list(reversed(_history)))
    st.dataframe(
        _hist_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "improved": st.column_config.CheckboxColumn("Improved"),
            "promoted": st.column_config.CheckboxColumn("Promoted"),
            "new_score": st.column_config.NumberColumn("New Score", format="%.4f"),
            "old_score": st.column_config.NumberColumn("Old Score", format="%.4f"),
            "delta": st.column_config.NumberColumn("Δ", format="%+.4f"),
        },
    )

# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.page_link("pages/5_Prediction.py", label="← Back to Prediction")
