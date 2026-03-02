"""Phase 8 — Monitoring & Drift Detection (Streamlit Page 6).

Two independent sections:
  1. Live API / Prometheus metrics  — requires the FastAPI server + Prometheus
  2. Data drift detection           — runs locally; no external services needed
  3. Concept drift monitoring       — compare prediction distributions over time
  4. Automated retraining           — trigger or schedule retraining
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats

from src.monitoring.drift import (
    PSI_NONE,
    PSI_SLIGHT,
    _psi_numerical,
    detect_drift,
    drift_summary,
)

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
            st.plotly_chart(fig, width='stretch')
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
            st.plotly_chart(fig, width='stretch')
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

if "df" in st.session_state:
    df_orig: pd.DataFrame = st.session_state["df"]
    feature_columns = [c for c in df_orig.columns if c != target_column] if target_column else list(df_orig.columns)
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
        width='stretch',
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
                st.plotly_chart(fig, width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3b — Per-Feature Drift Score Breakdown
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Per-Feature Drift Scores")

    try:
        valid_results = [r for r in results if "error" not in r]

        if valid_results:
            # Sort by PSI descending
            sorted_results = sorted(valid_results, key=lambda r: r["psi"], reverse=True)

            features = [r["column"] for r in sorted_results]
            psi_values = [r["psi"] for r in sorted_results]

            # Color bars by severity
            bar_colors = []
            for psi in psi_values:
                if psi < PSI_NONE:
                    bar_colors.append("#96CEB4")   # green — no drift
                elif psi < PSI_SLIGHT:
                    bar_colors.append("#FFEAA7")   # orange — slight drift
                else:
                    bar_colors.append("#FF6B6B")   # red — significant drift

            fig_psi = go.Figure()

            fig_psi.add_trace(go.Bar(
                y=features,
                x=psi_values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.4f}" for v in psi_values],
                textposition="outside",
                name="PSI",
            ))

            # Reference lines: PSI_NONE (0.10) and PSI_SLIGHT (0.25)
            x_max = max(psi_values + [PSI_SLIGHT * 1.2]) * 1.15

            fig_psi.add_vline(
                x=PSI_NONE,
                line_dash="dash",
                line_color="#FFEAA7",
                annotation_text=f"Slight ({PSI_NONE})",
                annotation_position="top right",
                annotation_font_color="#FFEAA7",
            )
            fig_psi.add_vline(
                x=PSI_SLIGHT,
                line_dash="dash",
                line_color="#FF6B6B",
                annotation_text=f"Significant ({PSI_SLIGHT})",
                annotation_position="top right",
                annotation_font_color="#FF6B6B",
            )

            fig_psi.update_layout(
                title="PSI Scores by Feature",
                xaxis_title="PSI Score",
                xaxis_range=[0, x_max],
                yaxis=dict(autorange="reversed"),
                height=max(300, len(features) * 35 + 100),
                **PLOTLY_LAYOUT,
            )

            st.plotly_chart(fig_psi, width='stretch')

            # ── Detailed breakdown table ───────────────────────────────────────
            st.caption("**Detailed per-feature breakdown:**")

            breakdown_rows = []
            for r in sorted_results:
                psi_val = r["psi"]
                if psi_val < PSI_NONE:
                    status_emoji = "✅"
                elif psi_val < PSI_SLIGHT:
                    status_emoji = "⚠️"
                else:
                    status_emoji = "🔴"

                breakdown_rows.append({
                    "Feature": r["column"],
                    "Type": r["type"],
                    "PSI": round(psi_val, 4),
                    "KS p-val": round(r["ks_pvalue"], 4) if "ks_pvalue" in r else None,
                    "Chi² p-val": round(r["chi2_pvalue"], 4) if r.get("chi2_pvalue") is not None else None,
                    "Status": status_emoji,
                })

            breakdown_df = pd.DataFrame(breakdown_rows)
            st.dataframe(
                breakdown_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "PSI": st.column_config.NumberColumn("PSI", format="%.4f"),
                    "KS p-val": st.column_config.NumberColumn("KS p-val", format="%.4f"),
                    "Chi² p-val": st.column_config.NumberColumn("Chi² p-val", format="%.4f"),
                    "Status": st.column_config.TextColumn("Status"),
                },
            )

        else:
            st.info("No valid drift results to display (all columns had errors).")

    except Exception as _exc:
        st.warning(f"Could not render per-feature drift scores: {_exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4b — Concept Drift Monitoring
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Concept Drift Monitoring")

st.info(
    "Upload prediction logs from different time periods to compare how your model's "
    "output distribution has changed. This helps detect concept drift (the model's "
    "behavior changing over time)."
)

# ── Upload two prediction CSVs ─────────────────────────────────────────────────
cd_col1, cd_col2 = st.columns(2)

with cd_col1:
    concept_ref_file = st.file_uploader(
        "Reference predictions (older)",
        type=["csv"],
        key="concept_ref",
        help="CSV with one column containing numeric predictions, class labels, or anomaly scores.",
    )

with cd_col2:
    concept_new_file = st.file_uploader(
        "Current predictions (newer)",
        type=["csv"],
        key="concept_new",
        help="CSV with one column containing numeric predictions, class labels, or anomaly scores.",
    )

if concept_ref_file is not None and concept_new_file is not None:
    try:
        df_concept_ref = pd.read_csv(concept_ref_file)
        df_concept_new = pd.read_csv(concept_new_file)

        # ── Column selector ────────────────────────────────────────────────────
        common_pred_cols = [
            c for c in df_concept_ref.columns if c in df_concept_new.columns
        ]
        if not common_pred_cols:
            st.warning(
                "No common columns found between the two prediction files. "
                "Make sure both files share the same column name for predictions."
            )
        else:
            # Auto-detect likely prediction column: prefer columns named
            # 'prediction', 'score', 'label', 'anomaly_score', etc.
            _priority_names = [
                "prediction", "predictions", "score", "scores",
                "label", "labels", "anomaly_score", "output",
            ]
            _default_col = next(
                (c for c in _priority_names if c in common_pred_cols),
                common_pred_cols[0],
            )
            _default_idx = common_pred_cols.index(_default_col)

            pred_column = st.selectbox(
                "Select the column containing predictions",
                options=common_pred_cols,
                index=_default_idx,
                key="concept_pred_col",
            )

            ref_preds = df_concept_ref[pred_column].dropna()
            new_preds = df_concept_new[pred_column].dropna()

            # Auto-detect format: numerical vs categorical
            _is_numerical = pd.api.types.is_numeric_dtype(ref_preds)
            _pred_type = "numerical" if _is_numerical else "categorical"
            st.caption(
                f"Detected prediction type: **{_pred_type}** "
                f"({len(ref_preds):,} reference rows, {len(new_preds):,} current rows)"
            )

            if st.button("Compare Distributions", type="primary", key="concept_compare_btn"):
                try:
                    if _is_numerical:
                        ref_arr = ref_preds.values.astype(float)
                        new_arr = new_preds.values.astype(float)

                        # KS test
                        ks_stat, ks_pval = stats.ks_2samp(ref_arr, new_arr)

                        # PSI
                        psi_concept = _psi_numerical(ref_arr, new_arr)

                    else:
                        # Categorical: use chi-squared proxy for KS stat display
                        ref_arr = ref_preds.astype(str)
                        new_arr = new_preds.astype(str)

                        cats = sorted(set(ref_arr.unique()) | set(new_arr.unique()), key=str)
                        ref_counts_c = ref_arr.value_counts()
                        new_counts_c = new_arr.value_counts()
                        ref_freq_c = np.array([ref_counts_c.get(c, 0) for c in cats], dtype=float)
                        new_freq_c = np.array([new_counts_c.get(c, 0) for c in cats], dtype=float)

                        eps = 1e-8
                        ref_freq_norm = (ref_freq_c + eps) / (ref_freq_c.sum() + eps * len(cats))
                        new_freq_norm = (new_freq_c + eps) / (new_freq_c.sum() + eps * len(cats))
                        psi_concept = float(
                            np.sum((new_freq_norm - ref_freq_norm) * np.log(new_freq_norm / ref_freq_norm))
                        )

                        # Chi-squared as approximate significance test
                        expected = ref_freq_c / ref_freq_c.sum() * new_freq_c.sum()
                        mask = expected > 0
                        try:
                            chi2_stat, ks_pval = stats.chisquare(new_freq_c[mask], f_exp=expected[mask])
                            ks_stat = chi2_stat
                        except Exception:
                            ks_stat, ks_pval = 0.0, 1.0

                    # Store in session state
                    st.session_state["concept_drift_result"] = {
                        "ks_stat": ks_stat,
                        "ks_pval": ks_pval,
                        "psi": psi_concept,
                        "ref_preds": ref_preds,
                        "new_preds": new_preds,
                        "pred_type": _pred_type,
                        "pred_column": pred_column,
                    }

                except Exception as _exc:
                    st.warning(f"Error computing concept drift statistics: {_exc}")

    except Exception as _exc:
        st.warning(f"Error reading prediction files: {_exc}")

# ── Display concept drift results ──────────────────────────────────────────────
if "concept_drift_result" in st.session_state:
    _cdr = st.session_state["concept_drift_result"]
    _ks_stat = _cdr["ks_stat"]
    _ks_pval = _cdr["ks_pval"]
    _psi_c = _cdr["psi"]
    _ref_p = _cdr["ref_preds"]
    _new_p = _cdr["new_preds"]
    _pred_type = _cdr["pred_type"]
    _pred_col = _cdr["pred_column"]

    # ── Metrics row ───────────────────────────────────────────────────────────
    _stat_label = "KS Statistic" if _pred_type == "numerical" else "Chi² Statistic"
    _pval_label = "KS p-value" if _pred_type == "numerical" else "Chi² p-value"

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(_stat_label, f"{_ks_stat:.4f}")
    mc2.metric(_pval_label, f"{_ks_pval:.4f}")

    if _psi_c < PSI_NONE:
        _psi_status = "No drift"
        _psi_color_note = "green"
    elif _psi_c < PSI_SLIGHT:
        _psi_status = "Slight drift"
        _psi_color_note = "orange"
    else:
        _psi_status = "Significant drift"
        _psi_color_note = "red"

    mc3.metric("PSI", f"{_psi_c:.4f}", help=f"Status: {_psi_status}")

    # ── Verdict banner ────────────────────────────────────────────────────────
    if _psi_c < PSI_NONE and _ks_pval >= 0.05:
        st.success("No concept drift — prediction distributions are stable.")
    elif _psi_c < PSI_SLIGHT or _ks_pval >= 0.05:
        st.warning(
            "Possible concept drift — the prediction distribution has shifted slightly. "
            "Monitor closely and consider retraining if the trend continues."
        )
    else:
        st.error(
            "Significant concept drift detected — the model's output distribution has "
            "changed substantially. Retraining is strongly recommended."
        )

    # ── Distribution chart ────────────────────────────────────────────────────
    try:
        fig_concept = go.Figure()

        if _pred_type == "numerical":
            fig_concept.add_trace(go.Histogram(
                x=_ref_p.values,
                name="Reference",
                opacity=0.65,
                marker_color="#4ECDC4",
                histnorm="probability",
            ))
            fig_concept.add_trace(go.Histogram(
                x=_new_p.values,
                name="Current",
                opacity=0.65,
                marker_color="#FF6B6B",
                histnorm="probability",
            ))
            fig_concept.update_layout(
                barmode="overlay",
                title=f"Prediction Distribution: {_pred_col}  (PSI={_psi_c:.4f})",
                xaxis_title=_pred_col,
                yaxis_title="Density",
                **PLOTLY_LAYOUT,
            )
        else:
            _ref_vc = _ref_p.astype(str).value_counts(normalize=True)
            _new_vc = _new_p.astype(str).value_counts(normalize=True)
            _cats = sorted(set(_ref_vc.index) | set(_new_vc.index), key=str)

            fig_concept.add_trace(go.Bar(
                x=_cats,
                y=[_ref_vc.get(c, 0) for c in _cats],
                name="Reference",
                marker_color="#4ECDC4",
                opacity=0.85,
            ))
            fig_concept.add_trace(go.Bar(
                x=_cats,
                y=[_new_vc.get(c, 0) for c in _cats],
                name="Current",
                marker_color="#FF6B6B",
                opacity=0.85,
            ))
            fig_concept.update_layout(
                barmode="group",
                title=f"Class Distribution: {_pred_col}  (PSI={_psi_c:.4f})",
                xaxis_title="Predicted Class",
                yaxis_title="Proportion",
                **PLOTLY_LAYOUT,
            )

        st.plotly_chart(fig_concept, width='stretch')

    except Exception as _exc:
        st.warning(f"Could not render concept drift chart: {_exc}")

# ── Anomaly Rate Over Time ─────────────────────────────────────────────────────
st.subheader("Anomaly Rate Over Time")

st.info(
    "Upload a CSV with two columns: `timestamp` (date or datetime) and "
    "`anomaly_rate` (fraction of anomalies detected in that time window)."
)

anomaly_rate_file = st.file_uploader(
    "Anomaly rate tracking CSV",
    type=["csv"],
    key="anomaly_rate_upload",
    help="Required columns: 'timestamp' and 'anomaly_rate'.",
)

if anomaly_rate_file is not None:
    try:
        df_anomaly = pd.read_csv(anomaly_rate_file)

        if "timestamp" not in df_anomaly.columns or "anomaly_rate" not in df_anomaly.columns:
            st.warning(
                "CSV must contain columns named `timestamp` and `anomaly_rate`. "
                f"Found: {list(df_anomaly.columns)}"
            )
        else:
            try:
                df_anomaly["timestamp"] = pd.to_datetime(df_anomaly["timestamp"])
            except Exception:
                st.warning("Could not parse `timestamp` column as datetime. Check the date format.")

            df_anomaly = df_anomaly.sort_values("timestamp")
            df_anomaly["anomaly_rate"] = pd.to_numeric(df_anomaly["anomaly_rate"], errors="coerce")
            df_anomaly = df_anomaly.dropna(subset=["anomaly_rate"])

            contamination_rate = 0.05  # default baseline threshold

            # Check if any point exceeds 2x the baseline rate
            _above_threshold = df_anomaly["anomaly_rate"] > 2 * contamination_rate
            if _above_threshold.any():
                _n_above = int(_above_threshold.sum())
                _max_rate = float(df_anomaly["anomaly_rate"].max())
                st.warning(
                    f"Alert: {_n_above} time window(s) show an anomaly rate above "
                    f"2x the baseline ({contamination_rate * 100:.1f}%). "
                    f"Maximum observed rate: {_max_rate:.2%}. "
                    "This may indicate a significant shift in data quality or model behavior."
                )

            try:
                fig_anomaly = go.Figure()

                fig_anomaly.add_trace(go.Scatter(
                    x=df_anomaly["timestamp"],
                    y=df_anomaly["anomaly_rate"],
                    mode="lines+markers",
                    name="Anomaly Rate",
                    line=dict(color="#FF6B6B", width=2),
                    marker=dict(size=6, color="#FF6B6B"),
                ))

                # Threshold line at contamination_rate
                fig_anomaly.add_hline(
                    y=contamination_rate,
                    line_dash="dash",
                    line_color="#FFEAA7",
                    annotation_text=f"Baseline ({contamination_rate:.0%})",
                    annotation_position="bottom right",
                    annotation_font_color="#FFEAA7",
                )

                # 2x threshold warning line
                fig_anomaly.add_hline(
                    y=contamination_rate * 2,
                    line_dash="dot",
                    line_color="#FF6B6B",
                    annotation_text=f"2x Baseline ({contamination_rate * 2:.0%})",
                    annotation_position="top right",
                    annotation_font_color="#FF6B6B",
                )

                fig_anomaly.update_layout(
                    title="Anomaly Rate Over Time",
                    xaxis_title="Time",
                    yaxis_title="Anomaly Rate",
                    yaxis_tickformat=".1%",
                    **PLOTLY_LAYOUT,
                )

                st.plotly_chart(fig_anomaly, width='stretch')

                # Summary stats
                _ar_col1, _ar_col2, _ar_col3 = st.columns(3)
                _ar_col1.metric("Mean Anomaly Rate", f"{df_anomaly['anomaly_rate'].mean():.2%}")
                _ar_col2.metric("Max Anomaly Rate", f"{df_anomaly['anomaly_rate'].max():.2%}")
                _ar_col3.metric("Baseline Threshold", f"{contamination_rate:.0%}")

            except Exception as _exc:
                st.warning(f"Could not render anomaly rate chart: {_exc}")

    except Exception as _exc:
        st.warning(f"Error reading anomaly rate file: {_exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3c — Cluster Distribution Drift
# ══════════════════════════════════════════════════════════════════════════════
_task_type_mon = st.session_state.get("task_type", "")
if _task_type_mon == "clustering":
    st.divider()
    st.header("Cluster Distribution Drift")
    st.info(
        "Upload a new dataset to check whether the distribution of points across "
        "clusters has shifted relative to training. Uses chi-square test and PSI on "
        "cluster proportions, plus distance-based boundary outlier detection."
    )

    _cda_map = st.session_state.get("training_adapters", {})
    _cluster_adapter = None
    for _cda_val in _cda_map.values():
        if hasattr(_cda_val, "labels_"):
            _cluster_adapter = _cda_val
            break

    if _cluster_adapter is None:
        st.warning("No trained clustering model found in session. Train a clustering model first.")
    else:
        _clust_upload = st.file_uploader(
            "Upload new data CSV (same columns as training)",
            type=["csv"],
            key="cluster_drift_upload",
            help="The file will be preprocessed with the same pipeline used during training.",
        )

        if _clust_upload is not None:
            try:
                _df_clust_new = pd.read_csv(_clust_upload)
                _target_col_cd = st.session_state.get("target_column", "")
                if _target_col_cd and _target_col_cd in _df_clust_new.columns:
                    _df_clust_new = _df_clust_new.drop(columns=[_target_col_cd])
                st.write(f"Uploaded: **{len(_df_clust_new):,}** rows × {_df_clust_new.shape[1]} columns")

                if st.button("▶ Analyze Cluster Drift", key="run_cluster_drift", type="primary"):
                    _pipe_cd = st.session_state.get("preprocessing_pipeline")
                    with st.spinner("Processing..."):
                        if _pipe_cd is not None:
                            _X_clust_new = _pipe_cd.transform(_df_clust_new)
                        else:
                            _X_clust_new = _df_clust_new.select_dtypes(include=[np.number]).values
                        _labels_new_cd = _cluster_adapter.predict(_X_clust_new)
                    st.session_state["cluster_drift_result"] = {
                        "labels_new": _labels_new_cd,
                        "labels_ref": _cluster_adapter.labels_,
                        "X_ref": getattr(_cluster_adapter, "_X_train", None),
                        "X_new": _X_clust_new,
                    }
                    st.success(f"Cluster assignments computed for {len(_labels_new_cd):,} rows.")

            except Exception as _exc:
                st.error(f"Error processing cluster drift data: {_exc}")

    if "cluster_drift_result" in st.session_state:
        _cdr_c = st.session_state["cluster_drift_result"]
        _lbl_new = _cdr_c["labels_new"]
        _lbl_ref = _cdr_c["labels_ref"]
        _X_ref_cd = _cdr_c["X_ref"]
        _X_new_cd = _cdr_c["X_new"]

        # ── Cluster distribution comparison ────────────────────────────────────
        _ref_s = pd.Series(_lbl_ref)
        _new_s = pd.Series(_lbl_new)
        _all_cls = sorted(set(_ref_s.unique()) | set(_new_s.unique()))

        _ref_cnt = _ref_s.value_counts().reindex(_all_cls, fill_value=0)
        _new_cnt = _new_s.value_counts().reindex(_all_cls, fill_value=0)
        _ref_prop = _ref_cnt / max(len(_lbl_ref), 1)
        _new_prop = _new_cnt / max(len(_lbl_new), 1)

        # Chi-square test
        _exp_cnt = _ref_prop.values * len(_lbl_new)
        _obs_cnt = _new_cnt.values
        _valid_mask = _exp_cnt > 0
        if _valid_mask.sum() >= 2:
            _chi2_s, _chi2_p = stats.chisquare(
                f_obs=_obs_cnt[_valid_mask], f_exp=_exp_cnt[_valid_mask]
            )
        else:
            _chi2_s, _chi2_p = 0.0, 1.0

        # PSI on cluster proportions
        _eps = 1e-9
        _psi_c = float(np.sum(
            (_new_prop.values - _ref_prop.values)
            * np.log(
                np.maximum(_new_prop.values, _eps)
                / np.maximum(_ref_prop.values, _eps)
            )
        ))

        # Metrics row
        _cm1, _cm2, _cm3, _cm4 = st.columns(4)
        _cm1.metric("Chi² Statistic", f"{_chi2_s:.4f}")
        _cm2.metric("Chi² p-value", f"{_chi2_p:.4f}")
        _cm3.metric("Cluster PSI", f"{_psi_c:.4f}")
        _cm4.metric("Unique Clusters (new)", int(len(set(_lbl_new))))

        # Verdict
        if _chi2_p >= 0.05 and _psi_c < PSI_NONE:
            st.success("No significant cluster distribution drift detected.")
        elif _chi2_p >= 0.05 or _psi_c < PSI_SLIGHT:
            st.warning(
                "Possible cluster drift — the proportion of data in each cluster "
                "has shifted slightly. Monitor closely."
            )
        else:
            st.error(
                "Significant cluster drift detected — cluster proportions have changed "
                "substantially. Consider retraining the clustering model."
            )

        # Grouped bar chart: cluster proportions reference vs new
        try:
            _fig_cd = go.Figure()
            _fig_cd.add_trace(go.Bar(
                x=[f"Cluster {c}" for c in _all_cls],
                y=_ref_prop.values,
                name="Reference (Training)",
                marker_color="#4ECDC4",
                opacity=0.85,
            ))
            _fig_cd.add_trace(go.Bar(
                x=[f"Cluster {c}" for c in _all_cls],
                y=_new_prop.values,
                name="New Data",
                marker_color="#FF6B6B",
                opacity=0.85,
            ))
            _fig_cd.update_layout(
                barmode="group",
                title=f"Cluster Size Distribution  (PSI={_psi_c:.4f}  |  χ² p={_chi2_p:.4f})",
                xaxis_title="Cluster",
                yaxis_title="Proportion",
                yaxis_tickformat=".1%",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(_fig_cd, width='stretch')
        except Exception as _exc:
            st.warning(f"Could not render cluster distribution chart: {_exc}")

        # ── Distance-based boundary outlier detection ──────────────────────────
        st.subheader("Boundary Outlier Detection")
        st.caption(
            "Points whose distance to their assigned cluster centroid exceeds the "
            "95th-percentile of training distances are flagged as boundary outliers."
        )

        if _X_ref_cd is not None:
            try:
                # Compute per-cluster centroids from training data
                _centroids_cd = {}
                for _cl in sorted(set(_lbl_ref)):
                    _mask_cl = np.asarray(_lbl_ref) == _cl
                    if _mask_cl.sum() > 0:
                        _centroids_cd[_cl] = _X_ref_cd[_mask_cl].mean(axis=0)

                if _centroids_cd:
                    # Training distances to own centroid
                    _tr_dists = np.array([
                        np.linalg.norm(_X_ref_cd[i] - _centroids_cd[_lbl_ref[i]])
                        for i in range(len(_lbl_ref))
                        if _lbl_ref[i] in _centroids_cd
                    ])
                    _dist_thr = float(np.quantile(_tr_dists, 0.95)) if len(_tr_dists) > 0 else np.inf

                    # New data distances
                    _nd_dists = np.array([
                        np.linalg.norm(_X_new_cd[i] - _centroids_cd[_lbl_new[i]])
                        if _lbl_new[i] in _centroids_cd else np.inf
                        for i in range(len(_lbl_new))
                    ])
                    _n_bound = int((_nd_dists > _dist_thr).sum())
                    _bound_rate = _n_bound / max(len(_lbl_new), 1)

                    _bd1, _bd2, _bd3 = st.columns(3)
                    _bd1.metric("Distance Threshold (p95)", f"{_dist_thr:.4f}")
                    _bd2.metric("Points Outside Boundary", _n_bound)
                    _bd3.metric("Boundary Outlier Rate", f"{_bound_rate:.1%}")

                    if _bound_rate > 0.10:
                        st.warning(
                            f"**{_bound_rate:.1%}** of new data points fall outside "
                            "normal cluster boundaries — possible new patterns or data drift."
                        )
                    else:
                        st.success(
                            f"Only **{_bound_rate:.1%}** of new points exceed cluster "
                            "boundaries — distribution appears stable."
                        )

                    # Distance histogram: training vs new
                    try:
                        _fig_dist_cd = go.Figure()
                        _fig_dist_cd.add_trace(go.Histogram(
                            x=_tr_dists,
                            name="Training",
                            opacity=0.65,
                            marker_color="#4ECDC4",
                            histnorm="probability",
                        ))
                        _nd_finite = _nd_dists[np.isfinite(_nd_dists)]
                        _fig_dist_cd.add_trace(go.Histogram(
                            x=_nd_finite,
                            name="New Data",
                            opacity=0.65,
                            marker_color="#FF6B6B",
                            histnorm="probability",
                        ))
                        _fig_dist_cd.add_vline(
                            x=_dist_thr,
                            line_dash="dash",
                            line_color="#FFEAA7",
                            annotation_text=f"p95 = {_dist_thr:.3f}",
                            annotation_position="top right",
                            annotation_font_color="#FFEAA7",
                        )
                        _fig_dist_cd.update_layout(
                            barmode="overlay",
                            title="Distance to Nearest Cluster Centroid",
                            xaxis_title="Distance",
                            yaxis_title="Density",
                            **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(_fig_dist_cd, width='stretch')
                    except Exception as _exc:
                        st.warning(f"Could not render distance histogram: {_exc}")

            except Exception as _exc:
                st.warning(f"Error computing boundary distances: {_exc}")
        else:
            st.info(
                "Training data not available in adapter — distance-based detection skipped. "
                "(DBSCAN and Agglomerative adapters store training data; others may not.)"
            )

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Automated Retraining
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("4 — Automated Retraining")

import json  # noqa: E402 (deferred to avoid top-level import cost)
import os    # noqa: E402
from src.pipelines.retraining import MODEL_DISPLAY_NAMES, run_retraining  # noqa: E402
from src.pipelines.training import normalize_task_type  # noqa: E402

TASK_OPTIONS = [
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
]
TASK_LABELS = {
    "binary_classification": "Binary Classification",
    "multiclass_classification": "Multiclass Classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "anomaly_detection": "Anomaly Detection",
    "dimensionality_reduction": "Dimensionality Reduction",
}

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
_task_type_default = normalize_task_type(st.session_state.get("task_type", "binary_classification"))
_task_default_idx = TASK_OPTIONS.index(_task_type_default) if _task_type_default in TASK_OPTIONS else 0

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
            TASK_OPTIONS,
            index=_task_default_idx,
            format_func=lambda t: TASK_LABELS.get(t, t),
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
        width='stretch',
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
        width='stretch',
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
st.page_link("pages/6_Prediction.py", label="← Back to Prediction")
