"""Statistical data-drift detection — no UI imports.

Provides per-column drift tests that the Streamlit monitoring page and any
automated retraining job can consume.

Supported tests
---------------
* **KS test** (Kolmogorov-Smirnov) — numerical columns
* **Chi-squared test**              — categorical columns
* **PSI** (Population Stability Index) — both column types

PSI thresholds (standard industry convention)
---------------------------------------------
  PSI < 0.10   → no significant drift
  0.10 ≤ PSI < 0.25 → slight drift (monitor)
  PSI ≥ 0.25   → significant drift (consider retraining)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ──────────────────────────────────────────────────────────────────
PSI_BINS: int = 10
PSI_NONE: float = 0.10
PSI_SLIGHT: float = 0.25
KS_ALPHA: float = 0.05
CHI2_ALPHA: float = 0.05


# ── PSI helpers ────────────────────────────────────────────────────────────────

def _psi_numerical(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Compute PSI for a numerical column using equal-width bins over the combined range."""
    eps = 1e-8
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    if lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_freq = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    cur_freq = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

    return float(np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq)))


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """Compute PSI for a categorical column over all observed categories."""
    eps = 1e-8
    cats = sorted(set(reference.unique()) | set(current.unique()), key=str)
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    ref_freq = np.array([ref_counts.get(c, eps) for c in cats], dtype=float)
    cur_freq = np.array([cur_counts.get(c, eps) for c in cats], dtype=float)
    ref_freq /= ref_freq.sum()
    cur_freq /= cur_freq.sum()

    return float(np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq)))


def _psi_label(psi: float) -> tuple[str, str]:
    """Return (human label, traffic-light colour) for a PSI value."""
    if psi < PSI_NONE:
        return "No drift", "green"
    if psi < PSI_SLIGHT:
        return "Slight drift", "orange"
    return "Significant drift", "red"


# ── Per-column tests ───────────────────────────────────────────────────────────

def _test_numerical(col: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    ref_vals = ref.dropna().values
    cur_vals = cur.dropna().values

    if len(ref_vals) < 5 or len(cur_vals) < 5:
        return {"column": col, "type": "numerical", "error": "Insufficient non-null values (<5)."}

    ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
    psi = _psi_numerical(ref_vals, cur_vals)
    label, color = _psi_label(psi)
    drifted = psi >= PSI_NONE or ks_pval < KS_ALPHA

    return {
        "column": col,
        "type": "numerical",
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pval), 4),
        "psi": round(psi, 4),
        "drift_label": label,
        "drift_color": color,
        "drifted": drifted,
    }


def _test_categorical(col: str, ref: pd.Series, cur: pd.Series) -> dict[str, Any]:
    psi = _psi_categorical(ref, cur)
    label, color = _psi_label(psi)

    # Chi-squared: scale reference counts to current total
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    ref_freq = np.array([ref_counts.get(c, 0) for c in cats], dtype=float)
    cur_freq = np.array([cur_counts.get(c, 0) for c in cats], dtype=float)

    chi2_stat = chi2_pval = None
    if ref_freq.sum() > 0 and cur_freq.sum() > 0:
        expected = ref_freq / ref_freq.sum() * cur_freq.sum()
        mask = expected > 0
        if mask.sum() >= 2:
            try:
                chi2_stat, chi2_pval = stats.chisquare(cur_freq[mask], f_exp=expected[mask])
                chi2_stat = round(float(chi2_stat), 4)
                chi2_pval = round(float(chi2_pval), 4)
            except Exception:
                pass

    drifted = psi >= PSI_NONE or (chi2_pval is not None and chi2_pval < CHI2_ALPHA)

    return {
        "column": col,
        "type": "categorical",
        "chi2_statistic": chi2_stat,
        "chi2_pvalue": chi2_pval,
        "psi": round(psi, 4),
        "drift_label": label,
        "drift_color": color,
        "drifted": drifted,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_types: dict[str, str],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Run drift tests for all feature columns.

    Parameters
    ----------
    reference_df:
        Training / reference data in raw (pre-preprocessing) format.
    current_df:
        New / production data in the same format.
    column_types:
        Mapping of column name → detected type (``'numerical'`` or
        ``'categorical'``).  Other types are skipped.
    feature_columns:
        Ordered list of columns to test (target excluded).

    Returns
    -------
    list of per-column result dicts, each containing at minimum:
        ``column``, ``type``, ``psi``, ``drift_label``, ``drift_color``,
        ``drifted`` (bool).
    """
    results: list[dict[str, Any]] = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ctype = column_types.get(col, "text")
        if ctype == "numerical":
            results.append(_test_numerical(col, reference_df[col], current_df[col]))
        elif ctype == "categorical":
            results.append(_test_categorical(col, reference_df[col], current_df[col]))
        # datetime / text: skip
    return results


def drift_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate drift results into a high-level summary dict."""
    valid = [r for r in results if "error" not in r]
    drifted = [r for r in valid if r.get("drifted")]
    return {
        "total_columns": len(results),
        "tested_columns": len(valid),
        "drifted_columns": len(drifted),
        "drift_rate": round(len(drifted) / len(valid), 3) if valid else 0.0,
        "max_psi": round(max((r["psi"] for r in valid), default=0.0), 4),
        "drifted_column_names": [r["column"] for r in drifted],
    }
