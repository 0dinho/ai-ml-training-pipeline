"""Pydantic request/response models and schema-based input validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ── Request models ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-row prediction request.

    ``features`` is a flat dict mapping original training column names to values.
    The API applies the same preprocessing pipeline used during training.

    Example::

        {
          "features": {
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
          }
        }
    """

    features: dict[str, Any]


class BatchPredictRequest(BaseModel):
    """Batch prediction request — a list of feature dicts."""

    rows: list[dict[str, Any]]


# ── Response models ────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    # ── Supervised fields ──────────────────────────────────────────────────────
    prediction: Any
    probabilities: list[float] | None = None
    confidence: float | None = None
    # ── Clustering ─────────────────────────────────────────────────────────────
    cluster_label: int | None = None
    # ── Anomaly detection ──────────────────────────────────────────────────────
    is_anomaly: bool | None = None
    anomaly_score: float | None = None
    # ── Dimensionality reduction ───────────────────────────────────────────────
    reduced_coords: list[float] | None = None


class BatchPredictResponse(BaseModel):
    count: int
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    schema_loaded: bool
    task_type: str | None = None


class ModelInfoResponse(BaseModel):
    """Detailed model metadata returned by GET /model/info."""
    model_name: str
    model_type: str
    task_type: str
    feature_names: list[str]
    target_column: str
    n_features: int
    classes: list[Any] | None = None
    n_components: int | None = None
    metadata: dict[str, Any] = {}


# ── Schema-based validation ────────────────────────────────────────────────────

def validate_features(features: dict[str, Any], schema: dict) -> list[str]:
    """Validate a single feature dict against the saved training schema.

    Returns a (possibly empty) list of human-readable error strings.
    Only columns that the pipeline actually processes are required;
    text and dropped columns are skipped.
    """
    errors: list[str] = []
    column_config: dict = schema.get("column_config", {})
    column_types: dict = schema.get("column_types", {})

    for col, cfg in column_config.items():
        ctype = column_types.get(col, "text")

        # Text columns and columns with action=drop are never sent to the pipeline
        if ctype == "text":
            continue
        if cfg.get("action") == "drop":
            continue

        val = features.get(col)

        if val is None:
            # Only hard-require if imputation strategy is "drop" (row would be dropped)
            if cfg.get("imputation") == "drop":
                errors.append(f"'{col}' is required (imputation strategy is 'drop').")
            continue

        # Lightweight type coercion check
        if ctype == "numerical":
            try:
                float(val)
            except (TypeError, ValueError):
                errors.append(f"'{col}' must be numeric, got {type(val).__name__!r}.")

    return errors
