"""Load the active model and preprocessing pipeline from disk or MLflow."""
from __future__ import annotations

import json
import os
from typing import Any

import joblib

# ── Environment-variable configuration ────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "")
MLFLOW_MODEL_STAGE: str = os.getenv("MLFLOW_MODEL_STAGE", "latest")

PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", "artifacts/preprocessing_pipeline.joblib")
MODEL_PATH: str = os.getenv("MODEL_PATH", "")
SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "artifacts/schema.json")


def load_schema() -> dict | None:
    """Load the training schema produced by the preprocessing step."""
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH) as f:
            return json.load(f)
    return None


def load_pipeline() -> Any | None:
    """Load the fitted sklearn preprocessing pipeline from disk."""
    if os.path.exists(PIPELINE_PATH):
        return joblib.load(PIPELINE_PATH)
    return None


def load_active_model() -> "Any | None":
    """Load the active model as a ModelAdapter.

    Priority order:
      1. MLflow Model Registry  (when ``MLFLOW_MODEL_NAME`` env var is set)
      2. Explicit ``MODEL_PATH`` env var
      3. Auto-detect: most recently modified ``*_adapter.joblib`` or ``*_model.joblib``
         in ``artifacts/`` — both are wrapped transparently via ModelAdapter.load().
    """
    from src.models.adapter import ModelAdapter

    # 1. MLflow registry
    if MLFLOW_MODEL_NAME:
        try:
            import mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            raw = mlflow.sklearn.load_model(uri)
            print(f"[model_loader] Loaded from MLflow registry: {uri}")
            # Wrap legacy raw estimators
            if not isinstance(raw, ModelAdapter):
                return ModelAdapter._wrap_legacy(raw)
            return raw
        except Exception as e:
            print(f"[model_loader] MLflow load failed ({e}), falling back to disk.")

    # 2. Explicit path
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"[model_loader] Loaded from explicit path: {MODEL_PATH}")
        return ModelAdapter.load(MODEL_PATH)

    # 3. Auto-detect in artifacts/ — prefer *_adapter.joblib, fall back to *_model.joblib
    artifact_dir = "artifacts"
    if os.path.isdir(artifact_dir):
        all_files = os.listdir(artifact_dir)
        candidates = [
            f for f in all_files
            if f.endswith("_adapter.joblib") or f.endswith("_model.joblib")
        ]
        if candidates:
            candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(artifact_dir, f)),
                reverse=True,
            )
            path = os.path.join(artifact_dir, candidates[0])
            print(f"[model_loader] Auto-detected model: {path}")
            return ModelAdapter.load(path)

    print("[model_loader] No model found.")
    return None
