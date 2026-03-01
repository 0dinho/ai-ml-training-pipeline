"""AutoML Prediction API — Phase 7.

FastAPI service exposing:
  GET  /health          — liveness / readiness check
  POST /predict         — single-row inference
  POST /predict/batch   — multi-row inference
  GET  /metrics         — Prometheus metrics

Start with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
  MLFLOW_TRACKING_URI   — MLflow server URL           (default: http://localhost:5001)
  MLFLOW_MODEL_NAME     — Registered model name       (default: "" → use disk)
  MLFLOW_MODEL_STAGE    — Model stage/version         (default: latest)
  PIPELINE_PATH         — Path to preprocessing joblib (default: artifacts/preprocessing_pipeline.joblib)
  MODEL_PATH            — Explicit model joblib path  (default: "" → auto-detect)
  SCHEMA_PATH           — Path to schema JSON         (default: artifacts/schema.json)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from api.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_response,
    record_prediction,
)
from src.pipelines.training import MODEL_DISPLAY_NAMES
from api.model_loader import load_active_model, load_pipeline, load_schema
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    validate_features,
)

# ── Application state ──────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "schema": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup."""
    _state["schema"] = load_schema()
    _state["pipeline"] = load_pipeline()
    _state["model"] = load_active_model()

    schema = _state["schema"]
    if schema:
        print(f"[startup] Schema loaded — task: {schema.get('task_type')}, "
              f"features: {len(schema.get('feature_columns', []))}")
    else:
        print("[startup] No schema.json found — validation disabled.")

    if _state["pipeline"] is None:
        print("[startup] WARNING: preprocessing pipeline not found.")
    if _state["model"] is None:
        print("[startup] WARNING: no model loaded.")

    yield  # server is running


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoML Prediction API",
    description=(
        "Real-time inference API for models trained with the AutoML Platform. "
        "Submit raw feature values and receive predictions with optional "
        "confidence scores. The same preprocessing pipeline used during training "
        "is applied automatically."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware: Prometheus request tracking ────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    return response


# ── Shared inference logic ─────────────────────────────────────────────────────
def _run_inference(rows: list[dict[str, Any]]) -> list[PredictResponse]:
    """Preprocess *rows* and return predictions.

    Raises HTTPException on missing model/pipeline or validation errors.
    """
    model = _state["model"]
    pipeline = _state["pipeline"]
    schema = _state["schema"]

    if model is None:
        raise HTTPException(503, detail="Model not loaded. Check server logs.")
    if pipeline is None:
        raise HTTPException(503, detail="Preprocessing pipeline not loaded.")

    # Schema-based validation
    if schema:
        for idx, row in enumerate(rows):
            errors = validate_features(row, schema)
            if errors:
                raise HTTPException(
                    422,
                    detail={"row_index": idx, "errors": errors},
                )

    # Build DataFrame → apply datetime extraction → transform
    df = pd.DataFrame(rows)

    datetime_extract_cols: list[str] = (
        schema.get("datetime_extract_cols", []) if schema else []
    )
    to_extract = [c for c in datetime_extract_cols if c in df.columns]
    if to_extract:
        from src.pipelines.preprocessing import extract_datetime_features
        df = extract_datetime_features(df, to_extract)

    try:
        X = pipeline.transform(df)
    except Exception as exc:
        raise HTTPException(422, detail=f"Preprocessing failed: {exc}")

    # Determine task type from adapter or schema
    from src.models.adapter import ModelAdapter
    task_type: str = "binary_classification"
    if isinstance(model, ModelAdapter):
        task_type = model.canonical_task_type()
    elif schema:
        task_type = schema.get("task_type", "binary_classification")

    # Predict — dispatch by task type
    results: list[PredictResponse] = []

    if isinstance(model, ModelAdapter) and model.is_reduction():
        # Dimensionality reduction: transform returns 2D array
        coords = model.transform(X)
        for i in range(len(coords)):
            row_coords = [round(float(c), 6) for c in coords[i]]
            record_prediction(0, task_type, None)
            results.append(PredictResponse(prediction=None, reduced_coords=row_coords))

    elif isinstance(model, ModelAdapter) and model.is_anomaly():
        preds = model.predict(X)
        scores = model.decision_scores(X)
        for i, pred in enumerate(preds):
            score = float(scores[i]) if scores is not None else None
            record_prediction(int(pred), task_type, None)
            results.append(PredictResponse(
                prediction=int(pred),
                is_anomaly=bool(pred == 1),
                anomaly_score=round(score, 6) if score is not None else None,
            ))

    elif isinstance(model, ModelAdapter) and model.is_clustering():
        labels = model.predict(X)
        for label in labels:
            record_prediction(int(label), task_type, None)
            results.append(PredictResponse(
                prediction=int(label),
                cluster_label=int(label),
            ))

    else:
        # Supervised: classification or regression
        y_pred: np.ndarray = model.predict(X)

        y_proba: np.ndarray | None = None
        if isinstance(model, ModelAdapter):
            y_proba = model.predict_proba(X)
        elif hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                pass

        for i, pred in enumerate(y_pred):
            probabilities: list[float] | None = None
            confidence: float | None = None

            if task_type in ("binary_classification", "multiclass_classification",
                             "classification") and y_proba is not None:
                probabilities = [round(float(p), 6) for p in y_proba[i]]
                confidence = round(float(max(y_proba[i])), 6)

            record_prediction(pred, task_type, confidence)
            results.append(PredictResponse(
                prediction=pred.item() if isinstance(pred, np.generic) else pred,
                probabilities=probabilities,
                confidence=confidence,
            ))

    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """Liveness / readiness check — always returns 200."""
    schema = _state["schema"]
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        pipeline_loaded=_state["pipeline"] is not None,
        schema_loaded=schema is not None,
        task_type=schema.get("task_type") if schema else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["infra"])
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    model = _state["model"]
    schema = _state["schema"]

    if model is None:
        raise HTTPException(503, detail="No model loaded.")

    from src.models.adapter import ModelAdapter
    if isinstance(model, ModelAdapter):
        return ModelInfoResponse(
            model_name=MODEL_DISPLAY_NAMES.get(model.model_type, model.model_type),
            model_type=model.model_type,
            task_type=model.canonical_task_type(),
            feature_names=model.feature_names,
            target_column=model.target_column,
            n_features=len(model.feature_names),
            classes=model.classes_ if model.classes_ else None,
            n_components=model.n_components_,
            metadata=model.metadata,
        )

    # Legacy raw estimator fallback
    feature_names = schema.get("feature_names", []) if schema else []
    task_type = schema.get("task_type", "unknown") if schema else "unknown"
    target_column = schema.get("target_column", "") if schema else ""
    return ModelInfoResponse(
        model_name=type(model).__name__,
        model_type=type(model).__name__.lower(),
        task_type=task_type,
        feature_names=feature_names,
        target_column=target_column,
        n_features=len(feature_names),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Single-row prediction.

    Submit a flat dict of feature values under ``features``.
    Returns the predicted label and, for classification tasks,
    per-class probabilities and the top-class confidence score.
    """
    results = _run_inference([request.features])
    return results[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Batch prediction — submit multiple rows in a single request.

    ``rows`` is a list of feature dicts, one per observation.
    """
    if not request.rows:
        raise HTTPException(422, detail="'rows' must contain at least one item.")
    results = _run_inference(request.rows)
    return BatchPredictResponse(count=len(results), predictions=results)


@app.get("/metrics", tags=["infra"], include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint (text/plain; version=0.0.4)."""
    return metrics_response()


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
