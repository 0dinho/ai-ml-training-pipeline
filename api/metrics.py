"""Prometheus metrics definitions for the prediction API."""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# ── Metric definitions ─────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of HTTP requests received.",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "End-to-end HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_TOTAL = Counter(
    "api_predictions_total",
    "Total number of individual predictions served.",
    ["task_type"],
)

PREDICTION_CLASS_DIST = Counter(
    "api_prediction_class_total",
    "Per-class prediction count (classification tasks).",
    ["predicted_class"],
)

PREDICTION_CONFIDENCE = Histogram(
    "api_prediction_confidence",
    "Distribution of max predicted probability (classification).",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


# ── Helper functions ───────────────────────────────────────────────────────────

def record_prediction(pred: object, task_type: str, confidence: float | None = None) -> None:
    """Update counters and histograms for a single prediction."""
    PREDICTION_TOTAL.labels(task_type=task_type).inc()
    if task_type == "classification":
        PREDICTION_CLASS_DIST.labels(predicted_class=str(pred)).inc()
        if confidence is not None:
            PREDICTION_CONFIDENCE.observe(confidence)


def metrics_response() -> Response:
    """Return a Prometheus text-format response for the /metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
