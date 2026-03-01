"""Integration smoke tests — one end-to-end test per task type.

These tests verify the full pipeline: data → training → metrics.
They do NOT require Streamlit, MLflow, or any external service.

Run with:
    pytest tests/test_integration.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression

from src.pipelines.training import get_default_params, train_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(X, y=None, test_ratio=0.2):
    """Simple deterministic train/val split."""
    n = len(X)
    cut = int(n * (1 - test_ratio))
    if y is not None:
        return X[:cut], X[cut:], y[:cut], y[cut:]
    return X[:cut], X[cut:]


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def test_binary_classification_pipeline():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=2, random_state=42,
    )
    X_train, X_val, y_train, y_val = _split(X, y)

    result = train_model(
        model_type="logistic_regression",
        task_type="binary_classification",
        params=get_default_params("logistic_regression", "binary_classification"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert result.model is not None, "model must not be None"
    assert result.metrics, "metrics dict must not be empty"
    accuracy = result.metrics.get("accuracy")
    assert accuracy is not None, "accuracy must be present"
    assert 0.0 <= accuracy <= 1.0, f"accuracy out of range: {accuracy}"
    f1 = result.metrics.get("f1")
    if f1 is not None:
        assert 0.0 <= f1 <= 1.0, f"f1 out of range: {f1}"


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------

def test_multiclass_classification_pipeline():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=3, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_val, y_train, y_val = _split(X, y)

    result = train_model(
        model_type="random_forest",
        task_type="multiclass_classification",
        params=get_default_params("random_forest", "multiclass_classification"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert result.model is not None
    assert result.metrics
    accuracy = result.metrics.get("accuracy")
    assert accuracy is not None, "accuracy must be present"
    assert 0.0 <= accuracy <= 1.0, f"accuracy out of range: {accuracy}"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def test_regression_pipeline():
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42,
    )
    X_train, X_val, y_train, y_val = _split(X, y)

    result = train_model(
        model_type="ridge",
        task_type="regression",
        params=get_default_params("ridge", "regression"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert result.model is not None
    assert result.metrics
    r2 = result.metrics.get("r2")
    assert r2 is not None, "r2 must be present"
    assert np.isfinite(r2), f"r2 must be finite, got {r2}"
    rmse = result.metrics.get("rmse")
    if rmse is not None:
        assert rmse >= 0.0, f"rmse must be non-negative, got {rmse}"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_clustering_pipeline():
    X, _ = make_blobs(n_samples=200, n_features=5, centers=4, random_state=42)

    result = train_model(
        model_type="kmeans",
        task_type="clustering",
        params=get_default_params("kmeans", "clustering"),
        X_train=X,
        y_train=None,
        X_val=None,
        y_val=None,
    )

    assert result.model is not None
    assert result.metrics
    silhouette = result.metrics.get("silhouette")
    assert silhouette is not None, "silhouette score must be present"
    assert -1.0 <= silhouette <= 1.0, f"silhouette out of range: {silhouette}"


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_anomaly_detection_pipeline():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)

    result = train_model(
        model_type="isolation_forest",
        task_type="anomaly_detection",
        params=get_default_params("isolation_forest", "anomaly_detection"),
        X_train=X,
        y_train=None,
        X_val=None,
        y_val=None,
    )

    assert result.model is not None
    assert result.metrics
    anomaly_ratio = result.metrics.get("anomaly_ratio")
    assert anomaly_ratio is not None, "anomaly_ratio must be present"
    assert 0.0 <= anomaly_ratio <= 1.0, f"anomaly_ratio out of range: {anomaly_ratio}"


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def test_dimensionality_reduction_pipeline():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)

    result = train_model(
        model_type="pca",
        task_type="dimensionality_reduction",
        params=get_default_params("pca", "dimensionality_reduction"),
        X_train=X,
        y_train=None,
        X_val=None,
        y_val=None,
    )

    assert result.model is not None
    assert result.metrics
    explained_variance = result.metrics.get("explained_variance")
    assert explained_variance is not None, "explained_variance must be present"
    assert 0.0 <= explained_variance <= 1.0, f"explained_variance out of range: {explained_variance}"
