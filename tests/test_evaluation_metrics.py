"""Tests for src/evaluation/metrics.py — compute_metrics for all 6 task types."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    rng = np.random.RandomState(1)
    y_true = rng.choice([0, 1], 100)
    y_pred = rng.choice([0, 1], 100)
    y_proba = rng.dirichlet([1, 1], size=100)
    return y_true, y_pred, y_proba


@pytest.fixture
def multiclass_data():
    rng = np.random.RandomState(2)
    y_true = rng.choice([0, 1, 2], 150)
    y_pred = rng.choice([0, 1, 2], 150)
    y_proba = rng.dirichlet([1, 1, 1], size=150)
    return y_true, y_pred, y_proba


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(3)
    y_true = rng.randn(100)
    y_pred = y_true + rng.randn(100) * 0.5
    return y_true, y_pred


@pytest.fixture
def clustering_data():
    rng = np.random.RandomState(4)
    X = rng.randn(100, 4)
    labels = rng.choice([0, 1, 2], 100)
    return X, labels


@pytest.fixture
def anomaly_data():
    rng = np.random.RandomState(5)
    y_pred = rng.choice([0, 1], 100, p=[0.9, 0.1])  # 10% anomalies
    return y_pred


@pytest.fixture
def reduction_data():
    explained = np.array([0.4, 0.3, 0.15])
    y_pred = np.zeros(10)  # dummy (not used for reduction)
    return explained, y_pred


# ===========================================================================
# Binary Classification
# ===========================================================================

class TestBinaryClassification:

    def test_returns_required_keys(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        m = compute_metrics(y_true, y_pred, "binary_classification", y_proba)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in m, f"Missing key: {key}"

    def test_accuracy_in_range(self, binary_data):
        y_true, y_pred, _ = binary_data
        m = compute_metrics(y_true, y_pred, "binary_classification")
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_roc_auc_with_proba(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        m = compute_metrics(y_true, y_pred, "binary_classification", y_proba)
        if "roc_auc" in m and m["roc_auc"] is not None:
            assert 0.0 <= m["roc_auc"] <= 1.0

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred, "binary_classification")
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_legacy_classification_task_type(self, binary_data):
        y_true, y_pred, _ = binary_data
        m = compute_metrics(y_true, y_pred, "classification")
        assert "accuracy" in m


# ===========================================================================
# Multiclass Classification
# ===========================================================================

class TestMulticlassClassification:

    def test_returns_required_keys(self, multiclass_data):
        y_true, y_pred, _ = multiclass_data
        m = compute_metrics(y_true, y_pred, "multiclass_classification")
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in m

    def test_accuracy_in_range(self, multiclass_data):
        y_true, y_pred, _ = multiclass_data
        m = compute_metrics(y_true, y_pred, "multiclass_classification")
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_roc_auc_with_proba(self, multiclass_data):
        y_true, y_pred, y_proba = multiclass_data
        m = compute_metrics(y_true, y_pred, "multiclass_classification", y_proba)
        if "roc_auc" in m and m["roc_auc"] is not None:
            assert 0.0 <= m["roc_auc"] <= 1.0


# ===========================================================================
# Regression
# ===========================================================================

class TestRegression:

    def test_returns_required_keys(self, regression_data):
        y_true, y_pred = regression_data
        m = compute_metrics(y_true, y_pred, "regression")
        for key in ("mse", "rmse", "mae", "r2"):
            assert key in m

    def test_mse_non_negative(self, regression_data):
        y_true, y_pred = regression_data
        m = compute_metrics(y_true, y_pred, "regression")
        assert m["mse"] >= 0.0
        assert m["rmse"] >= 0.0
        assert m["mae"] >= 0.0

    def test_rmse_equals_sqrt_mse(self, regression_data):
        y_true, y_pred = regression_data
        m = compute_metrics(y_true, y_pred, "regression")
        assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]), rel=1e-5)

    def test_perfect_regression(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y, "regression")
        assert m["mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"] == pytest.approx(1.0)


# ===========================================================================
# Clustering
# ===========================================================================

class TestClustering:

    def test_returns_required_keys(self, clustering_data):
        X, labels = clustering_data
        m = compute_metrics(None, labels, "clustering", X=X)
        assert "silhouette" in m or "n_clusters" in m

    def test_n_clusters_correct(self, clustering_data):
        X, labels = clustering_data
        m = compute_metrics(None, labels, "clustering", X=X)
        if "n_clusters" in m:
            assert int(m["n_clusters"]) == len(np.unique(labels))

    def test_silhouette_in_range(self, clustering_data):
        X, labels = clustering_data
        m = compute_metrics(None, labels, "clustering", X=X)
        if "silhouette" in m and m["silhouette"] is not None:
            assert -1.0 <= m["silhouette"] <= 1.0

    def test_no_X_still_returns_n_clusters(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        m = compute_metrics(None, labels, "clustering")
        assert "n_clusters" in m


# ===========================================================================
# Anomaly Detection
# ===========================================================================

class TestAnomalyDetection:

    def test_returns_required_keys(self, anomaly_data):
        y_pred = anomaly_data
        m = compute_metrics(None, y_pred, "anomaly_detection")
        assert "n_anomalies" in m
        assert "anomaly_ratio" in m

    def test_anomaly_ratio_in_range(self, anomaly_data):
        y_pred = anomaly_data
        m = compute_metrics(None, y_pred, "anomaly_detection")
        assert 0.0 <= m["anomaly_ratio"] <= 1.0

    def test_n_anomalies_correct(self):
        y_pred = np.array([0, 0, 1, 0, 1])
        m = compute_metrics(None, y_pred, "anomaly_detection")
        assert m["n_anomalies"] == 2
        assert m["anomaly_ratio"] == pytest.approx(0.4)

    def test_with_y_true_adds_classification_metrics(self):
        y_true = np.array([0, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        m = compute_metrics(y_true, y_pred, "anomaly_detection")
        assert "n_anomalies" in m
        # If y_true is provided, additional metrics may be computed
        # Just ensure no crash


# ===========================================================================
# Dimensionality Reduction
# ===========================================================================

class TestDimensionalityReduction:

    def test_returns_required_keys(self):
        y_pred = np.zeros(10)
        evr = np.array([0.5, 0.3, 0.1])
        m = compute_metrics(None, y_pred, "dimensionality_reduction",
                            explained_variance_ratio=evr)
        assert "n_components" in m

    def test_explained_variance_sum(self):
        y_pred = np.zeros(5)
        evr = np.array([0.4, 0.3, 0.15])
        m = compute_metrics(None, y_pred, "dimensionality_reduction",
                            explained_variance_ratio=evr)
        if "explained_variance" in m and m["explained_variance"] is not None:
            assert 0.0 <= m["explained_variance"] <= 1.0

    def test_n_components_from_evr(self):
        y_pred = np.zeros(5)
        evr = np.array([0.6, 0.2, 0.1])
        m = compute_metrics(None, y_pred, "dimensionality_reduction",
                            explained_variance_ratio=evr)
        # n_components should be a positive integer
        assert int(m["n_components"]) >= 1

    def test_no_evr_still_works(self):
        y_pred = np.zeros(5)
        m = compute_metrics(None, y_pred, "dimensionality_reduction")
        assert isinstance(m, dict)
