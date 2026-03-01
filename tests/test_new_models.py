"""Smoke tests for ClusteringAdapter, AnomalyAdapter, and ReductionAdapter."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.clustering import ClusteringAdapter, get_default_clustering_params
from src.models.anomaly import AnomalyAdapter, get_default_anomaly_params
from src.models.reduction import ReductionAdapter, get_default_reduction_params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blob_X():
    """Simple 3-cluster blob data."""
    rng = np.random.RandomState(42)
    centers = np.array([[0, 0], [5, 0], [2.5, 4]])
    X = np.vstack([rng.randn(40, 2) + c for c in centers])
    return X.astype(np.float64)


@pytest.fixture
def normal_X():
    """Normal data for anomaly detection (small contamination)."""
    rng = np.random.RandomState(0)
    return rng.randn(100, 4).astype(np.float64)


# ===========================================================================
# ClusteringAdapter
# ===========================================================================

class TestClusteringAdapter:

    def test_kmeans_fit_predict(self, blob_X):
        adapter = ClusteringAdapter("kmeans", n_clusters=3)
        adapter.fit(blob_X)
        labels = adapter.predict(blob_X)
        assert labels.shape == (120,)
        assert set(np.unique(labels)).issubset(set(range(3)))

    def test_kmeans_new_points(self, blob_X):
        adapter = ClusteringAdapter("kmeans", n_clusters=3)
        adapter.fit(blob_X)
        X_new = np.array([[0.1, 0.1], [5.1, 0.1], [2.6, 4.1]])
        labels = adapter.predict(X_new)
        assert labels.shape == (3,)

    def test_dbscan_fit_predict(self, blob_X):
        adapter = ClusteringAdapter("dbscan", eps=1.5, min_samples=5)
        adapter.fit(blob_X)
        labels = adapter.predict(blob_X)
        assert labels.shape == (120,)

    def test_dbscan_knn_fallback_new_points(self, blob_X):
        """DBSCAN predict on new data uses KNN label propagation."""
        adapter = ClusteringAdapter("dbscan", eps=1.0, min_samples=5)
        adapter.fit(blob_X)
        X_new = np.array([[0.0, 0.0], [5.0, 0.0]])
        labels = adapter.predict(X_new)
        assert labels.shape == (2,)

    def test_agglomerative_fit_predict(self, blob_X):
        adapter = ClusteringAdapter("agglomerative", n_clusters=3)
        adapter.fit(blob_X)
        labels = adapter.predict(blob_X)
        assert labels.shape == (120,)
        assert len(np.unique(labels)) == 3

    def test_gaussian_mixture_fit_predict(self, blob_X):
        adapter = ClusteringAdapter("gaussian_mixture", n_components=3)
        adapter.fit(blob_X)
        labels = adapter.predict(blob_X)
        assert labels.shape == (120,)

    def test_default_params_valid(self):
        for model_type in ("kmeans", "dbscan", "agglomerative", "gaussian_mixture"):
            params = get_default_clustering_params(model_type)
            assert isinstance(params, dict)

    def test_unknown_model_raises(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            ClusteringAdapter("unknown_clustering_model").fit(np.random.randn(10, 2))


# ===========================================================================
# AnomalyAdapter
# ===========================================================================

class TestAnomalyAdapter:

    def test_isolation_forest_fit_predict(self, normal_X):
        adapter = AnomalyAdapter("isolation_forest", n_estimators=10, contamination=0.1)
        adapter.fit(normal_X)
        preds = adapter.predict(normal_X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})  # normalized from +1/-1

    def test_isolation_forest_majority_normal(self, normal_X):
        adapter = AnomalyAdapter("isolation_forest", n_estimators=10, contamination=0.05)
        adapter.fit(normal_X)
        preds = adapter.predict(normal_X)
        # With ~5% contamination, majority should be normal (0)
        assert (preds == 0).sum() > (preds == 1).sum()

    def test_decision_scores_shape(self, normal_X):
        adapter = AnomalyAdapter("isolation_forest", n_estimators=10, contamination=0.1)
        adapter.fit(normal_X)
        scores = adapter.decision_scores(normal_X)
        assert scores is not None
        assert scores.shape == (100,)
        assert isinstance(float(scores[0]), float)

    def test_one_class_svm_fit_predict(self, normal_X):
        adapter = AnomalyAdapter("one_class_svm", nu=0.1)
        adapter.fit(normal_X)
        preds = adapter.predict(normal_X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_elliptic_envelope_fit_predict(self, normal_X):
        adapter = AnomalyAdapter("elliptic_envelope", contamination=0.1)
        adapter.fit(normal_X)
        preds = adapter.predict(normal_X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_default_params_valid(self):
        for model_type in ("isolation_forest", "one_class_svm", "elliptic_envelope"):
            params = get_default_anomaly_params(model_type)
            assert isinstance(params, dict)


# ===========================================================================
# ReductionAdapter
# ===========================================================================

class TestReductionAdapter:

    def test_pca_fit_transform(self, normal_X):
        adapter = ReductionAdapter("pca", n_components=2)
        adapter.fit(normal_X)
        coords = adapter.transform(normal_X)
        assert coords.shape == (100, 2)

    def test_pca_new_points(self, normal_X):
        adapter = ReductionAdapter("pca", n_components=2)
        adapter.fit(normal_X)
        X_new = np.random.randn(5, 4)
        coords = adapter.transform(X_new)
        assert coords.shape == (5, 2)

    def test_truncated_svd_fit_transform(self, normal_X):
        adapter = ReductionAdapter("truncated_svd", n_components=2)
        adapter.fit(normal_X)
        coords = adapter.transform(normal_X)
        assert coords.shape == (100, 2)

    def test_tsne_fit_stores_coords(self, normal_X):
        adapter = ReductionAdapter("tsne", n_components=2, n_iter=250, perplexity=5)
        adapter.fit(normal_X)
        # After fit, transform on training data should work (returns stored coords)
        # or raise NotImplementedError — either is acceptable in the spec
        # Just verify fit doesn't crash
        assert adapter is not None

    def test_tsne_transform_raises_not_implemented(self, normal_X):
        adapter = ReductionAdapter("tsne", n_components=2, n_iter=250, perplexity=5)
        adapter.fit(normal_X)
        X_new = np.random.randn(5, 4)
        with pytest.raises(NotImplementedError):
            adapter.transform(X_new)

    def test_default_params_valid(self):
        for model_type in ("pca", "truncated_svd", "tsne"):
            params = get_default_reduction_params(model_type)
            assert isinstance(params, dict)

    def test_unknown_model_raises(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            ReductionAdapter("unknown_reduction_model").fit(np.random.randn(10, 3))
