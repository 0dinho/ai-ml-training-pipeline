"""Clustering model adapter and factory functions.

Supported algorithms:
    kmeans             — KMeans
    mean_shift         — MeanShift (bandwidth auto-estimated or configurable)
    dbscan             — DBSCAN (predict uses KNN label propagation for new data)
    agglomerative      — AgglomerativeClustering
    gaussian_mixture   — GaussianMixture (EM algorithm)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

CLUSTERING_MODELS: dict[str, str] = {
    "kmeans": "K-Means",
    "mean_shift": "Mean Shift",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative Clustering",
    "gaussian_mixture": "Gaussian Mixture Model",
}


class ClusteringAdapter(BaseEstimator):
    """Sklearn-compatible wrapper that provides a uniform fit/predict interface
    for clustering algorithms.

    All clustering algorithms are fit without labels (y=None).
    predict() always returns integer cluster labels.

    Note on DBSCAN / AgglomerativeClustering:
        These algorithms have no native predict() for new data.
        After fitting, this adapter stores (X_train, labels_) and uses
        KNN label propagation (nearest training neighbour) for new points.
        This is an approximation documented here intentionally.
    """

    def __init__(self, algorithm: str = "kmeans", **algorithm_params: Any) -> None:
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

    # ── sklearn API compliance ──────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        params = {"algorithm": self.algorithm}
        params.update(self.algorithm_params)
        return params

    def set_params(self, **params: Any) -> "ClusteringAdapter":
        if "algorithm" in params:
            self.algorithm = params.pop("algorithm")
        self.algorithm_params.update(params)
        self._inner = None
        return self

    # ── Core interface ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: None = None) -> "ClusteringAdapter":
        self._inner = _make_inner(self.algorithm, self.algorithm_params)

        if self.algorithm in ("kmeans", "gaussian_mixture", "mean_shift"):
            self._inner.fit(X)
            self.labels_ = self._inner.predict(X)
        else:
            # DBSCAN, Agglomerative: fit_predict on training set
            self.labels_ = self._inner.fit_predict(X)

        # Store training data for KNN fallback (DBSCAN / Agglomerative)
        self._X_train = X.copy()

        unique = np.unique(self.labels_[self.labels_ >= 0])
        self.n_clusters_ = int(len(unique))

        # Inertia (KMeans only)
        self.inertia_: float | None = getattr(self._inner, "inertia_", None)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return cluster labels for X.

        For KMeans, GaussianMixture, and MeanShift the native predict() is used.
        For DBSCAN and AgglomerativeClustering, KNN label propagation
        assigns each new point the label of its nearest training point.
        """
        self._check_fitted()
        if self.algorithm in ("kmeans", "gaussian_mixture", "mean_shift"):
            return self._inner.predict(X)
        return self._knn_predict(X)

    def fit_predict(self, X: np.ndarray, y: None = None) -> np.ndarray:
        self.fit(X)
        return self.labels_

    # ── Private helpers ─────────────────────────────────────────────────────

    def _knn_predict(self, X: np.ndarray) -> np.ndarray:
        """1-NN label propagation from training set."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(self._X_train)
        indices = nn.kneighbors(X, return_distance=False).ravel()
        return self.labels_[indices]

    def _check_fitted(self) -> None:
        if not hasattr(self, "_inner") or self._inner is None:
            raise RuntimeError("ClusteringAdapter must be fitted before predict().")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_inner(algorithm: str, params: dict[str, Any]) -> Any:
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans

        defaults = get_default_clustering_params("kmeans")
        return KMeans(**{**defaults, **params}, random_state=42, n_init="auto")
    elif algorithm == "mean_shift":
        from sklearn.cluster import MeanShift, estimate_bandwidth

        defaults = get_default_clustering_params("mean_shift")
        merged = {**defaults, **params}
        # bandwidth=None triggers auto-estimation at fit time via estimate_bandwidth
        return MeanShift(**merged)
    elif algorithm == "dbscan":
        from sklearn.cluster import DBSCAN

        defaults = get_default_clustering_params("dbscan")
        return DBSCAN(**{**defaults, **params})
    elif algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        defaults = get_default_clustering_params("agglomerative")
        return AgglomerativeClustering(**{**defaults, **params})
    elif algorithm == "gaussian_mixture":
        from sklearn.mixture import GaussianMixture

        defaults = get_default_clustering_params("gaussian_mixture")
        return GaussianMixture(**{**defaults, **params}, random_state=42)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")


def get_default_clustering_params(algorithm: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "kmeans": {"n_clusters": 8},
        "mean_shift": {"bandwidth": None, "bin_seeding": False},
        "dbscan": {"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
        "agglomerative": {"n_clusters": 8, "linkage": "ward"},
        "gaussian_mixture": {"n_components": 8, "covariance_type": "full"},
    }
    return defaults.get(algorithm, {})


def get_clustering_search_space(algorithm: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna hyperparameter sample for the given algorithm."""
    if algorithm == "kmeans":
        return {"n_clusters": trial.suggest_int("n_clusters", 2, 20)}
    elif algorithm == "mean_shift":
        # bandwidth=None → sklearn auto-estimates; tune bin_seeding only
        return {"bin_seeding": trial.suggest_categorical("bin_seeding", [True, False])}
    elif algorithm == "dbscan":
        return {
            "eps": trial.suggest_float("eps", 0.1, 2.0),
            "min_samples": trial.suggest_int("min_samples", 3, 20),
        }
    elif algorithm == "agglomerative":
        return {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "linkage": trial.suggest_categorical("linkage", ["ward", "complete", "average"]),
        }
    elif algorithm == "gaussian_mixture":
        return {
            "n_components": trial.suggest_int("n_components", 2, 20),
            "covariance_type": trial.suggest_categorical(
                "covariance_type", ["full", "diag", "tied", "spherical"]
            ),
        }
    return {}
