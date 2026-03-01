"""Dimensionality reduction model adapter and factory functions.

Supported algorithms:
    pca           — PCA (supports transform on new data, inverse_transform)
    umap          — UMAP (supports transform on new data)
    tsne          — t-SNE (fit_transform only; transform raises NotImplementedError)
    truncated_svd — TruncatedSVD (supports transform on new data, inverse_transform)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

REDUCTION_MODELS: dict[str, str] = {
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
    "truncated_svd": "Truncated SVD",
}


class ReductionAdapter(BaseEstimator):
    """Sklearn-compatible wrapper for dimensionality reduction algorithms.

    Provides a consistent fit / transform / fit_transform interface.

    Note on t-SNE:
        t-SNE does not support projecting new data points into an existing
        manifold. Calling transform() on a fitted t-SNE adapter raises
        NotImplementedError. The Prediction page in the Streamlit app
        shows a warning and re-runs the full embedding for batch prediction.
    """

    def __init__(
        self,
        algorithm: str = "pca",
        n_components: int = 2,
        **algorithm_params: Any,
    ) -> None:
        self.algorithm = algorithm
        self.n_components = n_components
        self.algorithm_params = algorithm_params

    # ── sklearn API compliance ──────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        params = {"algorithm": self.algorithm, "n_components": self.n_components}
        params.update(self.algorithm_params)
        return params

    def set_params(self, **params: Any) -> "ReductionAdapter":
        if "algorithm" in params:
            self.algorithm = params.pop("algorithm")
        if "n_components" in params:
            self.n_components = params.pop("n_components")
        self.algorithm_params.update(params)
        self._inner = None
        return self

    # ── Core interface ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: None = None) -> "ReductionAdapter":
        self._inner = _make_inner(self.algorithm, self.n_components, self.algorithm_params)

        if self.algorithm == "tsne":
            # t-SNE: fit_transform stores coordinates but cannot generalize
            self._tsne_coords = self._inner.fit_transform(X)
        else:
            self._inner.fit(X)

        self.n_components_ = self.n_components

        # Explained variance (PCA, TruncatedSVD only)
        self.explained_variance_ratio_: np.ndarray | None = getattr(
            self._inner, "explained_variance_ratio_", None
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into lower-dimensional space.

        Raises NotImplementedError for t-SNE (cannot project new points).
        """
        self._check_fitted()
        if self.algorithm == "tsne":
            raise NotImplementedError(
                "t-SNE does not support projecting new data points into an existing embedding. "
                "Use fit_transform() on the full dataset instead."
            )
        return self._inner.transform(X)

    def fit_transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        self.fit(X)
        if self.algorithm == "tsne":
            return self._tsne_coords
        return self._inner.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray | None:
        """Reconstruct original feature space (PCA and TruncatedSVD only).

        Returns None for algorithms that do not support reconstruction.
        """
        self._check_fitted()
        if hasattr(self._inner, "inverse_transform"):
            return self._inner.inverse_transform(X_reduced)
        return None

    # ── Private helpers ─────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not hasattr(self, "_inner") or self._inner is None:
            raise RuntimeError("ReductionAdapter must be fitted before transform().")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_inner(algorithm: str, n_components: int, params: dict[str, Any]) -> Any:
    defaults = get_default_reduction_params(algorithm)
    merged = {**defaults, **params}
    merged["n_components"] = n_components

    if algorithm == "pca":
        from sklearn.decomposition import PCA

        merged.pop("n_components", None)
        return PCA(n_components=n_components, **merged)
    elif algorithm == "umap":
        import umap

        return umap.UMAP(**merged, random_state=42)
    elif algorithm == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(**merged, random_state=42)
    elif algorithm == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD

        merged.pop("n_components", None)
        return TruncatedSVD(n_components=n_components, **merged, random_state=42)
    else:
        raise ValueError(f"Unknown reduction algorithm: {algorithm!r}")


def get_default_reduction_params(algorithm: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "pca": {"svd_solver": "auto", "whiten": False},
        "umap": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"},
        "tsne": {"perplexity": 30, "learning_rate": "auto", "n_iter": 1000},
        "truncated_svd": {"algorithm": "randomized", "n_iter": 5},
    }
    return defaults.get(algorithm, {})


def get_reduction_search_space(algorithm: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna hyperparameter sample for the given algorithm."""
    if algorithm == "pca":
        return {"n_components": trial.suggest_int("n_components", 2, 20)}
    elif algorithm == "umap":
        return {
            "n_components": trial.suggest_int("n_components", 2, 10),
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
            "min_dist": trial.suggest_float("min_dist", 0.01, 0.5),
        }
    elif algorithm == "tsne":
        return {
            "perplexity": trial.suggest_float("perplexity", 5.0, 50.0),
        }
    elif algorithm == "truncated_svd":
        return {"n_components": trial.suggest_int("n_components", 2, 20)}
    return {}
