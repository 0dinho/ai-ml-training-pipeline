"""ModelAdapter — single picklable artifact bundling model + task metadata.

Design goals:
- One joblib file per model; the API and Streamlit prediction page load only this.
- Carries enough metadata to build a correct response for all 6 task types.
- Backward-compatible: ModelAdapter.load() transparently wraps old raw estimators.
- MLflow-loggable as a sklearn-flavored artifact (exposes predict()).

Usage:
    adapter = ModelAdapter(
        model=fitted_estimator,
        task_type="clustering",
        model_type="kmeans",
        feature_names=["f1", "f2", ...],
    )
    adapter.save("artifacts/adapter_kmeans.joblib")
    loaded = ModelAdapter.load("artifacts/adapter_kmeans.joblib")
    labels = loaded.predict(X_new)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np

# Task type constants
SUPERVISED_TASK_TYPES = frozenset(
    {"binary_classification", "multiclass_classification", "regression", "classification"}
)
CLASSIFICATION_TASK_TYPES = frozenset(
    {"binary_classification", "multiclass_classification", "classification"}
)
UNSUPERVISED_TASK_TYPES = frozenset(
    {"clustering", "anomaly_detection", "dimensionality_reduction"}
)


@dataclass
class ModelAdapter:
    """Single picklable artifact that bundles model + metadata for inference.

    Fields
    ------
    model:
        The fitted estimator: sklearn estimator, XGBoost model,
        ClusteringAdapter, AnomalyAdapter, or ReductionAdapter.
    task_type:
        One of the 6 canonical task type strings (or legacy "classification").
    model_type:
        Algorithm identifier, e.g. "random_forest", "kmeans", "pca".
    feature_names:
        Post-preprocessing feature names (from fit_and_transform).
    target_column:
        Name of the supervised target. Empty string for unsupervised tasks.
    classes_:
        Class labels for classification tasks. Empty list for other tasks.
    n_components_:
        Number of output dimensions for dimensionality reduction tasks.
    metadata:
        Arbitrary dict for additional info (contamination, n_clusters, etc.).
    """

    model: Any
    task_type: str
    model_type: str
    feature_names: list[str] = field(default_factory=list)
    target_column: str = ""
    classes_: list = field(default_factory=list)
    n_components_: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Task type helpers
    # ------------------------------------------------------------------

    def canonical_task_type(self) -> str:
        """Normalize legacy 'classification' to a canonical type."""
        if self.task_type == "classification":
            if len(self.classes_) == 2:
                return "binary_classification"
            elif len(self.classes_) > 2:
                return "multiclass_classification"
            return "binary_classification"
        return self.task_type

    def is_supervised(self) -> bool:
        return self.task_type in SUPERVISED_TASK_TYPES

    def is_classification(self) -> bool:
        return self.task_type in CLASSIFICATION_TASK_TYPES

    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def is_clustering(self) -> bool:
        return self.task_type == "clustering"

    def is_anomaly(self) -> bool:
        return self.task_type == "anomaly_detection"

    def is_reduction(self) -> bool:
        return self.task_type == "dimensionality_reduction"

    # ------------------------------------------------------------------
    # Sklearn-compatible inference interface
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dispatch to the underlying model's predict().

        - Classification / Regression: model.predict(X) as-is
        - Clustering: cluster labels (int array)
        - Anomaly: 0/1 array (0=normal, 1=anomaly); normalises +1/-1 if raw estimator
        - Reduction: transformed coordinates (2D array)
        """
        if self.is_reduction():
            return self.transform(X)

        raw = self.model.predict(X)

        # Normalise anomaly predictions (+1/-1 → 0/1) when wrapping raw sklearn estimators
        if self.is_anomaly() and not _is_anomaly_adapter(self.model):
            from src.models.anomaly import _normalise_predictions
            return _normalise_predictions(raw)

        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities for classification tasks. None otherwise."""
        if not self.is_classification():
            return None
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into lower-dimensional space (reduction tasks only)."""
        if not self.is_reduction():
            raise RuntimeError(f"transform() not supported for task_type={self.task_type!r}")
        return self.model.transform(X)

    def decision_scores(self, X: np.ndarray) -> np.ndarray | None:
        """Return raw anomaly scores for anomaly detection tasks.

        Lower scores = more anomalous (consistent with AnomalyAdapter convention).
        Returns None if the model does not support scoring or task is not anomaly.
        """
        if not self.is_anomaly():
            return None
        if _is_anomaly_adapter(self.model):
            return self.model.decision_scores(X)
        # Raw sklearn anomaly detector
        if hasattr(self.model, "decision_function"):
            return -self.model.decision_function(X)
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Persist this adapter to a single joblib file. Returns path."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str) -> "ModelAdapter":
        """Load a ModelAdapter from a joblib file.

        If the file contains a raw sklearn estimator (old format),
        it is wrapped transparently via _wrap_legacy().
        """
        obj = joblib.load(path)
        if not isinstance(obj, ModelAdapter):
            return ModelAdapter._wrap_legacy(obj)
        return obj

    @staticmethod
    def _wrap_legacy(estimator: Any) -> "ModelAdapter":
        """Wrap a raw sklearn/XGBoost estimator from an older run."""
        has_proba = hasattr(estimator, "predict_proba")
        task_type = "binary_classification" if has_proba else "regression"
        return ModelAdapter(
            model=estimator,
            task_type=task_type,
            model_type="unknown_legacy",
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _is_anomaly_adapter(model: Any) -> bool:
    """Check if model is an AnomalyAdapter (avoiding circular import)."""
    return type(model).__name__ == "AnomalyAdapter"
