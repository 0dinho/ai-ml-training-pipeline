"""Anomaly detection model adapter and factory functions.

Supported algorithms:
    isolation_forest      — IsolationForest
    one_class_svm         — OneClassSVM
    local_outlier_factor  — LocalOutlierFactor (novelty=True for predict on new data)
    elliptic_envelope     — EllipticEnvelope

All adapters normalise to 0=normal / 1=anomaly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

ANOMALY_MODELS: dict[str, str] = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "local_outlier_factor": "Local Outlier Factor",
    "elliptic_envelope": "Elliptic Envelope",
}


class AnomalyAdapter(BaseEstimator):
    """Sklearn-compatible wrapper for anomaly detection algorithms.

    Normalises sklearn's +1 (inlier) / -1 (outlier) output to
    0 (normal) / 1 (anomaly) for consistent downstream handling.

    decision_scores() returns raw anomaly scores where lower values indicate
    more anomalous points (consistent with sklearn's decision_function).
    """

    def __init__(self, algorithm: str = "isolation_forest", **algorithm_params: Any) -> None:
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

    # ── sklearn API compliance ──────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        params = {"algorithm": self.algorithm}
        params.update(self.algorithm_params)
        return params

    def set_params(self, **params: Any) -> "AnomalyAdapter":
        if "algorithm" in params:
            self.algorithm = params.pop("algorithm")
        self.algorithm_params.update(params)
        self._inner = None
        return self

    # ── Core interface ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: None = None) -> "AnomalyAdapter":
        self._inner = _make_inner(self.algorithm, self.algorithm_params)
        self._inner.fit(X)

        # Record training anomaly ratio
        raw_preds = self._inner.predict(X)
        anomaly_mask = raw_preds == -1
        self.anomaly_ratio_train_: float = float(np.mean(anomaly_mask))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0 (normal) / 1 (anomaly) integer array."""
        self._check_fitted()
        raw = self._inner.predict(X)
        return _normalise_predictions(raw)

    def fit_predict(self, X: np.ndarray, y: None = None) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def decision_scores(self, X: np.ndarray) -> np.ndarray | None:
        """Return raw anomaly scores (lower = more anomalous).

        Returns None if the underlying model does not support decision_function.
        Note: scores are negated so that larger → more anomalous (consistent
        with the 0/1 prediction convention).
        """
        self._check_fitted()
        if hasattr(self._inner, "decision_function"):
            # sklearn returns higher = more normal; negate so higher = more anomalous
            return -self._inner.decision_function(X)
        return None

    # ── Private helpers ─────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not hasattr(self, "_inner") or self._inner is None:
            raise RuntimeError("AnomalyAdapter must be fitted before predict().")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_inner(algorithm: str, params: dict[str, Any]) -> Any:
    defaults = get_default_anomaly_params(algorithm)
    merged = {**defaults, **params}

    if algorithm == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(**merged, random_state=42)
    elif algorithm == "one_class_svm":
        from sklearn.svm import OneClassSVM

        return OneClassSVM(**merged)
    elif algorithm == "local_outlier_factor":
        from sklearn.neighbors import LocalOutlierFactor

        # novelty=True is required to call predict() on new data
        merged.setdefault("novelty", True)
        return LocalOutlierFactor(**merged)
    elif algorithm == "elliptic_envelope":
        from sklearn.covariance import EllipticEnvelope

        return EllipticEnvelope(**merged, random_state=42)
    else:
        raise ValueError(f"Unknown anomaly algorithm: {algorithm!r}")


def _normalise_predictions(raw: np.ndarray) -> np.ndarray:
    """Convert sklearn +1/-1 to 0/1 (0=normal, 1=anomaly)."""
    return np.where(raw == -1, 1, 0).astype(int)


def get_default_anomaly_params(algorithm: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "isolation_forest": {"n_estimators": 100, "contamination": "auto", "max_features": 1.0},
        "one_class_svm": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"},
        "local_outlier_factor": {"n_neighbors": 20, "contamination": "auto"},
        "elliptic_envelope": {"contamination": 0.1},
    }
    return defaults.get(algorithm, {})


def get_anomaly_search_space(algorithm: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna hyperparameter sample for the given algorithm."""
    if algorithm == "isolation_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "contamination": trial.suggest_float("contamination", 0.01, 0.3),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    elif algorithm == "one_class_svm":
        return {
            "nu": trial.suggest_float("nu", 0.01, 0.5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif algorithm == "local_outlier_factor":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
            "contamination": trial.suggest_float("contamination", 0.01, 0.3),
        }
    elif algorithm == "elliptic_envelope":
        return {"contamination": trial.suggest_float("contamination", 0.01, 0.4)}
    return {}
