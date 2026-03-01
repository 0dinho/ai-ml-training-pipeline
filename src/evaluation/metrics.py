"""Unified metrics computation for all 6 supported task types.

Task types:
    binary_classification / multiclass_classification / classification (legacy)
    regression
    clustering
    anomaly_detection
    dimensionality_reduction
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Task type constants (also re-exported here for convenience)
# ---------------------------------------------------------------------------

SUPERVISED_TASK_TYPES = frozenset(
    {"binary_classification", "multiclass_classification", "regression", "classification"}
)
CLASSIFICATION_TASK_TYPES = frozenset(
    {"binary_classification", "multiclass_classification", "classification"}
)
UNSUPERVISED_TASK_TYPES = frozenset(
    {"clustering", "anomaly_detection", "dimensionality_reduction"}
)


def normalize_task_type(task_type: str) -> str:
    """Normalize legacy 'classification' → 'binary_classification'."""
    if task_type == "classification":
        return "binary_classification"
    return task_type


def is_classification(task_type: str) -> bool:
    return task_type in CLASSIFICATION_TASK_TYPES


def is_supervised(task_type: str) -> bool:
    return task_type in SUPERVISED_TASK_TYPES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    task_type: str,
    y_proba: np.ndarray | None = None,
    X: np.ndarray | None = None,
    explained_variance_ratio: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for any of the 6 supported task types.

    Parameters
    ----------
    y_true:
        Ground-truth labels (None for unsupervised tasks unless labeled
        reference data is available for anomaly detection).
    y_pred:
        Model predictions. For classification: integer class labels.
        For regression: continuous values. For clustering: integer cluster
        labels. For anomaly detection: 0/1 (0=normal, 1=anomaly).
        For dimensionality reduction: 2-D array of reduced coordinates.
    task_type:
        One of the 6 canonical task type strings (or legacy "classification").
    y_proba:
        Class probabilities for classification tasks (shape [n, n_classes]).
    X:
        Original feature matrix; required for clustering metrics
        (silhouette, Davies-Bouldin, Calinski-Harabasz).
    explained_variance_ratio:
        Per-component explained variance from PCA / TruncatedSVD.

    Returns
    -------
    dict mapping metric name → float value.
    """
    task_type = normalize_task_type(task_type)

    if is_classification(task_type):
        return _classification_metrics(y_true, y_pred, task_type, y_proba)
    elif task_type == "regression":
        return _regression_metrics(y_true, y_pred)
    elif task_type == "clustering":
        return _clustering_metrics(y_pred, X)
    elif task_type == "anomaly_detection":
        return _anomaly_metrics(y_pred, y_true)
    elif task_type == "dimensionality_reduction":
        return _reduction_metrics(y_pred, explained_variance_ratio)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    binary = task_type in ("binary_classification", "classification")
    avg = "binary" if binary else "weighted"

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            if binary:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass

    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    n = len(y_true)
    # Count predictors from y_pred shape — we don't have X here, so we can't
    # compute adjusted R2 without p. Store raw metrics; callers can compute adj-R2.
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
    }


def _clustering_metrics(
    labels: np.ndarray,
    X: np.ndarray | None,
) -> dict[str, float]:
    """Compute clustering quality metrics.

    Requires X (original feature matrix) for silhouette, Davies-Bouldin,
    and Calinski-Harabasz. If X is None, only n_clusters is returned.
    """
    unique = np.unique(labels[labels >= 0])  # exclude DBSCAN noise (-1)
    metrics: dict[str, float] = {"n_clusters": float(len(unique))}

    if X is None or len(unique) < 2:
        return metrics

    # Remove DBSCAN noise points from metric computation
    mask = labels >= 0
    X_clean, labels_clean = X[mask], labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return metrics

    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        metrics["silhouette"] = float(silhouette_score(X_clean, labels_clean))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_clean, labels_clean))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_clean, labels_clean))
    except Exception:
        pass

    return metrics


def _anomaly_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
) -> dict[str, float]:
    """Compute anomaly detection metrics.

    y_pred should be 0/1 (0=normal, 1=anomaly).
    If y_true (ground truth labels) is provided, also computes
    supervised precision/recall/f1.
    """
    n = len(y_pred)
    n_anomalies = int(np.sum(y_pred == 1))
    metrics: dict[str, float] = {
        "anomaly_ratio": float(n_anomalies / n) if n > 0 else 0.0,
        "n_anomalies": float(n_anomalies),
    }

    if y_true is not None:
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score

            metrics["precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, zero_division=0)
            )
        except Exception:
            pass

    return metrics


def _reduction_metrics(
    coords: np.ndarray,
    explained_variance_ratio: np.ndarray | None,
) -> dict[str, float]:
    """Metrics for dimensionality reduction tasks."""
    n_components = coords.shape[1] if coords.ndim == 2 else 1
    metrics: dict[str, float] = {"n_components": float(n_components)}

    if explained_variance_ratio is not None and len(explained_variance_ratio) > 0:
        metrics["explained_variance"] = float(np.sum(explained_variance_ratio))
        for i, v in enumerate(explained_variance_ratio):
            metrics[f"explained_variance_pc{i + 1}"] = float(v)

    return metrics
