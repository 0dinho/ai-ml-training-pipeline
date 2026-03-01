"""Automated retraining orchestrator.

Loads the processed train/val/test splits from disk (written during Phase 3),
trains a fresh model, compares it against the currently deployed model on the
held-out test set, logs everything to MLflow, and optionally promotes the new
model to the MLflow Model Registry.

Designed to be called from:
  - The Streamlit monitoring page  ("Retrain Now" button)
  - The APScheduler script         (scheduled runs)
  - Any CI/CD or cron job
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from src.pipelines.training import (
    MODEL_DISPLAY_NAMES,
    TrainingResult,
    compute_metrics,
    get_default_params,
    is_classification_type,
    is_supervised,
    log_to_mlflow,
    normalize_task_type,
    save_model,
    train_model,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR: str = "data/processed"
ARTIFACT_DIR: str = "artifacts"
MLFLOW_URI: str = "http://localhost:5001"
MLFLOW_EXPERIMENT: str = "automl-experiments"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrainingResult:
    """Full outcome of a single retraining cycle."""

    model_type: str
    model_name: str
    task_type: str
    new_result: TrainingResult          # complete training outcome (new model)
    new_test_metrics: dict[str, float]  # evaluated on held-out test set
    old_test_metrics: dict[str, float] | None  # benchmark metrics (may be None)
    primary_metric: str
    improved: bool
    metric_delta: float                 # new_score − old_score (positive = better)
    mlflow_run_id: str | None
    artifact_path: str | None
    promoted: bool = False
    promotion_error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary dict for session-state history tables."""
        return {
            "timestamp": self.timestamp,
            "model": self.model_name,
            "task": self.task_type,
            "primary_metric": self.primary_metric,
            "new_score": round(self.new_test_metrics.get(self.primary_metric, 0), 4),
            "old_score": (
                round(self.old_test_metrics.get(self.primary_metric, 0), 4)
                if self.old_test_metrics else None
            ),
            "delta": round(self.metric_delta, 4),
            "improved": self.improved,
            "promoted": self.promoted,
            "mlflow_run_id": self.mlflow_run_id[:8] if self.mlflow_run_id else None,
        }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_splits(
    data_dir: str = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the six processed CSV files written during preprocessing."""
    def _load(name: str) -> np.ndarray:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run the Preprocessing page first."
            )
        df = pd.read_csv(path)
        # y files have a single column; X files may have many
        return df.iloc[:, 0].values if df.shape[1] == 1 else df.values

    X_train = _load("X_train")
    X_val   = _load("X_val")
    X_test  = _load("X_test")
    y_train = _load("y_train")
    y_val   = _load("y_val")
    y_test  = _load("y_test")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _load_current_model(model_type: str, artifact_dir: str = ARTIFACT_DIR) -> Any | None:
    """Return the most recently saved model for *model_type*, or None.

    Checks for both new ModelAdapter format ({model_type}_adapter.joblib) and
    legacy format ({model_type}_model.joblib).
    """
    from src.models.adapter import ModelAdapter

    for filename in (f"{model_type}_adapter.joblib", f"{model_type}_model.joblib"):
        path = os.path.join(artifact_dir, filename)
        if os.path.exists(path):
            try:
                return ModelAdapter.load(path)
            except Exception:
                return None
    return None


def _primary_key(task_type: str) -> str:
    """Return the primary evaluation metric for the given task type."""
    task_type = normalize_task_type(task_type)
    mapping = {
        "binary_classification": "accuracy",
        "multiclass_classification": "accuracy",
        "regression": "r2",
        "clustering": "silhouette",
        "anomaly_detection": "anomaly_ratio",
        "dimensionality_reduction": "explained_variance",
    }
    return mapping.get(task_type, "accuracy")


def _higher_is_better(task_type: str) -> bool:
    """Return True when a higher primary metric value means a better model.

    For anomaly_detection the primary metric is anomaly_ratio, which does not
    follow a simple "higher is better" rule — a stable or lower ratio is
    preferable, so this returns False.
    """
    return normalize_task_type(task_type) != "anomaly_detection"


def _eval_on_test(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray | None,
    task_type: str,
) -> dict[str, float]:
    task_type = normalize_task_type(task_type)

    # Handle ModelAdapter and raw estimators uniformly
    from src.models.adapter import ModelAdapter
    if isinstance(model, ModelAdapter):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if model.is_classification() else None
    else:
        y_pred = model.predict(X_test)
        y_proba = None
        if is_classification_type(task_type) and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

    if is_supervised(task_type) and y_test is not None:
        return compute_metrics(y_test, y_pred, task_type, y_proba)
    else:
        return compute_metrics(None, y_pred, task_type, X=X_test)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_retraining(
    model_type: str,
    task_type: str,
    params: dict[str, Any] | None = None,
    cv_folds: int = 5,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str = MLFLOW_URI,
    registry_name: str | None = None,
    auto_promote: bool = False,
    data_dir: str = DATA_DIR,
    artifact_dir: str = ARTIFACT_DIR,
    log_callback: Callable[[str], None] | None = None,
) -> RetrainingResult:
    """Run a full retraining cycle and return the outcome.

    Steps
    -----
    1. Load saved train / val / test splits from *data_dir*.
    2. Evaluate the currently deployed model on the test set (benchmark).
    3. Train a new model with *params* (defaults if None).
    4. Evaluate the new model on the test set.
    5. Compare primary metrics; decide if the new model is better.
    6. Log the new run to MLflow (tagged as a retrain run).
    7. Save the model artifact to *artifact_dir*.
    8. If *auto_promote* and the new model is better, register it in the
       MLflow Model Registry under *registry_name*.

    Parameters
    ----------
    model_type:
        One of ``'random_forest'``, ``'xgboost'``, ``'neural_network'``.
    task_type:
        ``'classification'`` or ``'regression'``.
    params:
        Hyperparameters for the new model. ``None`` → use ``get_default_params``.
    cv_folds:
        Number of cross-validation folds during training.
    experiment_name / tracking_uri:
        MLflow experiment and server settings.
    registry_name:
        Name under which to register the model (required when *auto_promote*).
    auto_promote:
        If True and the new model outperforms the current one, register it.
    data_dir:
        Directory containing the processed CSV splits.
    artifact_dir:
        Directory for saving joblib model files.
    log_callback:
        Optional ``callable(str)`` for streaming progress messages to the UI.
    """
    def _log(msg: str) -> None:
        print(f"[retraining] {msg}")
        if log_callback:
            log_callback(msg)

    model_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    _log(f"Starting retraining — model={model_type}, task={task_type}")

    task_type = normalize_task_type(task_type)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _log("Loading processed data splits…")
    X_train, X_val, X_test, y_train, y_val, y_test = _load_splits(data_dir)

    # For unsupervised tasks, y arrays contain dummy NaN values — treat as None
    if not is_supervised(task_type):
        y_train, y_val, y_test = None, None, None

    _log(
        f"Loaded — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} rows"
    )

    # ── 2. Benchmark current model ────────────────────────────────────────────
    _log("Loading current model for benchmarking…")
    current_model = _load_current_model(model_type, artifact_dir)
    old_test_metrics: dict[str, float] | None = None
    if current_model is not None:
        try:
            old_test_metrics = _eval_on_test(current_model, X_test, y_test, task_type)
            pk = _primary_key(task_type)
            _log(
                f"Current model — {pk}: "
                f"{old_test_metrics.get(pk, 0):.4f}"
            )
        except Exception as exc:
            _log(f"Could not evaluate current model: {exc}")
    else:
        _log("No current model found — this is the first training run.")

    # ── 3. Train new model ────────────────────────────────────────────────────
    if params is None:
        params = get_default_params(model_type, task_type)
    _log(f"Training {model_name} with params: {params}")

    new_result: TrainingResult = train_model(
        model_type=model_type,
        task_type=task_type,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cv_folds=cv_folds,
    )
    _log(f"Training done in {new_result.training_time:.1f}s — val metrics: {new_result.metrics}")

    # ── 4. Evaluate on test set ───────────────────────────────────────────────
    _log("Evaluating new model on held-out test set…")
    new_test_metrics = _eval_on_test(new_result.model, X_test, y_test, task_type)
    _log(f"Test metrics: {new_test_metrics}")

    # Merge test metrics into result so they appear in MLflow
    new_result.metrics.update({f"test_{k}": v for k, v in new_test_metrics.items()})

    # ── 5. Compare ────────────────────────────────────────────────────────────
    pk = _primary_key(task_type)
    new_score = new_test_metrics.get(pk, 0.0)
    old_score = old_test_metrics.get(pk, 0.0) if old_test_metrics else None
    if task_type == "anomaly_detection":
        # For anomaly detection: stable (within 5pp) or lower ratio counts as improved
        improved = old_score is None or abs(new_score - old_score) < 0.05 or new_score < old_score
    else:
        improved = old_score is None or new_score > old_score
    delta = new_score - (old_score if old_score is not None else 0.0)
    _log(
        f"Comparison — {pk}: new={new_score:.4f}, "
        f"old={f'{old_score:.4f}' if old_score is not None else 'N/A'}, "
        f"Δ={delta:+.4f}, improved={improved}"
    )

    # ── 6. MLflow logging ─────────────────────────────────────────────────────
    _log("Logging to MLflow…")
    original_name = new_result.model_name
    new_result.model_name = f"{original_name} (retrain)"
    run_id = log_to_mlflow(
        new_result,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    new_result.model_name = original_name

    if run_id:
        # Attach comparison metadata to the run
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            client.set_tag(run_id, "retrain", "true")
            client.set_tag(run_id, "improved", str(improved).lower())
            client.log_metric(run_id, f"delta_{pk}", round(delta, 6))
            if old_score is not None:
                client.log_metric(run_id, f"old_{pk}", round(old_score, 6))
        except Exception:
            pass
        _log(f"MLflow run logged: {run_id}")
    else:
        _log("MLflow logging skipped (server unavailable).")

    # ── 7. Save artifact ──────────────────────────────────────────────────────
    artifact_path = save_model(new_result, artifact_dir)
    _log(f"Model saved → {artifact_path}")

    # ── 8. Optional promotion ─────────────────────────────────────────────────
    promoted = False
    promotion_error: str | None = None

    if auto_promote and registry_name and improved:
        _log(f"Auto-promoting to registry as '{registry_name}'…")
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            # Open the existing run to add the sklearn model (makes it registrable)
            with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(
                run_name=f"{model_name} (registry)"
            ) as reg_run:
                if not run_id:
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics(new_test_metrics)
                mlflow.sklearn.log_model(
                    new_result.model,
                    "model",
                    registered_model_name=registry_name,
                )
            promoted = True
            _log(f"Promoted → '{registry_name}'.")
        except Exception as exc:
            promotion_error = str(exc)
            _log(f"Promotion failed: {exc}")

    elif auto_promote and not improved:
        _log("Auto-promotion skipped — new model did not outperform the current one.")
    elif auto_promote and not registry_name:
        _log("Auto-promotion skipped — no registry name provided.")

    new_result.mlflow_run_id = run_id
    new_result.artifact_path = artifact_path

    return RetrainingResult(
        model_type=model_type,
        model_name=model_name,
        task_type=task_type,
        new_result=new_result,
        new_test_metrics=new_test_metrics,
        old_test_metrics=old_test_metrics,
        primary_metric=pk,
        improved=improved,
        metric_delta=delta,
        mlflow_run_id=run_id,
        artifact_path=artifact_path,
        promoted=promoted,
        promotion_error=promotion_error,
    )
