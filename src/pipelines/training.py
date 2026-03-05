"""Model training logic — no Streamlit imports.

Provides functions for training all supported model types across 6 task types:
    binary_classification, multiclass_classification, regression,
    clustering, anomaly_detection, dimensionality_reduction

Legacy task_type value "classification" is silently normalised to
"binary_classification" at every dispatch entry point.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


# ---------------------------------------------------------------------------
# Task type constants & normalization
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


def is_classification_type(task_type: str) -> bool:
    return task_type in CLASSIFICATION_TASK_TYPES


def is_supervised(task_type: str) -> bool:
    return task_type in SUPERVISED_TASK_TYPES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Container for a single model's training outcome."""

    model_name: str  # "Random Forest", "XGBoost", etc.
    model_type: str  # "random_forest", "xgboost", etc.
    model: Any  # fitted estimator
    params: dict[str, Any]  # final hyperparams
    metrics: dict[str, float]  # validation metrics
    cv_scores: np.ndarray | None
    cv_mean: float | None
    cv_std: float | None
    training_time: float
    artifact_path: str | None = None
    mlflow_run_id: str | None = None


MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Supervised — classification
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "naive_bayes_gaussian": "Gaussian Naive Bayes",
    "naive_bayes_bernoulli": "Bernoulli Naive Bayes",
    "svm": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "decision_tree": "Decision Tree",
    "knn": "k-Nearest Neighbours",
    # Supervised — regression
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "elastic_net": "Elastic Net",
    "svr": "SVR (RBF)",
    "gradient_boosting": "Gradient Boosting",
    # Unsupervised — clustering
    "kmeans": "K-Means",
    "mean_shift": "Mean Shift",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative Clustering",
    "gaussian_mixture": "Gaussian Mixture Model",
    # Unsupervised — anomaly detection
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "local_outlier_factor": "Local Outlier Factor",
    "elliptic_envelope": "Elliptic Envelope",
    # Unsupervised — dimensionality reduction
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
    "truncated_svd": "Truncated SVD",
}

# Model family groupings (used for MLflow tagging)
CLUSTERING_MODEL_TYPES = {"kmeans", "dbscan", "agglomerative", "gaussian_mixture"}
ANOMALY_MODEL_TYPES = {"isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"}
REDUCTION_MODEL_TYPES = {"pca", "umap", "tsne", "truncated_svd"}
CLASSIFICATION_MODEL_TYPES = {
    "random_forest", "xgboost", "logistic_regression",
    "naive_bayes_gaussian", "naive_bayes_bernoulli", "svm", "svm_linear",
    "decision_tree", "knn",
}
REGRESSION_MODEL_TYPES = {
    "random_forest", "xgboost", "linear_regression",
    "ridge", "lasso", "elastic_net", "decision_tree", "svr", "knn",
    "gradient_boosting",
}


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

def get_default_params(model_type: str, task_type: str) -> dict[str, Any]:
    """Return sensible default hyperparameters for *model_type*."""
    task_type = normalize_task_type(task_type)

    # ── Legacy supervised models ────────────────────────────────────────
    if model_type == "random_forest":
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
    # ── Classification models ──────────────────────────────────────────
    elif model_type in CLASSIFICATION_MODEL_TYPES:
        from src.models.classifiers import get_default_classification_params
        return get_default_classification_params(model_type)

    # ── New regression models ───────────────────────────────────────────
    elif model_type in REGRESSION_MODEL_TYPES:
        from src.models.regressors import get_default_regression_params
        return get_default_regression_params(model_type)

    # ── Clustering ──────────────────────────────────────────────────────
    elif model_type in CLUSTERING_MODEL_TYPES:
        from src.models.clustering import get_default_clustering_params
        return get_default_clustering_params(model_type)

    # ── Anomaly detection ───────────────────────────────────────────────
    elif model_type in ANOMALY_MODEL_TYPES:
        from src.models.anomaly import get_default_anomaly_params
        return get_default_anomaly_params(model_type)

    # ── Dimensionality reduction ────────────────────────────────────────
    elif model_type in REDUCTION_MODEL_TYPES:
        from src.models.reduction import get_default_reduction_params
        return get_default_reduction_params(model_type)

    else:
        return {}


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------

def get_search_space(model_type: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna-suggested hyperparameter dict for *model_type*."""
    import optuna  # noqa: F401 — only needed when Optuna is actually used

    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None],
            ),
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
    # ── Classification models ────────────────────────────────────────────
    elif model_type in CLASSIFICATION_MODEL_TYPES - {"random_forest", "xgboost"}:
        from src.models.classifiers import get_classification_search_space
        return get_classification_search_space(model_type, trial)

    elif model_type in REGRESSION_MODEL_TYPES - {"random_forest", "xgboost"}:
        from src.models.regressors import get_regression_search_space
        return get_regression_search_space(model_type, trial)

    # ── Unsupervised ─────────────────────────────────────────────────────
    elif model_type in CLUSTERING_MODEL_TYPES:
        from src.models.clustering import get_clustering_search_space
        return get_clustering_search_space(model_type, trial)

    elif model_type in ANOMALY_MODEL_TYPES:
        from src.models.anomaly import get_anomaly_search_space
        return get_anomaly_search_space(model_type, trial)

    elif model_type in REDUCTION_MODEL_TYPES:
        from src.models.reduction import get_reduction_search_space
        return get_reduction_search_space(model_type, trial)

    else:
        return {}


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_sklearn_model(
    model_type: str,
    task_type: str,
    params: dict[str, Any],
) -> BaseEstimator:
    """Instantiate an unfitted estimator for any of the 6 supported task types."""
    task_type = normalize_task_type(task_type)

    # ── Legacy / existing supervised models ─────────────────────────────
    if model_type == "random_forest":
        if is_classification_type(task_type):
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=42, n_jobs=-1, **params)

    elif model_type == "xgboost":
        import xgboost as xgb
        if is_classification_type(task_type):
            return xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                **params,
            )
        else:
            return xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)

    # ── Classification models ────────────────────────────────────────────
    elif task_type in ("binary_classification", "multiclass_classification") and \
            model_type in CLASSIFICATION_MODEL_TYPES:
        from src.models.classifiers import get_classification_model
        return get_classification_model(model_type, task_type, params)

    # ── New regression models ────────────────────────────────────────────
    elif task_type == "regression" and model_type in REGRESSION_MODEL_TYPES:
        from src.models.regressors import get_regression_model
        return get_regression_model(model_type, params)

    # ── Clustering ───────────────────────────────────────────────────────
    elif task_type == "clustering":
        from src.models.clustering import ClusteringAdapter
        return ClusteringAdapter(algorithm=model_type, **params)

    # ── Anomaly detection ────────────────────────────────────────────────
    elif task_type == "anomaly_detection":
        from src.models.anomaly import AnomalyAdapter
        return AnomalyAdapter(algorithm=model_type, **params)

    # ── Dimensionality reduction ─────────────────────────────────────────
    elif task_type == "dimensionality_reduction":
        from src.models.reduction import ReductionAdapter
        n_components = params.pop("n_components", 2)
        return ReductionAdapter(algorithm=model_type, n_components=n_components, **params)

    else:
        raise ValueError(f"Unknown model_type={model_type!r} for task_type={task_type!r}")


# ---------------------------------------------------------------------------
# Metrics computation
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

    Delegates to src.evaluation.metrics.compute_metrics.
    Accepts all existing call signatures (legacy 'classification' normalised).
    """
    from src.evaluation.metrics import compute_metrics as _eval_compute_metrics
    return _eval_compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        task_type=task_type,
        y_proba=y_proba,
        X=X,
        explained_variance_ratio=explained_variance_ratio,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray | None,
    task_type: str,
    cv_folds: int = 5,
) -> tuple[np.ndarray | None, float | None, float | None]:
    """Run K-fold cross-validation, returning (scores, mean, std).

    Returns (None, None, None) for unsupervised task types — CV is not
    applicable without labels.
    """
    task_type = normalize_task_type(task_type)

    if not is_supervised(task_type):
        return None, None, None

    if is_classification_type(task_type):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "r2"

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores, float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def run_optuna_tuning(
    model_type: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_trials: int = 50,
    timeout: int | None = 300,
    callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Run Optuna study and return the best hyperparameters.

    *callback(trial_number, n_trials, best_value)* is called after each trial.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(model_type, trial)
        model = create_sklearn_model(model_type, task_type, params)
        _, mean_score, _ = run_cross_validation(
            model, X_train, y_train, task_type, cv_folds,
        )
        if callback:
            callback(trial.number + 1, n_trials, mean_score)
        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Reconstruct full params dict from best trial
    best_params = get_search_space(model_type, study.best_trial)
    return best_params


# ---------------------------------------------------------------------------
# Full training orchestrator
# ---------------------------------------------------------------------------

def train_model(
    model_type: str,
    task_type: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cv_folds: int = 5,
) -> TrainingResult:
    """Train a single model, evaluate, and run CV (if supervised).

    For unsupervised tasks (clustering, anomaly_detection,
    dimensionality_reduction), y_train / y_val may be None and CV is skipped.
    """
    task_type = normalize_task_type(task_type)
    model_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    start = time.time()

    # Make a copy of params to avoid mutating caller's dict
    params = dict(params)
    model = create_sklearn_model(model_type, task_type, params)

    # ── Fit ─────────────────────────────────────────────────────────────
    if is_supervised(task_type):
        model.fit(X_train, y_train)
    else:
        model.fit(X_train)

    training_time = time.time() - start

    # ── Metrics ──────────────────────────────────────────────────────────
    if task_type in ("binary_classification", "multiclass_classification") and X_val is not None:
        y_pred = model.predict(X_val)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_val)
            except Exception:
                pass
        metrics = compute_metrics(y_val, y_pred, task_type, y_proba)

    elif task_type == "regression" and X_val is not None:
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, task_type)

    elif task_type == "clustering":
        labels = model.predict(X_train)
        metrics = compute_metrics(None, labels, task_type, X=X_train)
        if hasattr(model, "inertia_") and model.inertia_ is not None:
            metrics["inertia"] = float(model.inertia_)

    elif task_type == "anomaly_detection":
        preds = model.predict(X_train)
        metrics = compute_metrics(None, preds, task_type)
        if hasattr(model, "anomaly_ratio_train_"):
            metrics["anomaly_ratio"] = float(model.anomaly_ratio_train_)

    elif task_type == "dimensionality_reduction":
        coords = model.fit_transform(X_train) if hasattr(model, "fit_transform") else model.transform(X_train)
        evr = getattr(model, "explained_variance_ratio_", None)
        metrics = compute_metrics(None, coords, task_type, explained_variance_ratio=evr)

    else:
        metrics = {}

    # ── Cross-validation (supervised only) ──────────────────────────────
    cv_scores, cv_mean, cv_std = None, None, None
    if is_supervised(task_type) and y_train is not None:
        try:
            cv_model = clone(model)
            cv_scores, cv_mean, cv_std = run_cross_validation(
                cv_model, X_train, y_train, task_type, cv_folds,
            )
        except Exception:
            pass

    return TrainingResult(
        model_name=model_name,
        model_type=model_type,
        model=model,
        params=params,
        metrics=metrics,
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        training_time=training_time,
    )


# ---------------------------------------------------------------------------
# MLflow logging (best-effort)
# ---------------------------------------------------------------------------

def log_to_mlflow(
    result: TrainingResult,
    experiment_name: str = "automl-experiments",
    tracking_uri: str = "http://localhost:5001",
) -> str | None:
    """Best-effort MLflow logging. Returns run_id or None on failure."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Determine model family for tagging
        if result.model_type in CLUSTERING_MODEL_TYPES:
            model_family = "clustering"
        elif result.model_type in ANOMALY_MODEL_TYPES:
            model_family = "anomaly_detection"
        elif result.model_type in REDUCTION_MODEL_TYPES:
            model_family = "dimensionality_reduction"
        elif result.model_type in {"random_forest", "xgboost",
                                    "logistic_regression", "svm", "svm_linear",
                                    "naive_bayes_gaussian", "naive_bayes_bernoulli",
                                    "decision_tree", "knn"}:
            model_family = "classification"
        else:
            model_family = "regression"

        with mlflow.start_run(run_name=result.model_name) as run:
            mlflow.log_params(
                {k: str(v) for k, v in result.params.items()},
            )
            mlflow.log_metrics(result.metrics)
            if result.cv_mean is not None:
                mlflow.log_metric("cv_mean", result.cv_mean)
            if result.cv_std is not None:
                mlflow.log_metric("cv_std", result.cv_std)
            mlflow.log_metric("training_time", result.training_time)
            mlflow.set_tags({
                "task_type": normalize_task_type(
                    getattr(result, "task_type", model_family)
                    if hasattr(result, "task_type") else model_family
                ),
                "model_family": model_family,
            })

            if result.artifact_path and os.path.exists(result.artifact_path):
                mlflow.log_artifact(result.artifact_path)

            return run.info.run_id
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    result: TrainingResult,
    artifact_dir: str = "artifacts",
    task_type: str = "",
    feature_names: list[str] | None = None,
    target_column: str = "",
) -> str:
    """Save the fitted model as a ModelAdapter via joblib. Returns the file path."""
    from src.models.adapter import ModelAdapter

    os.makedirs(artifact_dir, exist_ok=True)
    filename = f"{result.model_type}_adapter.joblib"
    path = os.path.join(artifact_dir, filename)

    canonical_task = normalize_task_type(task_type) if task_type else "binary_classification"
    classes = list(getattr(result.model, "classes_", []))

    adapter = ModelAdapter(
        model=result.model,
        task_type=canonical_task,
        model_type=result.model_type,
        feature_names=feature_names or [],
        target_column=target_column,
        classes_=classes,
        n_components_=getattr(result.model, "n_components_", None),
        metadata={k: v for k, v in result.metrics.items()},
    )
    adapter.save(path)
    result.artifact_path = path
    return path


def save_model_legacy(
    result: TrainingResult,
    artifact_dir: str = "artifacts",
) -> str:
    """Save the raw fitted estimator via joblib (legacy format). Returns the file path."""
    os.makedirs(artifact_dir, exist_ok=True)
    filename = f"{result.model_type}_model.joblib"
    path = os.path.join(artifact_dir, filename)
    joblib.dump(result.model, path)
    result.artifact_path = path
    return path


def load_model(path: str) -> Any:
    """Load a model from a joblib file (supports both raw and ModelAdapter)."""
    from src.models.adapter import ModelAdapter
    return ModelAdapter.load(path)


def create_model_adapter(
    result: TrainingResult,
    task_type: str,
    feature_names: list[str] | None = None,
    target_column: str = "",
) -> "Any":
    """Wrap a TrainingResult in a ModelAdapter for downstream use."""
    from src.models.adapter import ModelAdapter

    canonical_task = normalize_task_type(task_type)
    classes = list(getattr(result.model, "classes_", []))

    return ModelAdapter(
        model=result.model,
        task_type=canonical_task,
        model_type=result.model_type,
        feature_names=feature_names or [],
        target_column=target_column,
        classes_=classes,
        n_components_=getattr(result.model, "n_components_", None),
        metadata={k: v for k, v in result.metrics.items()},
    )


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def get_training_summary(results: list[TrainingResult]) -> dict[str, Any]:
    """Aggregate training results for UI / Phase 5 consumption."""
    if not results:
        return {"models_trained": 0, "best_model": None, "comparison": []}

    comparison = []
    for r in results:
        entry: dict[str, Any] = {
            "model_name": r.model_name,
            "model_type": r.model_type,
            "training_time": round(r.training_time, 2),
        }
        entry.update(r.metrics)
        if r.cv_mean is not None:
            entry["cv_mean"] = round(r.cv_mean, 4)
            entry["cv_std"] = round(r.cv_std, 4) if r.cv_std is not None else None
        if r.artifact_path:
            entry["artifact_path"] = r.artifact_path
        if r.mlflow_run_id:
            entry["mlflow_run_id"] = r.mlflow_run_id
        comparison.append(entry)

    # Determine best model by primary metric (task-type-aware)
    sample_metrics = results[0].metrics
    if sample_metrics.get("accuracy") is not None:
        primary_metric = "accuracy"
    elif sample_metrics.get("r2") is not None:
        primary_metric = "r2"
    elif sample_metrics.get("silhouette") is not None:
        primary_metric = "silhouette"
    elif sample_metrics.get("explained_variance") is not None:
        primary_metric = "explained_variance"
    elif sample_metrics.get("anomaly_ratio") is not None:
        # For anomaly: lower ratio isn't necessarily "better" — just pick first
        primary_metric = "anomaly_ratio"
    elif sample_metrics:
        primary_metric = list(sample_metrics.keys())[0]
    else:
        primary_metric = None

    # For anomaly_ratio, lower is generally not "better", so just take first model
    if primary_metric == "anomaly_ratio":
        best = results[0]
    elif primary_metric:
        best = max(results, key=lambda r: r.metrics.get(primary_metric, 0))
    else:
        best = results[0]

    return {
        "models_trained": len(results),
        "best_model": best.model_name,
        "best_model_type": best.model_type,
        "primary_metric": primary_metric,
        "best_score": best.metrics.get(primary_metric),
        "comparison": comparison,
    }
