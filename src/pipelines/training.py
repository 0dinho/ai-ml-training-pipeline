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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
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

    model_name: str  # "Random Forest", "XGBoost", "Neural Network"
    model_type: str  # "random_forest", "xgboost", "neural_network"
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
    # Supervised — existing
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
    # Supervised — new classification
    "logistic_regression": "Logistic Regression",
    "naive_bayes_gaussian": "Gaussian Naive Bayes",
    "naive_bayes_bernoulli": "Bernoulli Naive Bayes",
    "svm": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "decision_tree": "Decision Tree",
    "knn": "k-Nearest Neighbours",
    "mlp": "MLP",
    # Supervised — new regression
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
    "autoencoder": "Autoencoder (PyTorch)",
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
    "random_forest", "xgboost", "neural_network", "logistic_regression",
    "naive_bayes_gaussian", "naive_bayes_bernoulli", "svm", "svm_linear",
    "decision_tree", "knn", "mlp",
}
REGRESSION_MODEL_TYPES = {
    "random_forest", "xgboost", "neural_network", "linear_regression",
    "ridge", "lasso", "elastic_net", "decision_tree", "svr", "knn",
    "gradient_boosting", "mlp",
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
    elif model_type == "neural_network":
        return {
            "hidden_layers": [64, 32],
            "activation": "relu",
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "early_stopping_patience": 5,
        }

    # ── New classification models ───────────────────────────────────────
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
    elif model_type == "neural_network":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = [
            trial.suggest_int(f"layer_{i}_size", 16, 256, step=16)
            for i in range(n_layers)
        ]
        return {
            "hidden_layers": hidden_layers,
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "leaky_relu"],
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True,
            ),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": 50,  # fixed during tuning; early stopping handles it
            "early_stopping_patience": 5,
        }

    # ── New supervised models ────────────────────────────────────────────
    elif model_type in CLASSIFICATION_MODEL_TYPES - {"random_forest", "xgboost", "neural_network"}:
        from src.models.classifiers import get_classification_search_space
        return get_classification_search_space(model_type, trial)

    elif model_type in REGRESSION_MODEL_TYPES - {"random_forest", "xgboost", "neural_network"}:
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
# TorchEstimator — sklearn-compatible neural network wrapper
# ---------------------------------------------------------------------------

class TorchEstimator(BaseEstimator):
    """Sklearn-compatible wrapper around a PyTorch feedforward network.

    Works with ``cross_val_score``, ``clone``, and joblib serialisation.
    Automatically uses CUDA when available.
    """

    def __init__(
        self,
        task_type: str = "classification",
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 5,
    ):
        self.task_type = task_type
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

    # ---- internal helpers ------------------------------------------------

    def _get_device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_activation(self):
        import torch.nn as nn
        return {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }[self.activation]

    def _build_network(self, input_dim: int, output_dim: int):
        import torch.nn as nn

        layers: list[nn.Module] = []
        in_features = input_dim
        act_cls = self._get_activation()

        for h in self.hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(act_cls())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_features = h

        layers.append(nn.Linear(in_features, output_dim))
        return nn.Sequential(*layers)

    # ---- sklearn API -----------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> "TorchEstimator":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = self._get_device()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        if self.task_type == "classification":
            # Map labels to contiguous ints
            unique_labels = np.unique(y)
            self.classes_ = unique_labels
            self._label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
            self._idx_to_label = {i: lbl for lbl, i in self._label_to_idx.items()}
            y_int = np.array([self._label_to_idx[v] for v in y])
            y_t = torch.tensor(y_int, dtype=torch.long).to(device)
            output_dim = len(unique_labels)
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()
            y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1).to(device)
            output_dim = 1

        self.net_ = self._build_network(X_t.shape[1], output_dim).to(device)

        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)

        # Optional validation tensors for early stopping
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            if self.task_type == "classification":
                y_val_int = np.array([self._label_to_idx[v] for v in y_val])
                y_val_t = torch.tensor(y_val_int, dtype=torch.long).to(device)
            else:
                y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()
                y_val_t = torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1).to(device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = self.net_(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= len(dataset)

            # Early stopping
            if has_val:
                self.net_.eval()
                with torch.no_grad():
                    val_out = self.net_(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if best_state is not None:
                            self.net_.load_state_dict(best_state)
                        if progress_callback:
                            progress_callback(epoch + 1, self.epochs, epoch_loss)
                        break

            if progress_callback:
                progress_callback(epoch + 1, self.epochs, epoch_loss)

        # Restore best weights if we finished all epochs with val data
        if has_val and best_state is not None and patience_counter < self.early_stopping_patience:
            self.net_.load_state_dict(best_state)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = self._get_device()
        self.net_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = self.net_(X_t)
        if self.task_type == "classification":
            indices = out.argmax(dim=1).cpu().numpy()
            return np.array([self._idx_to_label[i] for i in indices])
        else:
            return out.cpu().numpy().ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        device = self._get_device()
        self.net_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = self.net_(X_t)
            proba = torch.softmax(out, dim=1)
        return proba.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.task_type == "classification":
            return float(np.mean(self.predict(X) == y))
        else:
            from sklearn.metrics import r2_score
            return float(r2_score(y, self.predict(X)))

    def get_params(self, deep: bool = True) -> dict:
        return {
            "task_type": self.task_type,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }

    def set_params(self, **params: Any) -> "TorchEstimator":
        for key, value in params.items():
            setattr(self, key, value)
        return self


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
                use_label_encoder=False,
                **params,
            )
        else:
            return xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)

    elif model_type == "neural_network":
        return TorchEstimator(task_type=task_type, **params)

    # ── New classification models ────────────────────────────────────────
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
    progress_callback: Callable[[int, int, float], None] | None = None,
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
    if model_type == "neural_network" and is_supervised(task_type):
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            progress_callback=progress_callback,
        )
    elif is_supervised(task_type):
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
        elif result.model_type in {"random_forest", "xgboost", "neural_network",
                                    "logistic_regression", "svm", "svm_linear",
                                    "naive_bayes_gaussian", "naive_bayes_bernoulli",
                                    "decision_tree", "knn", "mlp"}:
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
