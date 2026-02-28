"""Model training logic — no Streamlit imports.

Provides functions for training Random Forest, XGBoost, and Neural Network
models with optional Optuna hyperparameter tuning, cross-validation,
MLflow logging, and model persistence.
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
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

def get_default_params(model_type: str, task_type: str) -> dict[str, Any]:
    """Return sensible default hyperparameters for *model_type*."""
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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
    """Instantiate an unfitted estimator."""
    if model_type == "random_forest":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=42, n_jobs=-1, **params)

    elif model_type == "xgboost":
        import xgboost as xgb
        if task_type == "classification":
            return xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
                **params,
            )
        else:
            return xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                **params,
            )

    elif model_type == "neural_network":
        return TorchEstimator(task_type=task_type, **params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics appropriate for *task_type*."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    if task_type == "classification":
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "weighted"
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }
        # ROC AUC — needs probability scores
        if y_proba is not None:
            try:
                if n_classes == 2:
                    # Use probability of the positive class
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted",
                    )
                metrics["roc_auc"] = float(auc)
            except (ValueError, IndexError):
                pass
        return metrics
    else:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Run K-fold cross-validation, returning (scores, mean, std)."""
    if task_type == "classification":
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
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 5,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TrainingResult:
    """Train a single model, evaluate on validation set, and run CV."""
    model_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    start = time.time()

    model = create_sklearn_model(model_type, task_type, params)

    # Fit — neural network gets extra kwargs
    if model_type == "neural_network":
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            progress_callback=progress_callback,
        )
    else:
        model.fit(X_train, y_train)

    training_time = time.time() - start

    # Predictions & metrics
    y_pred = model.predict(X_val)
    y_proba = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
        except Exception:
            pass

    metrics = compute_metrics(y_val, y_pred, task_type, y_proba)

    # Cross-validation (clone to get unfitted estimator)
    try:
        cv_model = clone(model)
        cv_scores, cv_mean, cv_std = run_cross_validation(
            cv_model, X_train, y_train, task_type, cv_folds,
        )
    except Exception:
        cv_scores, cv_mean, cv_std = None, None, None

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
) -> str:
    """Save the fitted model via joblib. Returns the file path."""
    os.makedirs(artifact_dir, exist_ok=True)
    filename = f"{result.model_type}_model.joblib"
    path = os.path.join(artifact_dir, filename)
    joblib.dump(result.model, path)
    result.artifact_path = path
    return path


def load_model(path: str) -> Any:
    """Load a model from a joblib file."""
    return joblib.load(path)


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

    # Determine best model by primary metric
    if results[0].metrics.get("accuracy") is not None:
        primary_metric = "accuracy"
    elif results[0].metrics.get("r2") is not None:
        primary_metric = "r2"
    else:
        primary_metric = list(results[0].metrics.keys())[0]

    best = max(results, key=lambda r: r.metrics.get(primary_metric, 0))

    return {
        "models_trained": len(results),
        "best_model": best.model_name,
        "best_model_type": best.model_type,
        "primary_metric": primary_metric,
        "best_score": best.metrics.get(primary_metric),
        "comparison": comparison,
    }
