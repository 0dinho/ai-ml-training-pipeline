"""Classification model factory functions.

Returns unfitted sklearn-compatible estimators for binary and multiclass
classification tasks. RF and XGBoost remain in training.py for backward
compatibility; this module adds new model types.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

CLASSIFICATION_MODELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "naive_bayes_gaussian": "Gaussian Naive Bayes",
    "naive_bayes_bernoulli": "Bernoulli Naive Bayes",
    "svm": "Support Vector Machine (RBF)",
    "svm_linear": "Support Vector Machine (Linear)",
    "decision_tree": "Decision Tree",
    "knn": "k-Nearest Neighbours",
}


def get_classification_model(
    model_type: str,
    task_type: str,
    params: dict[str, Any],
) -> BaseEstimator:
    """Return an unfitted classification estimator.

    Parameters
    ----------
    model_type:
        One of the keys in CLASSIFICATION_MODELS.
    task_type:
        'binary_classification' or 'multiclass_classification'.
        Used to configure multi_class strategies where needed.
    params:
        Hyperparameters forwarded to the estimator constructor.
    """
    multi = task_type == "multiclass_classification"

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        defaults = get_default_classification_params("logistic_regression")
        p = {**defaults, **params}
        if multi:
            p.setdefault("multi_class", "multinomial")
        return LogisticRegression(**p, random_state=42)

    elif model_type == "naive_bayes_gaussian":
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB(**params)

    elif model_type == "naive_bayes_bernoulli":
        from sklearn.naive_bayes import BernoulliNB

        defaults = get_default_classification_params("naive_bayes_bernoulli")
        return BernoulliNB(**{**defaults, **params})

    elif model_type == "svm":
        from sklearn.svm import SVC

        defaults = get_default_classification_params("svm")
        p = {**defaults, **params}
        p["probability"] = True  # required for predict_proba
        return SVC(**p, random_state=42)

    elif model_type == "svm_linear":
        from sklearn.svm import LinearSVC

        from sklearn.calibration import CalibratedClassifierCV

        defaults = get_default_classification_params("svm_linear")
        inner = LinearSVC(**{**defaults, **params}, random_state=42)
        # Wrap with calibration to expose predict_proba
        return CalibratedClassifierCV(inner, cv=3)

    elif model_type == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        defaults = get_default_classification_params("decision_tree")
        return DecisionTreeClassifier(**{**defaults, **params}, random_state=42)

    elif model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        defaults = get_default_classification_params("knn")
        return KNeighborsClassifier(**{**defaults, **params})

    else:
        raise ValueError(f"Unknown classification model_type: {model_type!r}")


def get_default_classification_params(model_type: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "logistic_regression": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "naive_bayes_gaussian": {},
        "naive_bayes_bernoulli": {"alpha": 1.0},
        "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "svm_linear": {"C": 1.0, "max_iter": 2000},
        "decision_tree": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "knn": {"n_neighbors": 5, "metric": "minkowski", "weights": "uniform"},
    }
    return defaults.get(model_type, {})


def get_classification_search_space(model_type: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna hyperparameter sample."""
    if model_type == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    elif model_type == "svm":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif model_type == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_type == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    return {}
