"""Regression model factory functions.

Returns unfitted sklearn-compatible estimators for regression tasks.
RF, XGBoost, and NeuralNet remain in training.py for backward compatibility;
this module adds the remaining model types.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator

REGRESSION_MODELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "elastic_net": "Elastic Net",
    "decision_tree": "Decision Tree",
    "svr": "Support Vector Regressor (RBF)",
    "knn": "k-Nearest Neighbours Regressor",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP Regressor",
}


def get_regression_model(
    model_type: str,
    params: dict[str, Any],
) -> BaseEstimator:
    """Return an unfitted regression estimator."""
    if model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**params)

    elif model_type == "ridge":
        from sklearn.linear_model import Ridge

        defaults = get_default_regression_params("ridge")
        return Ridge(**{**defaults, **params}, random_state=42)

    elif model_type == "lasso":
        from sklearn.linear_model import Lasso

        defaults = get_default_regression_params("lasso")
        return Lasso(**{**defaults, **params}, random_state=42)

    elif model_type == "elastic_net":
        from sklearn.linear_model import ElasticNet

        defaults = get_default_regression_params("elastic_net")
        return ElasticNet(**{**defaults, **params}, random_state=42)

    elif model_type == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor

        defaults = get_default_regression_params("decision_tree")
        return DecisionTreeRegressor(**{**defaults, **params}, random_state=42)

    elif model_type == "svr":
        from sklearn.svm import SVR

        defaults = get_default_regression_params("svr")
        return SVR(**{**defaults, **params})

    elif model_type == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        defaults = get_default_regression_params("knn")
        return KNeighborsRegressor(**{**defaults, **params})

    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = get_default_regression_params("gradient_boosting")
        return GradientBoostingRegressor(**{**defaults, **params}, random_state=42)

    elif model_type == "mlp":
        from sklearn.neural_network import MLPRegressor

        defaults = get_default_regression_params("mlp")
        return MLPRegressor(**{**defaults, **params}, random_state=42)

    else:
        raise ValueError(f"Unknown regression model_type: {model_type!r}")


def get_default_regression_params(model_type: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "linear_regression": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 1.0, "max_iter": 2000},
        "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 2000},
        "decision_tree": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "svr": {"C": 1.0, "kernel": "rbf", "epsilon": 0.1, "gamma": "scale"},
        "knn": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        },
        "mlp": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_iter": 200,
        },
    }
    return defaults.get(model_type, {})


def get_regression_search_space(model_type: str, trial: Any) -> dict[str, Any]:
    """Return an Optuna hyperparameter sample."""
    if model_type == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
    elif model_type == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
    elif model_type == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    elif model_type == "svr":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif model_type == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif model_type == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    elif model_type == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    elif model_type == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        sizes = tuple(
            trial.suggest_int(f"n_units_{i}", 32, 256) for i in range(n_layers)
        )
        return {
            "hidden_layer_sizes": sizes,
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    return {}
