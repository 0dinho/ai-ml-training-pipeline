"""Unit tests for src/pipelines/training.py"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.training import (
    TrainingResult,
    compute_metrics,
    create_sklearn_model,
    get_default_params,
    get_search_space,
    get_training_summary,
    load_model,
    log_to_mlflow,
    run_cross_validation,
    run_optuna_tuning,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Small classification dataset (numpy arrays)."""
    rng = np.random.RandomState(42)
    n = 120
    X = rng.randn(n, 5)
    y = rng.choice(["A", "B"], n, p=[0.6, 0.4])
    # 80 train / 20 val / 20 test
    return {
        "X_train": X[:80], "y_train": y[:80],
        "X_val": X[80:100], "y_val": y[80:100],
        "X_test": X[100:], "y_test": y[100:],
    }


@pytest.fixture
def regression_data():
    """Small regression dataset (numpy arrays)."""
    rng = np.random.RandomState(42)
    n = 120
    X = rng.randn(n, 5)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.randn(n) * 0.1
    return {
        "X_train": X[:80], "y_train": y[:80],
        "X_val": X[80:100], "y_val": y[80:100],
        "X_test": X[100:], "y_test": y[100:],
    }


@pytest.fixture
def multiclass_data():
    """Small multiclass classification dataset."""
    rng = np.random.RandomState(42)
    n = 150
    X = rng.randn(n, 4)
    y = rng.choice(["cat", "dog", "fish"], n)
    return {
        "X_train": X[:100], "y_train": y[:100],
        "X_val": X[100:130], "y_val": y[100:130],
        "X_test": X[130:], "y_test": y[130:],
    }


def _make_training_result(model_type="random_forest", task_type="classification", **overrides):
    """Helper to build a minimal TrainingResult for testing."""
    from sklearn.ensemble import RandomForestClassifier
    defaults = {
        "model_name": "Random Forest",
        "model_type": model_type,
        "model": RandomForestClassifier(n_estimators=10, random_state=42),
        "params": {"n_estimators": 10},
        "metrics": {"accuracy": 0.85, "f1": 0.83},
        "cv_scores": np.array([0.8, 0.85, 0.9]),
        "cv_mean": 0.85,
        "cv_std": 0.05,
        "training_time": 1.23,
    }
    defaults.update(overrides)
    return TrainingResult(**defaults)


# ---------------------------------------------------------------------------
# TestGetDefaultParams
# ---------------------------------------------------------------------------

class TestGetDefaultParams:
    def test_random_forest_classification(self):
        params = get_default_params("random_forest", "classification")
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "min_samples_split" in params
        assert "max_features" in params

    def test_random_forest_regression(self):
        params = get_default_params("random_forest", "regression")
        assert "n_estimators" in params

    def test_xgboost(self):
        params = get_default_params("xgboost", "classification")
        assert "learning_rate" in params
        assert "subsample" in params
        assert "colsample_bytree" in params
        assert "gamma" in params

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_default_params("unknown", "classification")

    def test_values_are_reasonable(self):
        params = get_default_params("random_forest", "classification")
        assert 10 <= params["n_estimators"] <= 1000
        assert 1 <= params["max_depth"] <= 100


# ---------------------------------------------------------------------------
# TestCreateSklearnModel
# ---------------------------------------------------------------------------

class TestCreateSklearnModel:
    def test_rf_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        params = get_default_params("random_forest", "classification")
        model = create_sklearn_model("random_forest", "classification", params)
        assert isinstance(model, RandomForestClassifier)

    def test_rf_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        params = get_default_params("random_forest", "regression")
        model = create_sklearn_model("random_forest", "regression", params)
        assert isinstance(model, RandomForestRegressor)

    def test_xgb_classifier(self):
        import xgboost as xgb
        params = get_default_params("xgboost", "classification")
        model = create_sklearn_model("xgboost", "classification", params)
        assert isinstance(model, xgb.XGBClassifier)

    def test_xgb_regressor(self):
        import xgboost as xgb
        params = get_default_params("xgboost", "regression")
        model = create_sklearn_model("xgboost", "regression", params)
        assert isinstance(model, xgb.XGBRegressor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_sklearn_model("unknown", "classification", {})


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_classification_metrics(self):
        y_true = np.array(["A", "B", "A", "B", "A"])
        y_pred = np.array(["A", "B", "A", "A", "A"])
        metrics = compute_metrics(y_true, y_pred, "classification")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] == 0.8

    def test_classification_with_proba(self):
        y_true = np.array(["A", "B", "A", "B", "A"])
        y_pred = np.array(["A", "B", "A", "A", "A"])
        y_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2],
        ])
        metrics = compute_metrics(y_true, y_pred, "classification", y_proba)
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        metrics = compute_metrics(y_true, y_pred, "regression")
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))

    def test_multiclass_classification(self):
        y_true = np.array(["cat", "dog", "fish", "cat", "dog", "fish"])
        y_pred = np.array(["cat", "dog", "cat", "cat", "fish", "fish"])
        metrics = compute_metrics(y_true, y_pred, "classification")
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y_true, y_pred, "regression")
        assert metrics["mse"] == 0.0
        assert metrics["r2"] == 1.0


# ---------------------------------------------------------------------------
# TestRunCrossValidation
# ---------------------------------------------------------------------------

class TestRunCrossValidation:
    def test_classification_cv(self, classification_data):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores, mean, std = run_cross_validation(
            model,
            classification_data["X_train"],
            classification_data["y_train"],
            "classification",
            cv_folds=3,
        )
        assert len(scores) == 3
        assert 0 <= mean <= 1
        assert std >= 0

    def test_regression_cv(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scores, mean, std = run_cross_validation(
            model,
            regression_data["X_train"],
            regression_data["y_train"],
            "regression",
            cv_folds=3,
        )
        assert len(scores) == 3
        assert std >= 0

    def test_cv_folds_match(self, classification_data):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        for folds in [2, 3, 5]:
            scores, _, _ = run_cross_validation(
                model,
                classification_data["X_train"],
                classification_data["y_train"],
                "classification",
                cv_folds=folds,
            )
            assert len(scores) == folds


# ---------------------------------------------------------------------------
# TestTrainModel
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_train_random_forest_classification(self, classification_data):
        params = get_default_params("random_forest", "classification")
        params["n_estimators"] = 10  # speed
        result = train_model(
            "random_forest", "classification", params,
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_val"], classification_data["y_val"],
            cv_folds=3,
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == "Random Forest"
        assert result.model_type == "random_forest"
        assert "accuracy" in result.metrics
        assert result.training_time > 0

    def test_train_xgboost_classification(self, classification_data):
        params = get_default_params("xgboost", "classification")
        params["n_estimators"] = 10
        result = train_model(
            "xgboost", "classification", params,
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_val"], classification_data["y_val"],
            cv_folds=3,
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == "XGBoost"
        assert "accuracy" in result.metrics

    def test_train_random_forest_regression(self, regression_data):
        params = get_default_params("random_forest", "regression")
        params["n_estimators"] = 10
        result = train_model(
            "random_forest", "regression", params,
            regression_data["X_train"], regression_data["y_train"],
            regression_data["X_val"], regression_data["y_val"],
            cv_folds=3,
        )
        assert "mse" in result.metrics
        assert "r2" in result.metrics

    def test_cv_scores_populated(self, classification_data):
        params = {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2,
                  "min_samples_leaf": 1, "max_features": "sqrt"}
        result = train_model(
            "random_forest", "classification", params,
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_val"], classification_data["y_val"],
            cv_folds=3,
        )
        assert result.cv_scores is not None
        assert len(result.cv_scores) == 3
        assert result.cv_mean is not None
        assert result.cv_std is not None



# ---------------------------------------------------------------------------
# TestRunOptunaTuning
# ---------------------------------------------------------------------------

class TestRunOptunaTuning:
    def test_rf_tuning(self, classification_data):
        best_params = run_optuna_tuning(
            "random_forest", "classification",
            classification_data["X_train"], classification_data["y_train"],
            cv_folds=2, n_trials=3, timeout=10,
        )
        assert "n_estimators" in best_params
        assert "max_depth" in best_params

    def test_xgb_tuning(self, classification_data):
        best_params = run_optuna_tuning(
            "xgboost", "classification",
            classification_data["X_train"], classification_data["y_train"],
            cv_folds=2, n_trials=3, timeout=10,
        )
        assert "learning_rate" in best_params
        assert "subsample" in best_params

    def test_callback_called(self, classification_data):
        calls = []
        def cb(trial_num, total, score):
            calls.append((trial_num, total, score))

        run_optuna_tuning(
            "random_forest", "classification",
            classification_data["X_train"], classification_data["y_train"],
            cv_folds=2, n_trials=3, timeout=10,
            callback=cb,
        )
        assert len(calls) == 3


# ---------------------------------------------------------------------------
# TestLogToMlflow
# ---------------------------------------------------------------------------

class TestLogToMlflow:
    def test_graceful_failure_no_server(self):
        """MLflow logging fails gracefully when no server is running."""
        result = _make_training_result()
        run_id = log_to_mlflow(
            result,
            experiment_name="test-exp",
            tracking_uri="http://localhost:99999",
        )
        # Should return None (no server), not raise
        assert run_id is None


# ---------------------------------------------------------------------------
# TestSaveLoadModel
# ---------------------------------------------------------------------------

class TestSaveLoadModel:
    def test_save_creates_file(self):
        result = _make_training_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(result, artifact_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".joblib")
            assert result.artifact_path == path

    def test_load_restores_model(self, classification_data):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(classification_data["X_train"], classification_data["y_train"])

        result = _make_training_result(model=model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(result, artifact_dir=tmpdir)
            loaded = load_model(path)
            assert isinstance(loaded, RandomForestClassifier)
            preds = loaded.predict(classification_data["X_val"])
            expected = model.predict(classification_data["X_val"])
            np.testing.assert_array_equal(preds, expected)

    def test_save_xgboost_model(self, classification_data):
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=10, random_state=42,
                                   eval_metric="logloss")
        model.fit(classification_data["X_train"], classification_data["y_train"])

        result = _make_training_result(model_type="xgboost", model=model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(result, artifact_dir=tmpdir)
            loaded = load_model(path)
            preds = loaded.predict(classification_data["X_val"])
            assert len(preds) == len(classification_data["X_val"])


# ---------------------------------------------------------------------------
# TestGetTrainingSummary
# ---------------------------------------------------------------------------

class TestGetTrainingSummary:
    def test_empty_results(self):
        summary = get_training_summary([])
        assert summary["models_trained"] == 0
        assert summary["best_model"] is None

    def test_single_result(self):
        result = _make_training_result()
        summary = get_training_summary([result])
        assert summary["models_trained"] == 1
        assert summary["best_model"] == "Random Forest"
        assert summary["primary_metric"] == "accuracy"
        assert len(summary["comparison"]) == 1

    def test_multiple_results(self):
        r1 = _make_training_result(
            model_name="Random Forest",
            model_type="random_forest",
            metrics={"accuracy": 0.85, "f1": 0.83},
        )
        r2 = _make_training_result(
            model_name="XGBoost",
            model_type="xgboost",
            metrics={"accuracy": 0.90, "f1": 0.88},
        )
        summary = get_training_summary([r1, r2])
        assert summary["models_trained"] == 2
        assert summary["best_model"] == "XGBoost"
        assert summary["best_score"] == 0.90

    def test_regression_summary(self):
        r1 = _make_training_result(
            model_name="Random Forest",
            metrics={"mse": 0.5, "rmse": 0.707, "mae": 0.4, "r2": 0.85},
        )
        summary = get_training_summary([r1])
        assert summary["primary_metric"] == "r2"
        assert summary["best_score"] == 0.85

    def test_comparison_contains_metrics(self):
        result = _make_training_result(
            metrics={"accuracy": 0.85, "f1": 0.83},
            cv_mean=0.84,
            cv_std=0.03,
        )
        summary = get_training_summary([result])
        comp = summary["comparison"][0]
        assert comp["accuracy"] == 0.85
        assert comp["cv_mean"] == 0.84
        assert comp["cv_std"] == 0.03
        assert comp["training_time"] == 1.23

    def test_artifact_and_mlflow_in_summary(self):
        result = _make_training_result(
            artifact_path="/tmp/model.joblib",
            mlflow_run_id="abc123",
        )
        summary = get_training_summary([result])
        comp = summary["comparison"][0]
        assert comp["artifact_path"] == "/tmp/model.joblib"
        assert comp["mlflow_run_id"] == "abc123"
