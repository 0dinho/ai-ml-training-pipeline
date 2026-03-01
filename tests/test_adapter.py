"""Smoke tests for ModelAdapter: wrap, roundtrip, and legacy load."""
from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.models.adapter import ModelAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_X():
    rng = np.random.RandomState(0)
    return rng.randn(50, 4).astype(np.float32)


@pytest.fixture
def binary_y():
    return np.array([0, 1] * 25, dtype=int)


@pytest.fixture
def fitted_rf_classifier(small_X, binary_y):
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(small_X, binary_y)
    return clf


@pytest.fixture
def fitted_rf_regressor(small_X):
    y = small_X[:, 0] * 2 + 1
    reg = RandomForestRegressor(n_estimators=5, random_state=0)
    reg.fit(small_X, y)
    return reg


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------

def test_adapter_classification_construction(fitted_rf_classifier, small_X):
    adapter = ModelAdapter(
        model=fitted_rf_classifier,
        task_type="binary_classification",
        model_type="random_forest",
        feature_names=["a", "b", "c", "d"],
        target_column="label",
        classes_=[0, 1],
    )
    assert adapter.is_classification()
    assert not adapter.is_clustering()
    assert not adapter.is_anomaly()
    assert not adapter.is_reduction()
    assert adapter.canonical_task_type() == "binary_classification"


def test_adapter_predict_classification(fitted_rf_classifier, small_X, binary_y):
    adapter = ModelAdapter(
        model=fitted_rf_classifier,
        task_type="binary_classification",
        model_type="random_forest",
    )
    preds = adapter.predict(small_X)
    assert preds.shape == (50,)
    assert set(preds).issubset({0, 1})


def test_adapter_predict_proba_classification(fitted_rf_classifier, small_X):
    adapter = ModelAdapter(
        model=fitted_rf_classifier,
        task_type="binary_classification",
        model_type="random_forest",
        classes_=[0, 1],
    )
    proba = adapter.predict_proba(small_X)
    assert proba is not None
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_adapter_regression_no_proba(fitted_rf_regressor, small_X):
    adapter = ModelAdapter(
        model=fitted_rf_regressor,
        task_type="regression",
        model_type="random_forest",
    )
    preds = adapter.predict(small_X)
    assert preds.shape == (50,)
    assert adapter.predict_proba(small_X) is None
    assert not adapter.is_classification()


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------

def test_adapter_save_load_roundtrip(fitted_rf_classifier, small_X):
    adapter = ModelAdapter(
        model=fitted_rf_classifier,
        task_type="binary_classification",
        model_type="random_forest",
        feature_names=["a", "b", "c", "d"],
        classes_=[0, 1],
        metadata={"accuracy": 0.9},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "adapter.joblib")
        adapter.save(path)
        loaded = ModelAdapter.load(path)

    assert loaded.task_type == "binary_classification"
    assert loaded.model_type == "random_forest"
    assert loaded.feature_names == ["a", "b", "c", "d"]
    assert loaded.classes_ == [0, 1]
    assert loaded.metadata.get("accuracy") == 0.9

    # Predictions should match
    preds_orig = adapter.predict(small_X)
    preds_loaded = loaded.predict(small_X)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


# ---------------------------------------------------------------------------
# Legacy wrap (_wrap_legacy)
# ---------------------------------------------------------------------------

def test_legacy_wrap_raw_sklearn(fitted_rf_classifier, small_X):
    """Loading a raw sklearn estimator saved without ModelAdapter wraps it transparently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_path = os.path.join(tmpdir, "raw_model.joblib")
        joblib.dump(fitted_rf_classifier, legacy_path)
        loaded = ModelAdapter.load(legacy_path)

    assert isinstance(loaded, ModelAdapter)
    assert loaded.is_classification()
    preds = loaded.predict(small_X)
    assert preds.shape == (50,)


def test_legacy_wrap_sets_binary_classification_for_two_class(fitted_rf_classifier, small_X):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "raw.joblib")
        joblib.dump(fitted_rf_classifier, path)
        loaded = ModelAdapter.load(path)
    assert loaded.canonical_task_type() == "binary_classification"


def test_legacy_wrap_regressor(fitted_rf_regressor, small_X):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "raw_reg.joblib")
        joblib.dump(fitted_rf_regressor, path)
        loaded = ModelAdapter.load(path)
    assert isinstance(loaded, ModelAdapter)
    preds = loaded.predict(small_X)
    assert preds.shape == (50,)


# ---------------------------------------------------------------------------
# canonical_task_type normalization
# ---------------------------------------------------------------------------

def test_canonical_task_type_normalizes_legacy(fitted_rf_classifier):
    adapter = ModelAdapter(
        model=fitted_rf_classifier,
        task_type="classification",  # legacy string
        model_type="random_forest",
    )
    assert adapter.canonical_task_type() == "binary_classification"
