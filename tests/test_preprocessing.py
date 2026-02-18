"""Unit tests for src/pipelines/preprocessing.py"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.preprocessing import (
    build_preprocessing_pipeline,
    drop_rows_with_missing,
    extract_datetime_features,
    fit_and_transform,
    generate_smart_defaults,
    get_preprocessing_summary,
    save_pipeline,
    save_processed_data,
    split_data,
    version_with_dvc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_df():
    """DataFrame with numerical, categorical, datetime, and text columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "salary": [50e3, 60e3, 70e3, 80e3, 90e3, 100e3, 110e3, 120e3, 130e3, 140e3],
        "city": ["Paris", "London", "Paris", "Berlin", "London",
                 "Paris", "Berlin", "London", "Paris", "Berlin"],
        "joined": pd.to_datetime([
            "2020-01-01", "2020-06-15", "2021-03-20", "2021-11-10", "2022-05-01",
            "2022-08-12", "2023-01-01", "2023-04-15", "2023-07-20", "2023-11-10",
        ]),
        "notes": [f"long_unique_note_{i}" for i in range(10)],
        "target": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })


@pytest.fixture
def column_types():
    return {
        "age": "numerical",
        "salary": "numerical",
        "city": "categorical",
        "joined": "datetime",
        "notes": "text",
        "target": "categorical",
    }


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        "b": [np.nan, np.nan, 3, 4, 5, 6, 7, 8, 9, 10],
        "c": list(range(1, 11)),
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def classification_df():
    """Larger DataFrame suitable for stratified splitting."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "cat": rng.choice(["x", "y", "z"], n),
        "target": rng.choice(["A", "B"], n, p=[0.6, 0.4]),
    })


@pytest.fixture
def regression_df():
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "target": rng.randn(n),
    })


# ---------------------------------------------------------------------------
# TestGenerateSmartDefaults
# ---------------------------------------------------------------------------

class TestGenerateSmartDefaults:
    def test_excludes_target(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        assert "target" not in config

    def test_numerical_defaults(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        assert config["age"]["imputation"] == "median"
        assert config["age"]["scaling"] == "standard"
        assert config["salary"]["imputation"] == "median"

    def test_categorical_defaults(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        assert config["city"]["imputation"] == "mode"
        assert config["city"]["encoding"] == "onehot"

    def test_datetime_defaults(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        assert config["joined"]["action"] == "drop"

    def test_text_defaults(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        assert config["notes"]["action"] == "drop"

    def test_all_columns_covered(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        expected = set(column_types.keys()) - {"target"}
        assert set(config.keys()) == expected

    def test_correct_types_stored(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        for col, cfg in config.items():
            assert cfg["type"] == column_types[col]


# ---------------------------------------------------------------------------
# TestBuildPreprocessingPipeline
# ---------------------------------------------------------------------------

class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        transformer, kept, dropped = build_preprocessing_pipeline(config, column_types)
        from sklearn.compose import ColumnTransformer
        assert isinstance(transformer, ColumnTransformer)

    def test_dropped_contains_datetime_text(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        _, _, dropped = build_preprocessing_pipeline(config, column_types)
        assert "joined" in dropped
        assert "notes" in dropped

    def test_kept_contains_numerical_categorical(self, column_types):
        config = generate_smart_defaults(column_types, "target")
        _, kept, _ = build_preprocessing_pipeline(config, column_types)
        assert "age" in kept
        assert "salary" in kept
        assert "city" in kept

    def test_custom_config(self):
        config = {
            "a": {"type": "numerical", "imputation": "mean", "scaling": "minmax"},
            "b": {"type": "categorical", "imputation": "mode", "encoding": "label"},
        }
        col_types = {"a": "numerical", "b": "categorical"}
        transformer, kept, dropped = build_preprocessing_pipeline(config, col_types)
        assert "a" in kept
        assert "b" in kept
        assert len(dropped) == 0

    def test_imputation_drop_excluded_from_pipeline(self):
        config = {
            "a": {"type": "numerical", "imputation": "drop", "scaling": "standard"},
            "b": {"type": "numerical", "imputation": "median", "scaling": "standard"},
        }
        col_types = {"a": "numerical", "b": "numerical"}
        _, kept, _ = build_preprocessing_pipeline(config, col_types)
        assert "a" not in kept
        assert "b" in kept


# ---------------------------------------------------------------------------
# TestExtractDatetimeFeatures
# ---------------------------------------------------------------------------

class TestExtractDatetimeFeatures:
    def test_extracts_features(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-15", "2021-06-20", "2022-12-25"]),
            "val": [1, 2, 3],
        })
        result = extract_datetime_features(df, ["dt"])
        assert "dt" not in result.columns
        assert "dt_year" in result.columns
        assert "dt_month" in result.columns
        assert "dt_day" in result.columns
        assert "dt_dayofweek" in result.columns
        assert "val" in result.columns
        assert list(result["dt_year"]) == [2020, 2021, 2022]
        assert list(result["dt_month"]) == [1, 6, 12]

    def test_original_unchanged(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01"]),
        })
        original_cols = list(df.columns)
        _ = extract_datetime_features(df, ["dt"])
        assert list(df.columns) == original_cols  # df not mutated

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"val": [1, 2]})
        result = extract_datetime_features(df, ["nonexistent"])
        assert list(result.columns) == ["val"]

    def test_multiple_datetime_columns(self):
        df = pd.DataFrame({
            "start": pd.to_datetime(["2020-01-01", "2021-06-15"]),
            "end": pd.to_datetime(["2020-12-31", "2021-12-31"]),
        })
        result = extract_datetime_features(df, ["start", "end"])
        assert "start_year" in result.columns
        assert "end_year" in result.columns
        assert "start" not in result.columns
        assert "end" not in result.columns


# ---------------------------------------------------------------------------
# TestDropRowsWithMissing
# ---------------------------------------------------------------------------

class TestDropRowsWithMissing:
    def test_drops_rows(self, df_with_missing):
        result = drop_rows_with_missing(df_with_missing, ["a"])
        assert len(result) == 9
        assert result["a"].isna().sum() == 0

    def test_drops_rows_multiple_columns(self, df_with_missing):
        result = drop_rows_with_missing(df_with_missing, ["a", "b"])
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_empty_columns_no_change(self, df_with_missing):
        result = drop_rows_with_missing(df_with_missing, [])
        assert len(result) == len(df_with_missing)

    def test_index_reset(self, df_with_missing):
        result = drop_rows_with_missing(df_with_missing, ["a"])
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# TestSplitData
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_split_sizes_classification(self, classification_df):
        splits = split_data(
            classification_df, "target",
            test_size=0.2, val_size=0.1,
            task_type="classification",
        )
        total = (
            len(splits["X_train"])
            + len(splits["X_val"])
            + len(splits["X_test"])
        )
        assert total == len(classification_df)

    def test_split_sizes_regression(self, regression_df):
        splits = split_data(
            regression_df, "target",
            test_size=0.2, val_size=0.1,
            task_type="regression",
        )
        total = (
            len(splits["X_train"])
            + len(splits["X_val"])
            + len(splits["X_test"])
        )
        assert total == len(regression_df)

    def test_stratification_classification(self, classification_df):
        splits = split_data(
            classification_df, "target",
            test_size=0.2, val_size=0.1,
            task_type="classification",
        )
        # Check proportions are roughly preserved
        original_ratio = (classification_df["target"] == "A").mean()
        train_ratio = (splits["y_train"] == "A").mean()
        assert abs(train_ratio - original_ratio) < 0.15

    def test_target_not_in_features(self, classification_df):
        splits = split_data(classification_df, "target")
        assert "target" not in splits["X_train"].columns

    def test_all_keys_present(self, classification_df):
        splits = split_data(classification_df, "target")
        expected_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
        assert set(splits.keys()) == expected_keys

    def test_reproducibility(self, classification_df):
        s1 = split_data(classification_df, "target", random_state=42)
        s2 = split_data(classification_df, "target", random_state=42)
        pd.testing.assert_frame_equal(s1["X_train"], s2["X_train"])

    def test_approximate_test_size(self, classification_df):
        splits = split_data(
            classification_df, "target",
            test_size=0.2, val_size=0.1,
        )
        test_frac = len(splits["X_test"]) / len(classification_df)
        assert abs(test_frac - 0.2) < 0.05


# ---------------------------------------------------------------------------
# TestFitAndTransform
# ---------------------------------------------------------------------------

class TestFitAndTransform:
    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame({
            "a": rng.randn(n),
            "b": rng.randn(n) * 10,
            "target": rng.choice([0, 1], n),
        })
        config = {
            "a": {"type": "numerical", "imputation": "median", "scaling": "standard"},
            "b": {"type": "numerical", "imputation": "median", "scaling": "standard"},
        }
        col_types = {"a": "numerical", "b": "numerical"}
        pipe, _, _ = build_preprocessing_pipeline(config, col_types)
        splits = split_data(df, "target", test_size=0.2, val_size=0.1,
                            task_type="classification")

        X_tr, X_v, X_te, names = fit_and_transform(
            pipe, splits["X_train"], splits["X_val"], splits["X_test"],
        )
        assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]
        assert len(names) == X_tr.shape[1]

    def test_no_nans_in_output(self):
        rng = np.random.RandomState(42)
        n = 50
        a_vals = rng.randn(n)
        a_vals[5] = np.nan
        a_vals[15] = np.nan
        df = pd.DataFrame({
            "a": a_vals,
            "b": rng.randn(n) * 10,
            "target": rng.choice([0, 1], n),
        })
        config = {
            "a": {"type": "numerical", "imputation": "median", "scaling": "standard"},
            "b": {"type": "numerical", "imputation": "mean", "scaling": "none"},
        }
        col_types = {"a": "numerical", "b": "numerical"}
        pipe, _, _ = build_preprocessing_pipeline(config, col_types)
        splits = split_data(df, "target", test_size=0.2, val_size=0.1,
                            task_type="classification")

        X_tr, X_v, X_te, _ = fit_and_transform(
            pipe, splits["X_train"], splits["X_val"], splits["X_test"],
        )
        assert not np.isnan(X_tr).any()
        assert not np.isnan(X_te).any()

    def test_feature_names_returned(self):
        df = pd.DataFrame({
            "num": range(20),
            "cat": ["a", "b"] * 10,
            "target": [0, 1] * 10,
        })
        config = {
            "num": {"type": "numerical", "imputation": "mean", "scaling": "standard"},
            "cat": {"type": "categorical", "imputation": "mode", "encoding": "onehot"},
        }
        col_types = {"num": "numerical", "cat": "categorical"}
        pipe, _, _ = build_preprocessing_pipeline(config, col_types)
        splits = split_data(df, "target", test_size=0.2, val_size=0.1,
                            task_type="classification")

        _, _, _, names = fit_and_transform(
            pipe, splits["X_train"], splits["X_val"], splits["X_test"],
        )
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# TestSavePipeline
# ---------------------------------------------------------------------------

class TestSavePipeline:
    def test_saves_joblib(self):
        config = {
            "a": {"type": "numerical", "imputation": "median", "scaling": "standard"},
        }
        pipe, _, _ = build_preprocessing_pipeline(config, {"a": "numerical"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_pipeline(pipe, artifact_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".joblib")

    def test_loaded_pipeline_works(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        config = {
            "a": {"type": "numerical", "imputation": "median", "scaling": "standard"},
        }
        pipe, _, _ = build_preprocessing_pipeline(config, {"a": "numerical"})
        pipe.fit(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_pipeline(pipe, artifact_dir=tmpdir)
            loaded = joblib.load(path)
            result = loaded.transform(df)
            assert result.shape == (5, 1)


# ---------------------------------------------------------------------------
# TestSaveProcessedData
# ---------------------------------------------------------------------------

class TestSaveProcessedData:
    def test_saves_six_csvs(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_val = np.array([[7, 8]])
        X_test = np.array([[9, 10]])
        y_train = pd.Series([0, 1, 0], name="target")
        y_val = pd.Series([1], name="target")
        y_test = pd.Series([0], name="target")
        names = ["f1", "f2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_processed_data(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                names, output_dir=tmpdir,
            )
            assert len(paths) == 6
            for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
                assert name in paths
                assert os.path.exists(paths[name])

    def test_csv_content(self):
        X_train = np.array([[1.0, 2.0]])
        X_val = np.array([[3.0, 4.0]])
        X_test = np.array([[5.0, 6.0]])
        y_train = pd.Series([0], name="target")
        y_val = pd.Series([1], name="target")
        y_test = pd.Series([0], name="target")
        names = ["f1", "f2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_processed_data(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                names, output_dir=tmpdir,
            )
            loaded = pd.read_csv(paths["X_train"])
            assert list(loaded.columns) == ["f1", "f2"]
            assert len(loaded) == 1


# ---------------------------------------------------------------------------
# TestVersionWithDvc
# ---------------------------------------------------------------------------

class TestVersionWithDvc:
    def test_graceful_failure_when_dvc_unavailable(self, tmp_path):
        fake_file = tmp_path / "fake.csv"
        fake_file.write_text("a,b\n1,2\n")
        # DVC may or may not be installed; the function should never raise
        result = version_with_dvc([str(fake_file)])
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestGetPreprocessingSummary
# ---------------------------------------------------------------------------

class TestGetPreprocessingSummary:
    def test_summary_structure(self):
        config = {
            "a": {"type": "numerical", "imputation": "median", "scaling": "standard"},
            "b": {"type": "categorical", "imputation": "mode", "encoding": "onehot"},
        }
        summary = get_preprocessing_summary(
            column_config=config,
            dropped_cols=["c"],
            kept_cols=["a", "b"],
            feature_names=["a", "b_x", "b_y"],
            split_sizes={"train": 70, "val": 10, "test": 20},
        )
        assert summary["total_columns_configured"] == 2
        assert summary["columns_kept"] == 2
        assert summary["columns_dropped"] == 1
        assert summary["features_after"] == 3
        assert summary["split_sizes"]["train"] == 70
        assert "c" in summary["dropped_column_names"]
