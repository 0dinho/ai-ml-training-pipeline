"""Unit tests for src/utils/data_utils.py"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.utils.data_utils import (
    compute_class_balance,
    compute_correlation_matrix,
    compute_missing_values,
    detect_column_types,
    get_dataset_summary,
    get_numerical_stats,
    load_dataset,
    save_uploaded_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame with mixed column types."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "city": ["Paris", "London", "Paris", "Berlin", "London"],
        "joined": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-03-20", "2021-11-10", "2022-05-01"]),
        "notes": [f"note_{i}" for i in range(5)],
    })


@pytest.fixture
def sample_df_with_missing():
    """DataFrame containing missing values."""
    return pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [np.nan, np.nan, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_csv(self, sample_df):
        buf = io.BytesIO()
        sample_df.to_csv(buf, index=False)
        buf.seek(0)
        result = load_dataset(buf, "data.csv")
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_load_excel(self, sample_df):
        buf = io.BytesIO()
        sample_df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        result = load_dataset(buf, "data.xlsx")
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_load_json(self, sample_df):
        buf = io.BytesIO()
        sample_df.to_json(buf)
        buf.seek(0)
        result = load_dataset(buf, "data.json")
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_unsupported_format(self):
        buf = io.BytesIO(b"fake data")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(buf, "data.parquet")


# ---------------------------------------------------------------------------
# save_uploaded_file
# ---------------------------------------------------------------------------

class TestSaveUploadedFile:
    def test_save_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b"col1,col2\n1,2\n3,4"
            buf = io.BytesIO(content)
            path = save_uploaded_file(buf, "test.csv", raw_dir=tmpdir)
            assert os.path.exists(path)
            assert os.path.basename(path) == "test.csv"
            with open(path, "rb") as f:
                assert f.read() == content

    def test_save_duplicate_adds_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b"col1,col2\n1,2"
            # First save
            buf1 = io.BytesIO(content)
            path1 = save_uploaded_file(buf1, "dup.csv", raw_dir=tmpdir)
            # Second save — should get a timestamp prefix
            buf2 = io.BytesIO(content)
            path2 = save_uploaded_file(buf2, "dup.csv", raw_dir=tmpdir)
            assert path1 != path2
            assert os.path.exists(path2)
            assert "dup_" in os.path.basename(path2)


# ---------------------------------------------------------------------------
# detect_column_types
# ---------------------------------------------------------------------------

class TestDetectColumnTypes:
    def test_numerical(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1.5, 2.5, 3.5]})
        types = detect_column_types(df)
        assert types["x"] == "numerical"
        assert types["y"] == "numerical"

    def test_categorical_low_cardinality(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue"] * 10})
        types = detect_column_types(df)
        assert types["color"] == "categorical"

    def test_text_high_cardinality(self):
        df = pd.DataFrame({"desc": [f"unique_text_{i}" for i in range(100)]})
        types = detect_column_types(df)
        assert types["desc"] == "text"

    def test_datetime_column(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        })
        types = detect_column_types(df)
        assert types["dt"] == "datetime"

    def test_datetime_from_string(self):
        df = pd.DataFrame({
            "dt_str": ["2020-01-01", "2020-06-15", "2021-03-20"],
        })
        types = detect_column_types(df)
        assert types["dt_str"] == "datetime"

    def test_categorical_under_20_unique(self):
        # 15 unique values — should be categorical regardless of cardinality
        df = pd.DataFrame({"cat": [f"val_{i}" for i in range(15)] * 100})
        types = detect_column_types(df)
        assert types["cat"] == "categorical"


# ---------------------------------------------------------------------------
# get_dataset_summary
# ---------------------------------------------------------------------------

class TestGetDatasetSummary:
    def test_summary_keys(self, sample_df):
        col_types = detect_column_types(sample_df)
        summary = get_dataset_summary(sample_df, col_types)
        assert summary["rows"] == 5
        assert summary["columns"] == 5
        assert "total_missing" in summary
        assert "missing_percentage" in summary
        assert "memory_usage_mb" in summary
        assert "duplicate_rows" in summary
        assert "column_type_counts" in summary

    def test_no_missing(self, sample_df):
        col_types = detect_column_types(sample_df)
        summary = get_dataset_summary(sample_df, col_types)
        assert summary["total_missing"] == 0
        assert summary["missing_percentage"] == 0.0

    def test_with_missing(self, sample_df_with_missing):
        col_types = detect_column_types(sample_df_with_missing)
        summary = get_dataset_summary(sample_df_with_missing, col_types)
        assert summary["total_missing"] == 3
        assert summary["rows"] == 5
        assert summary["columns"] == 3


# ---------------------------------------------------------------------------
# compute_missing_values
# ---------------------------------------------------------------------------

class TestComputeMissingValues:
    def test_missing_counts(self, sample_df_with_missing):
        result = compute_missing_values(sample_df_with_missing)
        assert len(result) == 3
        assert list(result.columns) == ["column", "missing_count", "missing_percentage"]
        row_a = result[result["column"] == "a"].iloc[0]
        assert row_a["missing_count"] == 1
        row_b = result[result["column"] == "b"].iloc[0]
        assert row_b["missing_count"] == 2
        row_c = result[result["column"] == "c"].iloc[0]
        assert row_c["missing_count"] == 0

    def test_no_missing(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = compute_missing_values(df)
        assert result["missing_count"].sum() == 0


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrix:
    def test_with_numerical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "c": [1, 1, 1, 1]})
        col_types = {"a": "numerical", "b": "numerical", "c": "numerical"}
        corr = compute_correlation_matrix(df, col_types)
        assert corr is not None
        assert corr.shape == (3, 3)
        assert corr.loc["a", "a"] == pytest.approx(1.0)
        assert corr.loc["a", "b"] == pytest.approx(-1.0)

    def test_less_than_two_numerical(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        col_types = {"a": "numerical", "b": "categorical"}
        assert compute_correlation_matrix(df, col_types) is None

    def test_zero_numerical(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        col_types = {"a": "categorical", "b": "categorical"}
        assert compute_correlation_matrix(df, col_types) is None


# ---------------------------------------------------------------------------
# compute_class_balance
# ---------------------------------------------------------------------------

class TestComputeClassBalance:
    def test_class_balance(self):
        df = pd.DataFrame({"target": ["A", "A", "B", "B", "B", "C"]})
        result = compute_class_balance(df, "target")
        assert len(result) == 3
        assert list(result.columns) == ["class", "count", "percentage"]
        assert result["count"].sum() == 6
        # B should be most frequent
        assert result.iloc[0]["class"] == "B"
        assert result.iloc[0]["count"] == 3

    def test_single_class(self):
        df = pd.DataFrame({"label": ["X"] * 10})
        result = compute_class_balance(df, "label")
        assert len(result) == 1
        assert result.iloc[0]["percentage"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# get_numerical_stats
# ---------------------------------------------------------------------------

class TestGetNumericalStats:
    def test_stats_returned(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        col_types = {"x": "numerical", "y": "numerical"}
        stats = get_numerical_stats(df, col_types)
        assert stats is not None
        assert "x" in stats.index
        assert "y" in stats.index
        assert "mean" in stats.columns

    def test_no_numerical_columns(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        col_types = {"a": "categorical", "b": "categorical"}
        assert get_numerical_stats(df, col_types) is None
