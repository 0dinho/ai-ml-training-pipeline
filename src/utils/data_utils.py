"""
Pure-Python utility functions for dataset loading, exploration, and analysis.
No Streamlit imports — independently testable.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_dataset(file_buffer, filename: str) -> pd.DataFrame:
    """Read CSV/Excel/JSON into a DataFrame based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_buffer)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(file_buffer, engine="openpyxl")
    elif ext == ".json":
        return pd.read_json(file_buffer)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use CSV, Excel, or JSON.")


def save_uploaded_file(file_buffer, filename: str, raw_dir: str = "data/raw") -> str:
    """Save the uploaded file to raw_dir. Timestamp-prefix if duplicate."""
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, filename)
    if os.path.exists(dest):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        dest = os.path.join(raw_dir, f"{name}_{ts}{ext}")
    file_buffer.seek(0)
    with open(dest, "wb") as f:
        f.write(file_buffer.read())
    return dest


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Classify each column as numerical, categorical, datetime, or text.

    Heuristic for object columns:
      - Try datetime parsing first.
      - If <= 20 unique values OR cardinality < 5% of rows -> categorical.
      - Otherwise -> text.
    """
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        else:
            # Attempt datetime parsing
            try:
                pd.to_datetime(df[col])
                col_types[col] = "datetime"
                continue
            except (ValueError, TypeError):
                pass

            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique <= 20 or (n_rows > 0 and n_unique / n_rows < 0.05):
                col_types[col] = "categorical"
            else:
                col_types[col] = "text"
    return col_types


def get_dataset_summary(df: pd.DataFrame, column_types: dict[str, str]) -> dict:
    """Return summary dict: row/col counts, missing stats, memory usage, duplicate count."""
    total_missing = int(df.isnull().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_missing": total_missing,
        "missing_percentage": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_type_counts": {
            t: sum(1 for v in column_types.values() if v == t)
            for t in sorted(set(column_types.values()))
        },
    }


def compute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing-value counts and percentages per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2) if len(df) > 0 else missing_count * 0
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.values,
    })


def compute_correlation_matrix(
    df: pd.DataFrame, column_types: dict[str, str]
) -> pd.DataFrame | None:
    """Pearson correlation for numerical columns. None if fewer than 2 numerical cols."""
    num_cols = [c for c, t in column_types.items() if t == "numerical" and c in df.columns]
    if len(num_cols) < 2:
        return None
    return df[num_cols].corr()


def compute_class_balance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Value counts + percentages for the target column."""
    counts = df[target_col].value_counts()
    pcts = (counts / len(df) * 100).round(2) if len(df) > 0 else counts * 0
    return pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "percentage": pcts.values,
    })


def get_numerical_stats(df: pd.DataFrame, column_types: dict[str, str]) -> pd.DataFrame | None:
    """Transposed describe() for numerical columns. None if no numerical columns."""
    num_cols = [c for c, t in column_types.items() if t == "numerical" and c in df.columns]
    if not num_cols:
        return None
    return df[num_cols].describe().T


def detect_task_type(df: pd.DataFrame, target_col: str | None) -> str:
    """Auto-suggest one of the 6 canonical task types based on target column properties.

    Rules:
      - No target column selected  → "clustering" (default unsupervised)
      - Target is non-numeric      → classify by unique count
      - Target is numeric + 2 unique values → "binary_classification"
      - Target is numeric + 3–20 unique values → heuristic: classification vs regression
      - Target is numeric + >20 unique values → "regression"

    Returns one of:
        "binary_classification", "multiclass_classification", "regression",
        "clustering"
    """
    if not target_col or target_col not in df.columns:
        return "clustering"

    series = df[target_col].dropna()
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if n_unique == 2:
        return "binary_classification"
    elif not is_numeric:
        if n_unique <= 20:
            return "multiclass_classification"
        return "multiclass_classification"  # text target → treat as multiclass
    elif n_unique <= 20:
        # Numeric with few unique values → likely classification
        return "multiclass_classification"
    else:
        return "regression"
