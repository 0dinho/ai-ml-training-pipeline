"""
Pure-Python preprocessing logic — no Streamlit imports.

Provides smart defaults, sklearn pipeline construction, data splitting,
persistence, and DVC versioning for the preprocessing stage.
"""

import os
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------

def generate_smart_defaults(
    column_types: dict[str, str],
    target_column: str,
) -> dict[str, dict]:
    """Return per-column config dict with sensible defaults.

    Rules:
      - numerical  → imputation=median, scaling=standard
      - categorical → imputation=mode, encoding=onehot
      - datetime   → action=drop
      - text       → action=drop

    The *target_column* is excluded from the config.
    """
    config: dict[str, dict] = {}
    for col, ctype in column_types.items():
        if col == target_column:
            continue
        if ctype == "numerical":
            config[col] = {
                "type": ctype,
                "imputation": "median",
                "scaling": "standard",
            }
        elif ctype == "categorical":
            config[col] = {
                "type": ctype,
                "imputation": "mode",
                "encoding": "onehot",
            }
        elif ctype == "datetime":
            config[col] = {
                "type": ctype,
                "action": "drop",
            }
        else:  # text
            config[col] = {
                "type": ctype,
                "action": "drop",
            }
    return config


# ---------------------------------------------------------------------------
# Datetime feature extraction
# ---------------------------------------------------------------------------

def extract_datetime_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> pd.DataFrame:
    """Extract year/month/day/dayofweek from *datetime_columns*, drop originals."""
    df = df.copy()
    for col in datetime_columns:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df.drop(columns=[col], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Drop rows with missing values
# ---------------------------------------------------------------------------

def drop_rows_with_missing(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Drop rows that contain NaN in any of *columns*."""
    if not columns:
        return df
    return df.dropna(subset=columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    task_type: str = "classification",
) -> dict:
    """Split *df* into train / val / test sets.

    For classification the splits are stratified on the target.
    Returns dict with keys X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if task_type == "classification" else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: train vs val (relative to remaining data)
    relative_val = val_size / (1.0 - test_size)
    stratify_temp = y_temp if task_type == "classification" else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Build sklearn ColumnTransformer
# ---------------------------------------------------------------------------

_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _make_numerical_pipeline(imputation: str, scaling: str) -> Pipeline:
    """Build a Pipeline for a group of numerical columns."""
    steps = []
    strategy = "most_frequent" if imputation == "mode" else imputation
    steps.append(("imputer", SimpleImputer(strategy=strategy)))
    if scaling != "none":
        scaler_cls = _SCALER_MAP.get(scaling)
        if scaler_cls is not None:
            steps.append(("scaler", scaler_cls()))
    return Pipeline(steps)


def _make_categorical_pipeline(
    imputation: str,
    encoding: str,
) -> Pipeline:
    """Build a Pipeline for a group of categorical columns."""
    steps = []
    strategy = "most_frequent" if imputation == "mode" else imputation
    steps.append(("imputer", SimpleImputer(strategy=strategy)))
    if encoding == "onehot":
        from sklearn.preprocessing import OneHotEncoder
        steps.append(("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )))
    elif encoding in ("label", "ordinal"):
        steps.append(("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )))
    elif encoding == "target":
        steps.append(("encoder", TargetEncoder()))
    return Pipeline(steps)


def build_preprocessing_pipeline(
    column_config: dict[str, dict],
    column_types: dict[str, str],
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a ColumnTransformer from the per-column config.

    Columns sharing identical (type, imputation, scaling/encoding) settings
    are grouped into the same sub-pipeline for efficiency.

    Returns (transformer, kept_columns, dropped_columns).
    """
    dropped: list[str] = []
    # key = (type, recipe_tuple) -> list of column names
    groups: dict[tuple, list[str]] = {}

    for col, cfg in column_config.items():
        ctype = cfg["type"]

        # Datetime / text with action=drop
        if ctype in ("datetime", "text") and cfg.get("action") == "drop":
            dropped.append(col)
            continue

        # Columns whose imputation is "drop" are handled before pipeline
        if cfg.get("imputation") == "drop":
            continue

        if ctype == "numerical":
            key = ("numerical", cfg["imputation"], cfg.get("scaling", "none"))
        elif ctype == "categorical":
            key = ("categorical", cfg["imputation"], cfg.get("encoding", "onehot"))
        elif ctype == "datetime" and cfg.get("action") == "extract":
            # datetime extraction happens before pipeline; derived cols are numerical
            continue
        else:
            dropped.append(col)
            continue

        groups.setdefault(key, []).append(col)

    transformers = []
    kept: list[str] = []
    for idx, ((ctype, *recipe), cols) in enumerate(groups.items()):
        name = f"{ctype}_{idx}"
        if ctype == "numerical":
            imp, scl = recipe
            pipe = _make_numerical_pipeline(imp, scl)
        else:
            imp, enc = recipe
            pipe = _make_categorical_pipeline(imp, enc)
        transformers.append((name, pipe, cols))
        kept.extend(cols)

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return transformer, kept, dropped


# ---------------------------------------------------------------------------
# Fit & transform
# ---------------------------------------------------------------------------

def fit_and_transform(
    pipeline: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit on train, transform all splits.

    *y_train* is forwarded to ``pipeline.fit`` to support TargetEncoder.
    Returns (X_train_t, X_val_t, X_test_t, feature_names).
    """
    pipeline.fit(X_train, y_train)

    X_train_t = pipeline.transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    try:
        feature_names = list(pipeline.get_feature_names_out())
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_train_t.shape[1])]

    return X_train_t, X_val_t, X_test_t, feature_names


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pipeline(
    pipeline: ColumnTransformer,
    artifact_dir: str = "artifacts",
) -> str:
    """Persist the fitted pipeline via joblib. Returns the file path."""
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, path)
    return path


def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: str = "data/processed",
) -> dict[str, str]:
    """Save six CSV files to *output_dir*. Returns {name: path} mapping."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    for name, arr in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        df = pd.DataFrame(arr, columns=feature_names)
        p = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(p, index=False)
        paths[name] = p

    for name, series in [
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]:
        p = os.path.join(output_dir, f"{name}.csv")
        series.reset_index(drop=True).to_csv(p, index=False, header=True)
        paths[name] = p

    return paths


# ---------------------------------------------------------------------------
# Schema persistence (consumed by the inference API)
# ---------------------------------------------------------------------------

def save_schema(
    feature_columns: list[str],
    column_types: dict[str, str],
    column_config: dict[str, dict],
    task_type: str,
    target_column: str,
    feature_names: list[str],
    artifact_dir: str = "artifacts",
) -> str:
    """Save the training schema to JSON so the inference API can validate inputs.

    The file is written to *artifact_dir*/schema.json and contains:
      - feature_columns: original feature column names (before preprocessing)
      - column_types:    detected type per column
      - column_config:   per-column preprocessing config
      - datetime_extract_cols: columns whose datetime features were extracted
      - task_type / target_column: task metadata
      - feature_names: post-transformation feature names
    """
    import json

    datetime_extract_cols = [
        col for col, cfg in column_config.items()
        if cfg.get("type") == "datetime" and cfg.get("action") == "extract"
        and col in feature_columns
    ]

    schema: dict = {
        "feature_columns": feature_columns,
        "column_types": {k: v for k, v in column_types.items() if k in feature_columns},
        "column_config": {k: v for k, v in column_config.items() if k in feature_columns},
        "datetime_extract_cols": datetime_extract_cols,
        "task_type": task_type,
        "target_column": target_column,
        "feature_names": feature_names,
    }

    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "schema.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# DVC versioning (best-effort)
# ---------------------------------------------------------------------------

def version_with_dvc(file_paths: list[str]) -> bool:
    """Run ``dvc add`` on each path. Returns False if DVC is unavailable."""
    try:
        for fp in file_paths:
            subprocess.run(
                ["dvc", "add", fp],
                check=True,
                capture_output=True,
                text=True,
            )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def get_preprocessing_summary(
    column_config: dict[str, dict],
    dropped_cols: list[str],
    kept_cols: list[str],
    feature_names: list[str],
    split_sizes: dict[str, int],
) -> dict:
    """Return a summary dict suitable for UI display."""
    return {
        "total_columns_configured": len(column_config),
        "columns_kept": len(kept_cols),
        "columns_dropped": len(dropped_cols),
        "dropped_column_names": dropped_cols,
        "kept_column_names": kept_cols,
        "features_after": len(feature_names),
        "feature_names": feature_names,
        "split_sizes": split_sizes,
    }
