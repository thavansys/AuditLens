"""Data loading and profiling utilities for AuditLens."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def load_csv(path: str, target_col: Optional[str] = None):
    """Load a CSV and return features, encoded target, names and the raw frame.

    Compatible with existing callers.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 1:
        raise ValueError("CSV is empty.")

    target = target_col
    classes = []
    y = None

    if target and target in df.columns:
        y_raw = df[target]
        X = df.drop(columns=[target])
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
            y = y_raw.values
            classes = []
        else:
            y_codes, classes_raw = pd.factorize(y_raw.astype(str))
            y = y_codes.astype(int)
            classes = [str(c) for c in classes_raw]
    else:
        target = target_col
        X = df.copy()

    return X, y, list(X.columns), target, df, classes, {}


def get_profile(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Return a comprehensive profile of the dataset."""
    n_rows, n_cols = df.shape
    num_cols = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    cat_cols = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    mem_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
    missing_pct = round(float(df.isna().mean().mean() * 100), 2)

    class_balance = {}
    if target and target in df.columns:
        vc = df[target].astype(str).value_counts()
        class_balance = {str(k): int(v) for k, v in vc.items()}

    # Identify potential target candidates (low-cardinality or named target/label/y)
    target_candidates = []
    for c in df.columns:
        if c.lower() in ("target", "label", "y", "income", "churn", "default", "class", "outcome", "price"):
            target_candidates.append(c)
        elif df[c].nunique() == 2:
            target_candidates.append(c)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_numeric": int(num_cols),
        "n_categorical": int(cat_cols),
        "memory_mb": mem_mb,
        "missing_pct": missing_pct,
        "target": str(target) if target else None,
        "class_balance": class_balance,
        "target_candidates": list(dict.fromkeys(target_candidates)),
    }
