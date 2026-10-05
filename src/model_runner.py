"""Model training, comparison and evaluation."""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier


def build_preprocessor(X):
    """Build a ColumnTransformer for numerical and categorical features.

    Numerical: SimpleImputer(strategy='median')
    Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
    """
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), num_cols)
        )
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(
            ("cat", cat_pipeline, cat_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False
    )
    return preprocessor


def train_model(X, y, test_size=0.25, random_state=42):
    """Train LogisticRegression and XGBClassifier, auto-select the better
    model by 5-fold cross-validated F1, then evaluate on a hold-out set.

    Avoids data leakage by splitting X/y first, then fitting preprocessing
    only on the training split X_tr.
    Returns a bundle dict with model, preprocessor, metrics, comparison, X_test, y_test, y_pred.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    strat = y if len(np.unique(y)) > 1 else None

    # Step 1: train_test_split first (prevents data leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat)

    # Step 2: Build preprocessor and fit ONLY on training data
    preprocessor = build_preprocessor(X_tr)
    preprocessor.fit(X_tr)

    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names_out = [f"f_{i}" for i in range(preprocessor.transform(X_tr).shape[1])]

    X_tr_trans = pd.DataFrame(
        preprocessor.transform(X_tr),
        columns=feature_names_out,
        index=X_tr.index
    )
    X_te_trans = pd.DataFrame(
        preprocessor.transform(X_te),
        columns=feature_names_out,
        index=X_te.index
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBClassifier": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state),
    }

    rows = []
    for name, mdl in candidates.items():
        try:
            acc = float(cross_val_score(mdl, X_tr_trans, y_tr, cv=5, scoring="accuracy").mean())
            f1 = float(cross_val_score(mdl, X_tr_trans, y_tr, cv=5, scoring="f1").mean())
        except Exception:
            acc, f1 = np.nan, np.nan
        rows.append({"model": name, "cv_accuracy": acc, "cv_f1": f1})
    comparison = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)

    best_name = comparison.iloc[0]["model"]
    model = candidates[best_name]
    model.fit(X_tr_trans, y_tr)

    y_pred = model.predict(X_te_trans)
    y_prob = model.predict_proba(X_te_trans)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = evaluate_model(y_te, y_pred, y_prob)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names_out": feature_names_out,
        "model_name": best_name,
        "comparison": comparison,
        "X_train_trans": X_tr_trans.reset_index(drop=True),
        "X_test": X_te.reset_index(drop=True),
        "X_test_trans": X_te_trans.reset_index(drop=True),
        "test_positions": np.asarray(X_te.index),
        "y_test": np.asarray(y_te),
        "y_pred": np.asarray(y_pred),
        "metrics": metrics,
    }


def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics as plain Python types."""
    y_prob = None if y_prob is None else np.asarray(y_prob)
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if y_prob is not None else None,
        "confusion_matrix": np.asarray(confusion_matrix(y_true, y_pred)).tolist(),
    }
    return m


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    return joblib.load(path)
