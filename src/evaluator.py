"""Comprehensive model evaluation for classification and regression tasks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)


class PerformanceEvaluator:
    """Evaluates classification or regression model outputs on test/audit data."""

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_classes = len(np.unique(y_true))
        is_binary = n_classes <= 2

        metrics: Dict[str, Any] = {
            "task": "classification",
            "is_binary": is_binary,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average="weighted" if not is_binary else "binary", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average="weighted" if not is_binary else "binary", zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average="weighted" if not is_binary else "binary", zero_division=0)), 4),
            "confusion_matrix": np.asarray(confusion_matrix(y_true, y_pred)).tolist(),
            "roc_auc": None,
            "pr_auc": None,
            "classes": classes or [str(c) for c in np.unique(y_true)],
        }

        # Calculate ROC-AUC & PR-AUC if probabilities are available
        if y_prob is not None:
            try:
                y_prob_arr = np.asarray(y_prob)
                if is_binary:
                    # If 2D with 2 columns, take positive class column
                    prob_pos = y_prob_arr[:, 1] if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2 else y_prob_arr
                    # Make sure y_true is 0/1 for roc_auc
                    uniques = np.unique(y_true)
                    y_true_binary = (y_true == uniques[-1]).astype(int)
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true_binary, prob_pos)), 4)
                    metrics["pr_auc"] = round(float(average_precision_score(y_true_binary, prob_pos)), 4)
                elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == n_classes:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob_arr, multi_class="ovr")), 4)
            except Exception:
                pass

        try:
            metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            metrics["classification_report"] = {}

        return metrics

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        mape = None
        if not np.any(y_true == 0):
            try:
                mape = round(float(mean_absolute_percentage_error(y_true, y_pred)), 4)
            except Exception:
                mape = None

        residuals = y_true - y_pred
        return {
            "task": "regression",
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": mape,
            "residuals_summary": {
                "mean": round(float(np.mean(residuals)), 4),
                "std": round(float(np.std(residuals)), 4),
                "min": round(float(np.min(residuals)), 4),
                "max": round(float(np.max(residuals)), 4),
            }
        }
