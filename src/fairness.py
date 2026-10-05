"""Fairness auditing for classification and regression models."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, selection_rate, MetricFrame
)


SENSITIVE_PATTERNS = ("sex", "gender", "race", "ethnicity", "religion",
                      "marital", "age", "nationality")


def detect_sensitive_columns(df, explicit=None):
    if explicit:
        return [c for c in explicit if c in df.columns]
    return [c for c in df.columns if any(p in c.lower() for p in SENSITIVE_PATTERNS)]


def audit_fairness(y_true, y_pred, sensitive, out_dir="outputs"):
    return FairnessAuditor.audit_classification(y_true, y_pred, sensitive, out_dir)


def compute_verdict(audit, low_thresh=0.05, moderate_thresh=0.15):
    return FairnessAuditor.compute_verdict(audit)


class FairnessAuditor:
    """Performs subgroup disparity and performance checks across sensitive features."""

    @staticmethod
    def audit_classification(
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        sensitive_dict: Dict[str, np.ndarray],
        out_dir: str = "outputs"
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        y_pred = np.asarray(y_pred)
        has_ground_truth = y_true is not None
        if has_ground_truth:
            y_true = np.asarray(y_true)

        results: Dict[str, Any] = {}

        for col, groups in sensitive_dict.items():
            g = np.asarray(groups).astype(str)
            col_res: Dict[str, Any] = {
                "task": "classification",
                "has_ground_truth": has_ground_truth,
                "n_groups": len(np.unique(g)),
                "selection_rate_by_group": {},
                "demographic_parity_difference": None,
                "demographic_parity_ratio": None,
                "equalized_odds_difference": None,
                "chart_path": None,
            }

            # Group selection rate (works with or without ground truth)
            try:
                # We can calculate selection rate using y_pred
                uniques = np.unique(y_pred)
                pos_val = uniques[-1]
                y_pred_binary = (y_pred == pos_val).astype(int)

                group_rates = {}
                for group in np.unique(g):
                    mask = (g == group)
                    if np.sum(mask) > 0:
                        group_rates[str(group)] = round(float(np.mean(y_pred_binary[mask])), 4)
                col_res["selection_rate_by_group"] = group_rates

                # Parity difference and ratio from selection rates
                if group_rates:
                    rates = list(group_rates.values())
                    max_r, min_r = max(rates), min(rates)
                    col_res["demographic_parity_difference"] = round(float(max_r - min_r), 4)
                    col_res["demographic_parity_ratio"] = round(float(min_r / max_r), 4) if max_r > 0 else 0.0
            except Exception:
                pass

            # Ground truth metrics (Equalized Odds)
            if has_ground_truth:
                try:
                    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=g)
                    col_res["equalized_odds_difference"] = round(float(eo), 4)
                except Exception:
                    pass

            # Chart
            try:
                sel = col_res["selection_rate_by_group"]
                if sel and len(sel) <= 30:
                    plt.figure(figsize=(7, 4))
                    plt.bar(list(sel.keys()), list(sel.values()), color="#4C78A8")
                    plt.ylabel("Selection Rate (Positive Class)")
                    plt.title(f"Selection Rate by {col}")
                    plt.xticks(rotation=30, ha="right")
                    plt.ylim(0, max(1.0, max(sel.values()) * 1.2 if sel else 1.0))
                    chart = os.path.join(out_dir, f"fairness_{col}.png")
                    plt.tight_layout()
                    plt.savefig(chart, dpi=150)
                    plt.close()
                    col_res["chart_path"] = chart
            except Exception:
                pass

            results[col] = col_res

        return results

    @staticmethod
    def audit_regression(
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        sensitive_dict: Dict[str, np.ndarray],
        out_dir: str = "outputs"
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        y_pred = np.asarray(y_pred, dtype=float)
        has_ground_truth = y_true is not None
        if has_ground_truth:
            y_true = np.asarray(y_true, dtype=float)

        results: Dict[str, Any] = {}

        for col, groups in sensitive_dict.items():
            g = np.asarray(groups).astype(str)
            col_res: Dict[str, Any] = {
                "task": "regression",
                "has_ground_truth": has_ground_truth,
                "n_groups": len(np.unique(g)),
                "mean_prediction_by_group": {},
                "mae_by_group": {},
                "rmse_by_group": {},
                "chart_path": None,
            }

            for group in np.unique(g):
                mask = (g == group)
                if np.sum(mask) > 0:
                    preds_g = y_pred[mask]
                    col_res["mean_prediction_by_group"][str(group)] = round(float(np.mean(preds_g)), 4)
                    if has_ground_truth:
                        y_t_g = y_true[mask]
                        mae_g = np.mean(np.abs(y_t_g - preds_g))
                        rmse_g = np.sqrt(np.mean((y_t_g - preds_g) ** 2))
                        col_res["mae_by_group"][str(group)] = round(float(mae_g), 4)
                        col_res["rmse_by_group"][str(group)] = round(float(rmse_g), 4)

            # Chart
            try:
                means = col_res["mean_prediction_by_group"]
                if means and len(means) <= 30:
                    plt.figure(figsize=(7, 4))
                    plt.bar(list(means.keys()), list(means.values()), color="#59A14F")
                    plt.ylabel("Mean Prediction")
                    plt.title(f"Mean Prediction by {col}")
                    plt.xticks(rotation=30, ha="right")
                    chart = os.path.join(out_dir, f"group_perf_{col}.png")
                    plt.tight_layout()
                    plt.savefig(chart, dpi=150)
                    plt.close()
                    col_res["chart_path"] = chart
            except Exception:
                pass

            results[col] = col_res

        return results

    @staticmethod
    def compute_verdict(fairness_results: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        """Classify overall statistical disparity severity from demographic parity difference."""
        worst = 0.0
        for r in fairness_results.values():
            v = r.get("demographic_parity_difference")
            if v is not None and v == v:
                worst = max(worst, abs(v))
        if worst < 0.05:
            return "LOW", round(worst, 4)
        if worst < 0.15:
            return "MODERATE", round(worst, 4)
        return "HIGH", round(worst, 4)
