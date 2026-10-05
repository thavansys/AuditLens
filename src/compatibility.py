"""Compatibility analysis between uploaded models and datasets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd


class CompatibilityChecker:
    """Evaluates whether an uploaded model and dataset can be audited together."""

    @staticmethod
    def check_compatibility(
        adapter,
        df: pd.DataFrame,
        selected_features: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        meta = adapter.get_metadata()
        all_cols = list(df.columns)
        candidate_cols = [c for c in all_cols if c != target_col] if target_col else all_cols

        features_to_check = selected_features or candidate_cols
        n_dataset_features = len(features_to_check)
        expected_n_features = meta.get("n_features_in")
        expected_names = meta.get("feature_names_in")

        status = "Compatible"
        issues: List[str] = []
        recommendations: List[str] = []

        # 1. Base model support
        if meta.get("status") == "Unsupported":
            return {
                "status": "Incompatible",
                "can_audit": False,
                "issues": [meta.get("status_reason", "Unsupported model interface.")],
                "recommendations": ["Upload a model that implements scikit-learn standard predict() interface."],
                "matching_features": [],
                "missing_features": [],
                "extra_features": [],
            }

        # 2. Check feature names if model specifies them
        matching_features: List[str] = []
        missing_features: List[str] = []
        extra_features: List[str] = []

        if expected_names:
            expected_set = set(expected_names)
            dataset_set = set(features_to_check)
            matching_features = [f for f in expected_names if f in dataset_set]
            missing_features = [f for f in expected_names if f not in dataset_set]
            extra_features = [f for f in features_to_check if f not in expected_set]

            if missing_features:
                status = "Needs configuration"
                issues.append(f"Model expects {len(expected_names)} named features, but {len(missing_features)} are missing from the dataset: {missing_features[:5]}.")
                recommendations.append("Ensure the dataset includes all required feature columns or select matching features.")

        # 3. Check feature counts if count is known
        if expected_n_features is not None:
            if n_dataset_features != expected_n_features:
                if status == "Compatible":
                    status = "Needs configuration"
                issues.append(f"Model expects {expected_n_features} features, but {n_dataset_features} features are selected.")
                recommendations.append(f"Select exactly {expected_n_features} feature columns in the configuration panel.")

        # 4. Check for pipeline with preprocessing
        is_pipeline = meta.get("is_pipeline", False)
        has_non_numeric = any(not pd.api.types.is_numeric_dtype(df[c]) for c in features_to_check)
        if has_non_numeric and not is_pipeline:
            # Standalone estimator receiving raw string/categorical data
            status = "Warning"
            issues.append("Dataset contains non-numeric/categorical columns, and the uploaded model is not a Pipeline with built-in preprocessing.")
            recommendations.append("If this model was trained on one-hot/imputed arrays, upload the full Pipeline(preprocessor, model) or ensure features are pre-encoded.")

        can_audit = (status != "Incompatible") and (len(missing_features) == 0 if expected_names else True)

        return {
            "status": status,
            "can_audit": can_audit,
            "issues": issues,
            "recommendations": recommendations,
            "expected_n_features": expected_n_features,
            "n_selected_features": n_dataset_features,
            "matching_features": matching_features,
            "missing_features": missing_features,
            "extra_features": extra_features,
            "expected_names": expected_names,
        }
