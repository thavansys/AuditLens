"""Comprehensive dataset quality checks for AuditLens."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataQualityAuditor:
    """Audits data quality, identifying missingness, outliers, constant columns, and cardinality."""

    @staticmethod
    def audit_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        n_rows, n_cols = df.shape
        findings: List[Dict[str, Any]] = []
        overall_severity = "LOW"

        def add_finding(category: str, severity: str, message: str, details: Any = None):
            nonlocal overall_severity
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            if severity_rank.get(severity, 1) > severity_rank.get(overall_severity, 1):
                overall_severity = severity
            findings.append({
                "category": category,
                "severity": severity,
                "message": message,
                "details": details or {}
            })

        # 1. Missing values
        missing_series = df.isna().mean()
        high_missing = missing_series[missing_series > 0.3].to_dict()
        any_missing = missing_series[missing_series > 0].to_dict()
        total_missing_pct = float(df.isna().mean().mean() * 100)

        if high_missing:
            add_finding(
                "Missing Values", "HIGH",
                f"{len(high_missing)} column(s) exceed 30% missing values: {list(high_missing.keys())}.",
                high_missing
            )
        elif any_missing:
            add_finding(
                "Missing Values", "MEDIUM",
                f"Dataset contains {total_missing_pct:.2f}% total missing values across {len(any_missing)} column(s).",
                any_missing
            )

        # 2. Duplicate rows
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            dup_pct = (n_duplicates / n_rows) * 100
            sev = "HIGH" if dup_pct > 15 else "MEDIUM"
            add_finding(
                "Duplicates", sev,
                f"Found {n_duplicates} duplicate row(s) ({dup_pct:.1f}% of dataset).",
                {"count": n_duplicates, "pct": round(dup_pct, 2)}
            )

        # 3. Constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            add_finding(
                "Constant Features", "MEDIUM",
                f"{len(constant_cols)} column(s) have zero variance (single constant value): {constant_cols}.",
                constant_cols
            )

        # 4. High-cardinality categoricals
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        high_cardinality = [c for c in cat_cols if df[c].nunique() > 100 or (df[c].nunique() / n_rows > 0.6 and df[c].nunique() > 20)]
        if high_cardinality:
            add_finding(
                "High Cardinality", "MEDIUM",
                f"{len(high_cardinality)} categorical feature(s) possess very high cardinality (possible IDs or text): {high_cardinality}.",
                high_cardinality
            )

        # 5. Infinite values in numerics
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        inf_cols = []
        for c in num_cols:
            if np.isinf(df[c].dropna()).any():
                inf_cols.append(c)
        if inf_cols:
            add_finding(
                "Infinite Values", "HIGH",
                f"Numeric column(s) contain infinite (inf / -inf) values: {inf_cols}.",
                inf_cols
            )

        # 6. Target imbalance (if target provided)
        target_info = {}
        if target_col and target_col in df.columns:
            vc = df[target_col].dropna().value_counts(normalize=True)
            if len(vc) > 1:
                ratio = float(vc.iloc[0] / vc.iloc[-1])
                target_info = {"distribution": vc.to_dict(), "imbalance_ratio": round(ratio, 2)}
                if ratio > 10:
                    add_finding(
                        "Target Imbalance", "HIGH",
                        f"Severe target imbalance detected in '{target_col}' (majority class is {ratio:.1f}x larger than minority).",
                        target_info
                    )
                elif ratio > 4:
                    add_finding(
                        "Target Imbalance", "MEDIUM",
                        f"Moderate target imbalance detected in '{target_col}' ({ratio:.1f}x ratio).",
                        target_info
                    )

        return {
            "overall_severity": overall_severity,
            "total_missing_pct": round(total_missing_pct, 2),
            "n_duplicates": n_duplicates,
            "findings": findings,
            "target_info": target_info,
        }
