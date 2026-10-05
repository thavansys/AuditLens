"""Generic, comprehensive audit report generation for AuditLens."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _build_generic_prompt(ctx: Dict[str, Any]) -> str:
    meta = ctx.get("model_metadata", {})
    profile = ctx.get("profile", {})
    metrics = ctx.get("metrics", {})
    top_features = ctx.get("top_features")
    fairness = ctx.get("fairness", {})
    dq = ctx.get("data_quality", {})
    verdict = ctx.get("verdict", "N/A")

    top_lines = "No feature importance available."
    if top_features is not None and hasattr(top_features, "iterrows"):
        top_lines = "\n".join(
            f"- {r['feature']}: importance = {r['mean_abs_shap']:.4f}"
            for _, r in top_features.head(10).iterrows()
        )

    fair_lines = "No sensitive attributes audited."
    if fairness:
        lines = []
        for col, r in fairness.items():
            if r.get("task") == "classification":
                lines.append(
                    f"- {col}: demographic parity diff = {r.get('demographic_parity_difference')}, "
                    f"ratio = {r.get('demographic_parity_ratio')}, equalized odds diff = {r.get('equalized_odds_difference')}, "
                    f"selection rates = {r.get('selection_rate_by_group')}"
                )
            else:
                lines.append(
                    f"- {col} (Regression): mean prediction by group = {r.get('mean_prediction_by_group')}, "
                    f"MAE by group = {r.get('mae_by_group')}"
                )
        fair_lines = "\n".join(lines)

    dq_lines = f"Overall severity: {dq.get('overall_severity', 'LOW')}, Missing: {dq.get('total_missing_pct', 0)}%, Duplicates: {dq.get('n_duplicates', 0)}"

    return f"""You are an expert AI Model Risk & Compliance Auditor. Write a comprehensive, professional one-page audit report in MARKDOWN for the uploaded model and dataset.

Use exactly these sections:
## 1. Executive Summary
## 2. Model & Dataset Information
## 3. Compatibility & Data Quality
## 4. Model Performance Findings
## 5. Explainability & Key Decision Drivers
## 6. Subgroup Fairness & Disparity Analysis
## 7. Risks, Limitations & Recommendations

IMPORTANT GUIDELINES:
- Cite the actual statistics and numbers provided below.
- Do NOT assume or mention Adult Census unless explicitly identified in the dataset metadata.
- Fairness language: Do not state that the model is definitely discriminatory. Use objective phrasing such as "Potential statistical disparity detected" and explain that statistical disparity alone does not establish discrimination or legal non-compliance.
- Keep the language clear, precise, and regulatory-grade.

MODEL METADATA:
Model: {meta.get('model_name', 'Unknown')} ({meta.get('model_class', 'Unknown')})
Problem Type: {meta.get('problem_type', 'Unknown')}
Pipeline: {meta.get('is_pipeline', False)}

DATASET PROFILE:
Rows: {profile.get('n_rows')}, Columns: {profile.get('n_cols')}, Target: {profile.get('target', 'None (Unlabeled)')}

DATA QUALITY:
{dq_lines}

PERFORMANCE METRICS:
{metrics}

DISPARITY VERDICT:
{verdict}

TOP DECISION DRIVERS:
{top_lines}

FAIRNESS FINDINGS:
{fair_lines}
"""


def _generate_generic_template_report(ctx: Dict[str, Any]) -> str:
    meta = ctx.get("model_metadata", {})
    p = ctx.get("profile", {})
    m = ctx.get("metrics", {})
    top_features = ctx.get("top_features")
    fairness = ctx.get("fairness", {})
    dq = ctx.get("data_quality", {})
    verdict = ctx.get("verdict", "N/A")

    top_str = "- No global feature drivers computed."
    if top_features is not None and hasattr(top_features, "iterrows"):
        top_str = "\n".join(
            f"- **{r['feature']}**: mean |SHAP| = {r['mean_abs_shap']:.4f}"
            for _, r in top_features.head(10).iterrows()
        )

    fair_str = "- No sensitive attributes were audited for group differences."
    if fairness:
        lines = []
        for col, r in fairness.items():
            if r.get("task") == "classification":
                lines.append(
                    f"- **{col}**: demographic parity difference = {r.get('demographic_parity_difference')}, "
                    f"parity ratio = {r.get('demographic_parity_ratio')}, "
                    f"equalized odds difference = {r.get('equalized_odds_difference')}, "
                    f"selection rates = {r.get('selection_rate_by_group')}"
                )
            else:
                lines.append(
                    f"- **{col}** (Regression): mean prediction by group = {r.get('mean_prediction_by_group')}, "
                    f"MAE by group = {r.get('mae_by_group')}"
                )
        fair_str = "\n".join(lines)

    dq_findings = "\n".join([f"- **{f['category']}** ({f['severity']}): {f['message']}" for f in dq.get("findings", [])]) or "- No critical data quality defects detected."

    # Performance table
    perf_rows = []
    if m:
        if m.get("task") == "classification":
            perf_rows.append(f"| Accuracy | {m.get('accuracy', 'N/A')} |")
            perf_rows.append(f"| Precision | {m.get('precision', 'N/A')} |")
            perf_rows.append(f"| Recall | {m.get('recall', 'N/A')} |")
            perf_rows.append(f"| F1 | {m.get('f1', 'N/A')} |")
            perf_rows.append(f"| ROC AUC | {m.get('roc_auc', 'N/A')} |")
        elif m.get("task") == "regression":
            perf_rows.append(f"| MAE | {m.get('mae', 'N/A')} |")
            perf_rows.append(f"| RMSE | {m.get('rmse', 'N/A')} |")
            perf_rows.append(f"| R² | {m.get('r2', 'N/A')} |")
            perf_rows.append(f"| MAPE | {m.get('mape', 'N/A')} |")
    perf_table = "\n".join(perf_rows) if perf_rows else "_Ground truth target not provided; performance metrics cannot be evaluated._"

    recs = [
        "- Periodically validate incoming feature distributions against training baseline.",
        "- Perform continuous subgroup performance tracking across sensitive cohorts.",
        "- Ensure human-in-the-loop oversight on high-stakes predictions."
    ]
    if verdict == "HIGH":
        recs.insert(0, "- **Subgroup Disparity Review**: High statistical disparity detected. Conduct legal and domain reviews to determine root cause and business necessity.")

    recs_str = "\n".join(recs)

    return f"""# AuditLens Model Audit Report

> _AuditLens Automated Audit Report. Generated without external LLM dependency._

## 1. Executive Summary
Model **{meta.get('model_name', 'Unknown')}** (`{meta.get('model_class', 'Unknown')}`) was audited on **{p.get('n_rows', 0):,}** evaluation samples with **{p.get('n_cols', 0)}** features (Target: `{p.get('target', 'None')}`).
Statistical Disparity Verdict: **{verdict}**.
*Note: Statistical disparity alone does not establish discrimination or legal non-compliance.*

## 2. Model & Dataset Information
- **Model Architecture**: {meta.get('model_class', 'Unknown')}
- **Module**: {meta.get('module', 'Unknown')}
- **Pipeline Structure**: {meta.get('pipeline_steps', 'Standalone Estimator')}
- **Dataset Dimensions**: {p.get('n_rows', 0):,} rows × {p.get('n_cols', 0)} columns
- **Memory Consumption**: {p.get('memory_mb', 'N/A')} MB

## 3. Compatibility & Data Quality
{dq_findings}

## 4. Model Performance Findings
| Metric | Value |
|---|---|
{perf_table}

## 5. Explainability & Key Decision Drivers
{top_str}

## 6. Subgroup Fairness & Disparity Analysis
{fair_str}

## 7. Risks, Limitations & Recommendations
{recs_str}
"""


def generate_audit_report(context: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """Generate generic audit report using Gemini if key is provided, else fallback to template."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(_build_generic_prompt(context))
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
        except Exception:
            pass
    return _generate_generic_template_report(context)


def export_report(report_text: str, path_base: str) -> Dict[str, str]:
    """Save report to Markdown and Text files."""
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    md_path = path_base + ".md"
    txt_path = path_base + ".txt"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return {"markdown": md_path, "text": txt_path}
