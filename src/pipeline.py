"""End-to-end audit orchestration for AuditLens."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data_loader import load_csv, get_profile
from src.model_adapter import SklearnModelAdapter
from src.data_quality import DataQualityAuditor
from src.evaluator import PerformanceEvaluator
from src.explainability import ExplainabilityEngine
from src.fairness import FairnessAuditor
from src.reporter import generate_audit_report, export_report


def audit_uploaded_model(
    model_or_adapter: Any,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    sensitive_cols: Optional[List[str]] = None,
    out_dir: str = "outputs",
    api_key: Optional[str] = None,
    progress: Optional[Any] = None
) -> Dict[str, Any]:
    """Audit a pre-trained uploaded model against an evaluation dataset.

    Does NOT retrain the model.
    """
    def log(msg):
        if progress:
            progress(msg)

    os.makedirs(out_dir, exist_ok=True)
    adapter = model_or_adapter if isinstance(model_or_adapter, SklearnModelAdapter) else SklearnModelAdapter(model_or_adapter)
    meta = adapter.get_metadata()

    # 1. Prepare features and target
    log("Preparing dataset features...")
    all_cols = list(df.columns)
    candidate_features = [c for c in all_cols if c != target_col] if target_col else all_cols

    # Align with model's expected features if specified
    if adapter.feature_names_in:
        missing = [f for f in adapter.feature_names_in if f not in df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch: Uploaded model expects {len(adapter.feature_names_in)} features, "
                f"but {len(missing)} are missing from dataset: {missing[:5]}. "
                f"Expected features: {adapter.feature_names_in}"
            )
        feat_cols = adapter.feature_names_in
    else:
        feat_cols = selected_features or candidate_features
        if adapter.n_features_in is not None and len(feat_cols) != adapter.n_features_in:
            raise ValueError(
                f"Feature count mismatch: Uploaded model expects {adapter.n_features_in} features, "
                f"but dataset provides {len(feat_cols)} features ({feat_cols})."
            )

    X = df[feat_cols]

    y_true = None
    classes = meta.get("classes") or []
    if target_col and target_col in df.columns:
        y_true_raw = df[target_col]
        if meta.get("problem_type") == "regression" or (pd.api.types.is_numeric_dtype(y_true_raw) and y_true_raw.nunique() > 10):
            y_true = y_true_raw.values.astype(float)
        else:
            # Check if classes match
            raw_uniques = set(y_true_raw.dropna().unique())
            if classes and set(classes).issubset(raw_uniques):
                y_true = y_true_raw.values
            elif classes and set(classes) == {0, 1} and len(raw_uniques) == 2:
                # Map binary dataset labels to 0 and 1
                u_sorted = sorted(list(raw_uniques))
                mapping = {u_sorted[0]: 0, u_sorted[1]: 1}
                y_true = y_true_raw.map(mapping).values
            elif classes and set(str(c) for c in classes).issubset(set(str(c) for c in raw_uniques)):
                class_map = {str(c): i for i, c in enumerate(classes)}
                y_true = y_true_raw.astype(str).map(class_map).values
            else:
                y_codes, classes_uniques = pd.factorize(y_true_raw.astype(str))
                y_true = y_codes.astype(int)
                if not classes:
                    classes = [str(c) for c in classes_uniques]

    profile = get_profile(df, target_col)

    # 2. Data quality check
    log("Running data quality audit...")
    data_quality = DataQualityAuditor.audit_quality(df, target_col)

    # 3. Model inference (NO RETRAINING)
    log("Executing model inference...")
    try:
        y_pred = adapter.predict(X)
        y_prob = adapter.predict_proba(X)
    except Exception as e:
        err_msg = str(e)
        if "feature" in err_msg.lower() or "columns" in err_msg.lower():
            expected_info = meta.get('feature_names_in') or f"{meta.get('n_features_in')} features"
            raise ValueError(
                f"Model inference failed due to feature mismatch: {err_msg}\n"
                f"Expected features: {expected_info}. "
                f"Dataset columns: {list(X.columns)}."
            ) from e
        raise

    # 4. Performance evaluation (if target available)
    log("Evaluating model performance...")
    metrics: Dict[str, Any] = {}
    if y_true is not None:
        if meta.get("problem_type") == "regression":
            metrics = PerformanceEvaluator.evaluate_regression(y_true, y_pred)
        else:
            metrics = PerformanceEvaluator.evaluate_classification(y_true, y_pred, y_prob, classes)

    # 5. Explainability (SHAP)
    log("Building SHAP explainability engine...")
    explain_engine = ExplainabilityEngine(adapter, X)
    top_features = None
    shap_img = None
    if explain_engine.status != "Unsupported":
        log("Computing global feature importance...")
        top_features, shap_img = explain_engine.compute_global_importance(X, out_dir=out_dir)

    # 6. Fairness & Group Analysis
    log("Auditing subgroup disparities...")
    sensitive_dict = {}
    if sensitive_cols:
        for c in sensitive_cols:
            if c in df.columns:
                sensitive_dict[c] = df[c].values
    else:
        # Default auto-detect if not explicitly selected
        from src.fairness import SENSITIVE_PATTERNS
        for c in df.columns:
            if c != target_col and any(p in c.lower() for p in ("sex", "gender", "race", "ethnicity", "religion", "marital", "age", "nationality")):
                sensitive_dict[c] = df[c].values

    fairness = {}
    verdict = "N/A"
    worst = None
    if sensitive_dict:
        if meta.get("problem_type") == "regression":
            fairness = FairnessAuditor.audit_regression(y_true, y_pred, sensitive_dict, out_dir)
            verdict = "INFO"
        else:
            fairness = FairnessAuditor.audit_classification(y_true, y_pred, sensitive_dict, out_dir)
            verdict, worst = FairnessAuditor.compute_verdict(fairness)

    # 7. Generate report
    log("Generating comprehensive audit report...")
    context = {
        "model_metadata": meta,
        "profile": profile,
        "data_quality": data_quality,
        "metrics": metrics,
        "top_features": top_features,
        "fairness": fairness,
        "verdict": verdict,
    }
    report_text = generate_audit_report(context, api_key=api_key)
    report_paths = export_report(report_text, os.path.join(out_dir, "audit_report"))

    sample_instances = []
    if explain_engine.status != "Unsupported" and len(X) > 0:
        try:
            sample_rows = X.sample(min(5, len(X)), random_state=1)
            sample_instances = [explain_engine.explain_instance(sample_rows.loc[[idx]]) for idx in sample_rows.index]
        except Exception:
            pass

    log("Audit completed successfully.")
    return {
        "model_adapter": adapter,
        "model_metadata": meta,
        "model_name": meta.get("model_name", adapter.raw_model.__class__.__name__),
        "model": adapter.raw_model,
        "comparison": None,
        "profile": profile,
        "data_quality": data_quality,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_true,
        "X_eval": X,
        "X_test": X,
        "X_test_trans": X,
        "explain_engine": explain_engine,
        "explainer": explain_engine.explainer,
        "explainer_kind": explain_engine.explainer_kind,
        "top_features": top_features,
        "shap_img": shap_img,
        "instances": sample_instances,
        "fairness": fairness,
        "verdict": verdict,
        "worst_disparity": worst,
        "report_text": report_text,
        "report_paths": report_paths,
        "classes": classes,
    }


def run_full_audit(csv_path, target_col=None, sensitive_cols=None,
                   out_dir="outputs", api_key=None, progress=None):
    """Backwards-compatible demo pipeline runner for adult census dataset."""
    from src.model_runner import train_model
    def log(msg):
        if progress: progress(msg)

    log("Loading and profiling demo dataset...")
    X, y, feature_names, target, df_raw, classes, _ = load_csv(csv_path, target_col or "income")
    profile = get_profile(df_raw, target)

    log("Training baseline models for demo...")
    bundle = train_model(X, y)
    model = bundle["model"]
    preprocessor = bundle.get("preprocessor")

    X_train_trans = bundle["X_train_trans"]
    X_test_trans = bundle["X_test_trans"]

    log("Auditing with universal pipeline...")
    from sklearn.pipeline import Pipeline
    full_pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    adapter = SklearnModelAdapter(full_pipe, model_name=bundle["model_name"])

    det = sensitive_cols
    test_raw = df_raw.iloc[bundle["test_positions"]]

    res = audit_uploaded_model(
        adapter,
        test_raw,
        target_col=target,
        sensitive_cols=det,
        out_dir=out_dir,
        api_key=api_key,
        progress=progress
    )

    # Attach backward-compatible keys
    res["comparison"] = bundle["comparison"]
    res["model"] = model
    res["model_name"] = bundle["model_name"]
    res["explainer"] = res["explain_engine"].explainer
    res["explainer_kind"] = res["explain_engine"].explainer_kind
    res["X_test"] = bundle["X_test"]
    res["X_test_trans"] = X_test_trans

    # Build sample instance explanations for demo
    sample_rows = X_test_trans.sample(min(5, len(X_test_trans)), random_state=1)
    from src.explainer import explain_instance
    res["instances"] = [
        explain_instance(res["explainer"], sample_rows.loc[[idx]], model)
        for idx in sample_rows.index
    ]
    return res
