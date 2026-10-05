"""SHAP explainability engine."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def build_explainer(model, X_background):
    """Build a SHAP explainer; TreeExplainer for tree models, LinearExplainer for linear,
    KernelExplainer as fallback. Returns (explainer, explainer_kind)."""
    X_df = X_background if isinstance(X_background, pd.DataFrame) else pd.DataFrame(X_background)
    try:
        return shap.TreeExplainer(model), "TreeExplainer"
    except Exception:
        pass

    try:
        sample_bg = shap.sample(X_df, min(100, len(X_df)), random_state=42)
        return shap.LinearExplainer(model, sample_bg), "LinearExplainer"
    except Exception:
        pass

    sample = shap.sample(X_df, min(100, len(X_df)), random_state=42)
    predict_fn = getattr(model, "predict_proba", model.predict)
    return shap.KernelExplainer(predict_fn, sample), "KernelExplainer"


def _class_values(sv):
    """Extract SHAP values for the positive class from an Explanation object."""
    vals = np.asarray(sv.values)
    if vals.ndim == 3:  # (n, features, classes) -> positive class
        vals = vals[:, :, -1]
    return vals


def explain_instance(explainer, row, model):
    """Explain a single instance. `row` is a 1-row DataFrame of preprocessed features.

    Returns dict: prediction, probability, base_value, top_features.
    """
    row_df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
    sv = explainer(row_df)
    vals = _class_values(sv)[0]
    base_arr = np.ravel(np.asarray(sv.base_values))
    base = float(base_arr[-1] if base_arr.size else 0.0)

    prediction = int(model.predict(row_df)[0])
    probability = float(model.predict_proba(row_df)[0, -1]) if hasattr(model, "predict_proba") else None

    order = np.argsort(-np.abs(vals))
    top = []
    cols = list(row_df.columns)
    for i in order[:5]:
        top.append({
            "feature": cols[i],
            "shap_value": round(float(vals[i]), 4),
            "direction": "increases prediction" if vals[i] >= 0 else "decreases prediction",
        })
    return {"prediction": prediction, "probability": probability,
            "base_value": base, "top_features": top}


def global_importance(explainer, X, out_dir="outputs", max_display=10):
    """Compute mean |SHAP| per feature and save a summary plot.
    Returns (DataFrame sorted desc, png_path)."""
    os.makedirs(out_dir, exist_ok=True)
    Xs = X.sample(min(300, len(X)), random_state=42) if len(X) > 300 else X
    sv = explainer(Xs)
    vals = _class_values(sv)
    mean_abs = np.abs(vals).mean(axis=0)

    df = (pd.DataFrame({"feature": list(Xs.columns), "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

    plt.figure(figsize=(8, 5))
    shap.summary_plot(vals, Xs, feature_names=list(Xs.columns),
                      show=False, max_display=max_display)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return df, path
