"""Universal explainability engine with Tree, Linear, and Kernel fallbacks."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline


class ExplainabilityEngine:
    """Manages model explainability and handles model-specific adapters gracefully."""

    def __init__(self, model_or_adapter: Any, background_data: pd.DataFrame):
        self.raw_model = getattr(model_or_adapter, "raw_model", model_or_adapter)
        self.background_data = background_data
        self.explainer = None
        self.explainer_kind = "None"
        self.status = "Unsupported"
        self.status_reason = ""
        self.preprocessor = None
        self.effective_estimator = self.raw_model
        self.feature_names_out: Optional[List[str]] = None

        self._inspect_pipeline_stages()
        self._init_explainer()

    def _inspect_pipeline_stages(self):
        """Unpack pipeline steps if model is a sklearn Pipeline."""
        if isinstance(self.raw_model, Pipeline) and len(self.raw_model.steps) > 1:
            self.preprocessor = Pipeline(self.raw_model.steps[:-1])
            self.effective_estimator = self.raw_model.steps[-1][1]
        else:
            self.preprocessor = None
            self.effective_estimator = self.raw_model

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features through preprocessing pipeline if present."""
        if self.preprocessor is None:
            return X

        try:
            X_trans = self.preprocessor.transform(X)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()

            if self.feature_names_out is None:
                if hasattr(self.preprocessor, "get_feature_names_out"):
                    try:
                        self.feature_names_out = list(self.preprocessor.get_feature_names_out())
                    except Exception:
                        self.feature_names_out = [f"f_{i}" for i in range(X_trans.shape[1])]
                else:
                    self.feature_names_out = [f"f_{i}" for i in range(X_trans.shape[1])]

            cols = self.feature_names_out if len(self.feature_names_out) == X_trans.shape[1] else [f"f_{i}" for i in range(X_trans.shape[1])]
            return pd.DataFrame(X_trans, columns=cols, index=getattr(X, "index", None))
        except Exception:
            return X

    def _init_explainer(self):
        m = self.effective_estimator
        bg_raw = self.background_data
        sample_size = min(100, len(bg_raw))
        sample_bg_raw = shap.sample(bg_raw, sample_size, random_state=42) if len(bg_raw) > sample_size else bg_raw
        sample_bg = self._transform_features(sample_bg_raw)

        # 1. Attempt TreeExplainer on the effective estimator
        try:
            self.explainer = shap.TreeExplainer(m)
            self.explainer_kind = "TreeExplainer"
            self.status = "Supported"
            self.status_reason = "Tree-based model: Fast, exact TreeExplainer initialized."
            return
        except Exception:
            pass

        # 2. Attempt LinearExplainer on the effective estimator
        try:
            self.explainer = shap.LinearExplainer(m, sample_bg)
            self.explainer_kind = "LinearExplainer"
            self.status = "Supported"
            self.status_reason = "Linear model: LinearExplainer initialized with background sample."
            return
        except Exception:
            pass

        # 3. Model-agnostic KernelExplainer fallback
        try:
            raw_predict_fn = getattr(self.raw_model, "predict_proba", getattr(self.raw_model, "predict", None))
            if raw_predict_fn:
                def kernel_predict(data_array):
                    arr = np.asarray(data_array)
                    if hasattr(bg_raw, "columns") and arr.shape[1] == len(bg_raw.columns):
                        df_in = pd.DataFrame(arr, columns=bg_raw.columns)
                    else:
                        df_in = arr
                    return raw_predict_fn(df_in)

                bg_kernel = sample_bg_raw.values if hasattr(sample_bg_raw, "values") else np.asarray(sample_bg_raw)
                self.explainer = shap.KernelExplainer(kernel_predict, bg_kernel)
                self.explainer_kind = "KernelExplainer"
                self.status = "Limited"
                self.status_reason = "Model-agnostic KernelExplainer initialized (approximate sampling)."
                return
        except Exception:
            pass

        self.status = "Unsupported"
        self.status_reason = "Could not initialize SHAP explainer for this model structure."

    def _class_values(self, sv: Any) -> np.ndarray:
        if hasattr(sv, "values"):
            vals = np.asarray(sv.values)
        else:
            vals = np.asarray(sv)
        if vals.ndim == 3:  # (n_samples, n_features, n_classes) -> positive or last class
            vals = vals[:, :, -1]
        return vals

    def explain_instance(self, row: pd.DataFrame) -> Dict[str, Any]:
        """Explain a single instance prediction."""
        if self.status == "Unsupported" or self.explainer is None:
            return {"status": "Unsupported", "reason": self.status_reason}

        row_df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
        try:
            row_for_shap = row_df if self.explainer_kind == "KernelExplainer" else self._transform_features(row_df)

            sv = self.explainer(row_for_shap)
            vals = self._class_values(sv)[0]

            base = 0.0
            if hasattr(sv, "base_values"):
                base_arr = np.ravel(np.asarray(sv.base_values))
                base = float(base_arr[-1] if base_arr.size else 0.0)

            prediction = getattr(self.raw_model, "predict")(row_df)[0]
            if isinstance(prediction, (np.integer, int)):
                prediction = int(prediction)
            elif isinstance(prediction, (np.floating, float)):
                prediction = round(float(prediction), 4)

            prob = None
            if hasattr(self.raw_model, "predict_proba"):
                try:
                    prob_arr = self.raw_model.predict_proba(row_df)[0]
                    prob = float(prob_arr[-1])
                except Exception:
                    prob = None

            order = np.argsort(-np.abs(vals))
            cols = list(row_for_shap.columns) if hasattr(row_for_shap, "columns") else [f"f_{i}" for i in range(len(vals))]
            top = []
            for i in order[:10]:
                val_repr = str(row_for_shap.iloc[0, i]) if hasattr(row_for_shap, "iloc") else str(row_for_shap[0, i])
                top.append({
                    "feature": cols[i],
                    "value": val_repr,
                    "shap_value": round(float(vals[i]), 4),
                    "direction": "contributed to higher prediction" if vals[i] >= 0 else "contributed to lower prediction"
                })

            return {
                "status": "Success",
                "prediction": prediction,
                "probability": prob,
                "base_value": base,
                "top_features": top
            }
        except Exception as e:
            return {"status": "Error", "reason": f"SHAP instance explanation failed: {e}"}

    def compute_global_importance(
        self,
        X: pd.DataFrame,
        out_dir: str = "outputs",
        sample_limit: int = 300,
        max_display: int = 15
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Compute global feature importances and render summary plot."""
        if self.status == "Unsupported" or self.explainer is None:
            return None, None

        os.makedirs(out_dir, exist_ok=True)
        try:
            Xs_raw = X.sample(min(sample_limit, len(X)), random_state=42) if len(X) > sample_limit else X
            Xs_shap = Xs_raw if self.explainer_kind == "KernelExplainer" else self._transform_features(Xs_raw)

            sv = self.explainer(Xs_shap)
            vals = self._class_values(sv)
            mean_abs = np.abs(vals).mean(axis=0)

            cols = list(Xs_shap.columns) if hasattr(Xs_shap, "columns") else [f"f_{i}" for i in range(len(mean_abs))]
            df_imp = (pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs})
                      .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

            plt.figure(figsize=(9, 6))
            shap.summary_plot(vals, Xs_shap, feature_names=cols, show=False, max_display=max_display)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, "shap_summary.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            return df_imp, plot_path
        except Exception as e:
            return None, None
