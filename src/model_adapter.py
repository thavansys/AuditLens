"""Generic model adapter and metadata inspection interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class ModelAdapter(ABC):
    """Abstract adapter providing unified inspection and prediction."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run predictions on DataFrame X."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Run prediction probabilities on DataFrame X if supported."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return dict of model metadata and capabilities."""
        pass


class SklearnModelAdapter(ModelAdapter):
    """Concrete adapter for scikit-learn compatible estimators and pipelines."""

    def __init__(self, raw_model: Any, model_name: Optional[str] = None):
        self.raw_model = raw_model
        self.model_name = model_name or raw_model.__class__.__name__
        self._inspect()

    def _inspect(self):
        m = self.raw_model
        self.is_pipeline = isinstance(m, Pipeline)
        self.has_predict = hasattr(m, "predict") and callable(getattr(m, "predict"))
        self.has_predict_proba = hasattr(m, "predict_proba") and callable(getattr(m, "predict_proba"))
        self.has_decision_function = hasattr(m, "decision_function") and callable(getattr(m, "decision_function"))

        # Infer problem type
        self.problem_type = "unknown"
        try:
            if is_classifier(m):
                self.problem_type = "classification"
            elif is_regressor(m):
                self.problem_type = "regression"
        except Exception:
            pass

        if self.problem_type == "unknown":
            if hasattr(m, "classes_") or self.has_predict_proba:
                self.problem_type = "classification"
            elif hasattr(m, "_estimator_type"):
                self.problem_type = str(getattr(m, "_estimator_type"))

        # Classes
        self.classes = None
        if hasattr(m, "classes_"):
            try:
                self.classes = [str(c) for c in m.classes_]
            except Exception:
                self.classes = None

        # Features expected
        self.n_features_in = getattr(m, "n_features_in_", None)
        self.feature_names_in = None
        if hasattr(m, "feature_names_in_"):
            try:
                self.feature_names_in = list(m.feature_names_in_)
            except Exception:
                self.feature_names_in = None

        # If pipeline, inspect steps and extract features if root lacks them
        self.pipeline_steps = []
        self.preprocessor = None
        if self.is_pipeline:
            try:
                for step_name, step_obj in m.steps:
                    self.pipeline_steps.append((step_name, step_obj.__class__.__name__))
                    if isinstance(step_obj, ColumnTransformer):
                        self.preprocessor = step_obj
                    elif step_name.lower() in ("prep", "preprocessor", "transformer"):
                        self.preprocessor = step_obj
                
                # Check first step for feature metadata if root didn't expose it
                if len(m.steps) > 0:
                    first_obj = m.steps[0][1]
                    if self.feature_names_in is None and hasattr(first_obj, "feature_names_in_"):
                        try:
                            self.feature_names_in = list(first_obj.feature_names_in_)
                        except Exception:
                            pass
                    if self.n_features_in is None and hasattr(first_obj, "n_features_in_"):
                        self.n_features_in = getattr(first_obj, "n_features_in_", None)
            except Exception:
                pass

        # Check if fitted
        self.is_fitted = True
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(m)
        except Exception:
            # If check_is_fitted fails or not implemented for custom object
            if not (hasattr(m, "classes_") or hasattr(m, "n_features_in_") or hasattr(m, "coef_") or hasattr(m, "feature_importances_") or self.is_pipeline):
                self.is_fitted = False

        # Status
        if not self.has_predict:
            self.status = "Unsupported"
            self.status_reason = "Model does not implement a callable predict() method."
        elif not self.is_fitted:
            self.status = "Warning"
            self.status_reason = "Model does not appear to be fitted."
        else:
            self.status = "Compatible"
            self.status_reason = "Model implements scikit-learn standard prediction interface."

    def _prepare_input(self, X: pd.DataFrame) -> Any:
        """Align input DataFrame with expected model features if known."""
        if not isinstance(X, pd.DataFrame):
            return X

        if self.feature_names_in is not None:
            missing = [f for f in self.feature_names_in if f not in X.columns]
            if missing:
                raise ValueError(
                    f"Feature mismatch: Model expects {len(self.feature_names_in)} features, "
                    f"but {len(missing)} are missing from the input: {missing[:5]}. "
                    f"Expected features: {self.feature_names_in}"
                )
            return X[self.feature_names_in]

        if self.n_features_in is not None and X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Feature count mismatch: Model expects {self.n_features_in} features, "
                f"but received {X.shape[1]} features: {list(X.columns)}."
            )
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.has_predict:
            raise RuntimeError("Model does not have predict() capability.")
        X_in = self._prepare_input(X)
        try:
            return np.asarray(self.raw_model.predict(X_in))
        except Exception as e:
            err_str = str(e)
            if "feature" in err_str.lower() or "columns" in err_str.lower():
                raise ValueError(
                    f"Model prediction failed due to feature mismatch: {err_str}\n"
                    f"Expected features: {self.feature_names_in or f'{self.n_features_in} features'}. "
                    f"Received columns: {list(X.columns) if hasattr(X, 'columns') else 'non-DataFrame input'}."
                ) from e
            raise

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.has_predict_proba:
            return None
        try:
            X_in = self._prepare_input(X)
            return np.asarray(self.raw_model.predict_proba(X_in))
        except Exception:
            return None

    def decision_function(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.has_decision_function:
            return None
        try:
            X_in = self._prepare_input(X)
            return np.asarray(self.raw_model.decision_function(X_in))
        except Exception:
            return None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_class": self.raw_model.__class__.__name__,
            "module": self.raw_model.__class__.__module__,
            "problem_type": self.problem_type,
            "is_pipeline": self.is_pipeline,
            "pipeline_steps": self.pipeline_steps,
            "has_predict": self.has_predict,
            "has_predict_proba": self.has_predict_proba,
            "has_decision_function": self.has_decision_function,
            "is_fitted": self.is_fitted,
            "n_features_in": self.n_features_in,
            "feature_names_in": self.feature_names_in,
            "classes": self.classes,
            "status": self.status,
            "status_reason": self.status_reason,
        }
