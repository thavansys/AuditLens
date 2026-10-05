"""Safe model loading utilities for AuditLens."""
from __future__ import annotations

import io
import os
import pickle
import joblib


MAX_MODEL_SIZE_MB = 100


class ModelLoadingError(Exception):
    """Raised when model loading fails."""
    pass


def load_model_file(uploaded_file_or_path, trusted: bool = True):
    """Load a model from a file path or uploaded file-like buffer.

    Parameters:
        uploaded_file_or_path: Path string or file-like object with .name attribute.
        trusted: Safety acknowledgement flag.

    Returns:
        Loaded Python model object.
    """
    if isinstance(uploaded_file_or_path, str):
        path = uploaded_file_or_path
        if not os.path.exists(path):
            raise ModelLoadingError(f"Model file not found at: {path}")

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > MAX_MODEL_SIZE_MB:
            raise ModelLoadingError(
                f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_MODEL_SIZE_MB} MB."
            )

        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pkl", ".pickle", ".joblib"):
            raise ModelLoadingError(f"Unsupported model file extension '{ext}'. Expected .pkl, .pickle, or .joblib.")

        try:
            if ext == ".joblib":
                try:
                    return joblib.load(path)
                except Exception:
                    with open(path, "rb") as f:
                        return pickle.load(f)
            else:
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return joblib.load(path)
        except Exception as e:
            raise ModelLoadingError(f"Failed to deserialize model file: {e}")

    else:
        # File-like object (e.g. Streamlit UploadedFile)
        filename = getattr(uploaded_file_or_path, "name", "model.pkl")
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pkl", ".pickle", ".joblib"):
            raise ModelLoadingError(f"Unsupported model file extension '{ext}'. Expected .pkl, .pickle, or .joblib.")

        buf = uploaded_file_or_path.getvalue() if hasattr(uploaded_file_or_path, "getvalue") else uploaded_file_or_path.read()
        file_size_mb = len(buf) / (1024 * 1024)
        if file_size_mb > MAX_MODEL_SIZE_MB:
            raise ModelLoadingError(
                f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_MODEL_SIZE_MB} MB."
            )

        bio = io.BytesIO(buf)
        try:
            if ext == ".joblib":
                try:
                    return joblib.load(bio)
                except Exception:
                    bio.seek(0)
                    return pickle.load(bio)
            else:
                try:
                    return pickle.load(bio)
                except Exception:
                    bio.seek(0)
                    return joblib.load(bio)
        except Exception as e:
            raise ModelLoadingError(f"Failed to deserialize model buffer: {e}")


def validate_model_object(model: any) -> dict:
    """Validate that the loaded object implements standard ML prediction interface.

    Returns:
        dict with validation results and capabilities.
    Raises:
        ModelLoadingError if model does not have a callable predict() method.
    """
    has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
    if not has_predict:
        raise ModelLoadingError(
            f"Invalid model object of type '{model.__class__.__name__}': "
            "missing callable predict() method."
        )

    has_predict_proba = hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))
    class_name = model.__class__.__name__
    return {
        "valid": True,
        "class_name": class_name,
        "has_predict": True,
        "has_predict_proba": has_predict_proba,
    }
