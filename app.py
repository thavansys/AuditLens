"""AuditLens - AI Model Audit & Explainability Platform (Streamlit UI)."""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import run_full_audit, audit_uploaded_model
from src.explainer import explain_instance
from src.model_loader import load_model_file, validate_model_object, ModelLoadingError
from src.model_adapter import SklearnModelAdapter
from src.compatibility import CompatibilityChecker

st.set_page_config(page_title="AuditLens", page_icon="🔍", layout="wide")

VERDICT_COLORS = {"LOW": "#2E7D32", "MODERATE": "#F9A825", "HIGH": "#C62828", "N/A": "#757575"}

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.title("🔍 AuditLens")
    st.caption("*We don't just build AI models — we make them answer for their decisions.*")
    with st.expander("How it works"):
        st.markdown(
            "1. **Upload** a tabular CSV (or load the demo dataset)\n"
            "2. **Train** in AuditLens or **Upload** an existing pre-trained model\n"
            "3. **SHAP** explains every individual decision\n"
            "4. **Fairlearn** audits bias across sensitive groups\n"
            "5. An **LLM** writes a plain-English audit report")
    key = st.text_input("Gemini API key (optional)", type="password",
                        help="Leave empty to use template report mode.")
    if key:
        os.environ["GEMINI_API_KEY"] = key
    st.divider()
    st.caption("⚠️ Decision-support tool. Not a substitute for human "
               "judgment or legal compliance review.")

st.title("🔍 AuditLens")
st.subheader("AI Model Audit & Explainability Platform")

tab_upload, tab_model, tab_explain, tab_fair = st.tabs(
    ["📂 Upload & Overview", "🤖 Model", "🔎 Explain", "⚖️ Fairness & Report"])

if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "audit" not in st.session_state:
    st.session_state.audit = None
if "sensitive_sel" not in st.session_state:
    st.session_state.sensitive_sel = []
if "model_source" not in st.session_state:
    st.session_state.model_source = "Train model in AuditLens"
if "uploaded_model" not in st.session_state:
    st.session_state.uploaded_model = None
if "uploaded_model_name" not in st.session_state:
    st.session_state.uploaded_model_name = None

# ------------------------- Tab 1: Upload -------------------------
with tab_upload:
    st.header("Upload & Overview")
    demo_path = os.path.join(BASE, "data", "adult_sample.csv")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📊 Load demo dataset (Adult Census)", use_container_width=True):
            st.session_state.csv_path = demo_path
            st.session_state.target_col = None
            st.session_state.audit = None
            st.success("Demo dataset loaded.")
    with col_b:
        up = st.file_uploader("...or upload a CSV", type=["csv"])
        if up is not None:
            save_to = os.path.join(BASE, "data", "_upload.csv")
            with open(save_to, "wb") as f:
                f.write(up.getbuffer())
            st.session_state.csv_path = save_to
            st.session_state.target_col = None
            st.session_state.audit = None
            st.success(f"Loaded: {up.name}")

    if st.session_state.csv_path:
        try:
            df = pd.read_csv(st.session_state.csv_path)
            cols = list(df.columns)
            if len(cols) < 2:
                st.error("The CSV file must have at least 2 columns (one feature and one target).")
            else:
                default_target = cols[-1]
                st.session_state.target_col = st.selectbox(
                    "Target column", cols, index=cols.index(default_target),
                    key="target_pick")
                st.session_state.sensitive_sel = st.multiselect(
                    "Sensitive columns for fairness audit (auto-detected if empty)",
                    [c for c in cols if c != st.session_state.target_col],
                    default=st.session_state.sensitive_sel)

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{df.shape[0]:,}")
                c2.metric("Columns", df.shape[1])
                c3.metric("Missing %", f"{df.isna().mean().mean() * 100:.2f}")
                target_series = df[st.session_state.target_col].dropna().astype(str)
                if len(target_series) > 0:
                    bal = target_series.value_counts(normalize=True)
                    c4.metric("Top class", f"{bal.index[0]} ({bal.iloc[0] * 100:.0f}%)")
                else:
                    c4.metric("Top class", "N/A")
                st.dataframe(df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Load the demo dataset or upload a CSV to begin.")

# ------------------------- Tab 2: Model -------------------------
with tab_model:
    st.header("Model Evaluation & Audit")
    if not st.session_state.csv_path:
        st.info("Upload data first (tab 1).")
    elif not st.session_state.target_col:
        st.warning("Please choose a target column in Tab 1 before training or auditing.")
    else:
        st.subheader("Model Source")
        model_source = st.radio(
            "Select Model Source",
            ["Train model in AuditLens", "Upload existing model"],
            index=0 if st.session_state.model_source == "Train model in AuditLens" else 1,
            horizontal=True,
            key="model_source_radio"
        )
        st.session_state.model_source = model_source

        if model_source == "Train model in AuditLens":
            st.caption("Automatically trains and evaluates Logistic Regression and XGBoost, selecting the best performer.")
            if st.button("🚀 Run Full Audit", type="primary", use_container_width=True, key="train_audit_btn"):
                try:
                    with st.status("Running full audit...", expanded=True) as status:
                        audit = run_full_audit(
                            st.session_state.csv_path,
                            target_col=st.session_state.target_col,
                            sensitive_cols=st.session_state.sensitive_sel or None,
                            out_dir=os.path.join(BASE, "outputs"),
                            progress=lambda m: status.write(m))
                    st.session_state.audit = audit
                    st.success("Audit complete!")
                except Exception as e:
                    st.error(f"Audit failed: {e}")

        else:  # "Upload existing model"
            st.caption("Upload a pre-trained scikit-learn model or Pipeline (.pkl, .joblib) to audit directly without retraining.")
            model_file = st.file_uploader(
                "Upload existing model (.pkl or .joblib)",
                type=["pkl", "joblib", "pickle"],
                key="model_file_uploader",
                help="Upload a fitted scikit-learn estimator or Pipeline saved as a .pkl or .joblib file."
            )

            if model_file is not None:
                try:
                    loaded_obj = load_model_file(model_file)
                    val = validate_model_object(loaded_obj)
                    st.session_state.uploaded_model = loaded_obj
                    st.session_state.uploaded_model_name = getattr(model_file, "name", "uploaded_model")

                    adapter = SklearnModelAdapter(loaded_obj, model_name=st.session_state.uploaded_model_name)
                    meta = adapter.get_metadata()

                    st.success(f"Model loaded: **{meta['model_class']}** ({'Pipeline' if meta['is_pipeline'] else 'Estimator'})")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Model Class", meta["model_class"])
                    c2.metric("predict() Method", "Valid" if meta["has_predict"] else "Missing")
                    c3.metric("predict_proba()", "Available (ROC-AUC enabled)" if meta["has_predict_proba"] else "Not Available")

                    if meta["is_pipeline"] and meta["pipeline_steps"]:
                        steps_str = " ➔ ".join(f"{name} ({cls})" for name, cls in meta["pipeline_steps"])
                        st.caption(f"**Pipeline Steps:** {steps_str}")

                    # Compatibility pre-check against active dataset
                    df_current = pd.read_csv(st.session_state.csv_path)
                    compat = CompatibilityChecker.check_compatibility(
                        adapter, df_current, target_col=st.session_state.target_col
                    )

                    if not compat["can_audit"]:
                        st.error("⚠️ **Feature Mismatch Detected:** The uploaded model cannot accept this dataset:")
                        for issue in compat["issues"]:
                            st.write(f"- {issue}")
                        for rec in compat["recommendations"]:
                            st.info(f"💡 {rec}")
                    elif compat["issues"]:
                        for issue in compat["issues"]:
                            st.warning(f"ℹ️ {issue}")
                    else:
                        st.info("✅ Model interface and features are compatible with the loaded dataset.")

                except ModelLoadingError as mle:
                    st.error(f"Model validation error: {mle}")
                    st.session_state.uploaded_model = None
                except Exception as e:
                    st.error(f"Failed to load or validate model file: {e}")
                    st.session_state.uploaded_model = None

            if st.session_state.uploaded_model is not None:
                if st.button("🚀 Audit Uploaded Model", type="primary", use_container_width=True, key="audit_uploaded_btn"):
                    try:
                        df_current = pd.read_csv(st.session_state.csv_path)
                        adapter = SklearnModelAdapter(
                            st.session_state.uploaded_model,
                            model_name=st.session_state.uploaded_model_name or "Uploaded Model"
                        )
                        compat = CompatibilityChecker.check_compatibility(
                            adapter, df_current, target_col=st.session_state.target_col
                        )
                        if not compat["can_audit"]:
                            st.error(
                                "Cannot audit model due to feature mismatch:\n" +
                                "\n".join(f"- {iss}" for iss in compat["issues"])
                            )
                        else:
                            with st.status("Auditing uploaded model (no retraining)...", expanded=True) as status:
                                audit = audit_uploaded_model(
                                    adapter,
                                    df_current,
                                    target_col=st.session_state.target_col,
                                    sensitive_cols=st.session_state.sensitive_sel or None,
                                    out_dir=os.path.join(BASE, "outputs"),
                                    progress=lambda m: status.write(m)
                                )
                            st.session_state.audit = audit
                            st.success("Uploaded model audit complete!")
                    except Exception as e:
                        st.error(f"Audit failed: {e}")

        audit = st.session_state.audit
        if audit:
            st.divider()
            st.subheader(f"Audited Model: {audit.get('model_name', 'Model')}")
            if audit.get("comparison") is not None:
                st.caption("Training Comparison (Cross-Validation)")
                st.dataframe(audit["comparison"], use_container_width=True, hide_index=True)
            elif audit.get("model_metadata"):
                meta = audit["model_metadata"]
                pipe_desc = "Pipeline" if meta.get("is_pipeline") else "Estimator"
                st.caption(f"Uploaded Model: **{meta.get('model_class')}** ({pipe_desc})")

            m = audit.get("metrics", {})
            if m:
                if m.get("accuracy") is not None:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Accuracy", m.get("accuracy", "N/A"))
                    c2.metric("Precision", m.get("precision", "N/A"))
                    c3.metric("Recall", m.get("recall", "N/A"))
                    c4.metric("F1", m.get("f1", "N/A"))
                    c5.metric("ROC AUC", m.get("roc_auc") if m.get("roc_auc") is not None else "N/A")
                elif m.get("task") == "regression":
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MAE", m.get("mae", "N/A"))
                    c2.metric("MSE", m.get("mse", "N/A"))
                    c3.metric("RMSE", m.get("rmse", "N/A"))
                    c4.metric("R²", m.get("r2", "N/A"))

                if m.get("confusion_matrix"):
                    try:
                        import plotly.express as px
                        cm = np.array(m["confusion_matrix"])
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                        labels=dict(x="Predicted", y="Actual"),
                                        title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write("Confusion matrix:", m["confusion_matrix"])

# ------------------------- Tab 3: Explain -------------------------
with tab_explain:
    st.header("Explain Individual Decisions")
    audit = st.session_state.audit
    if not audit:
        st.info("Run the audit first (Model tab).")
    else:
        st.caption(f"Explainer: {audit.get('explainer_kind', 'SHAP Explainer')}")
        X_test = audit.get("X_test_trans", audit.get("X_test", audit.get("X_eval")))
        if X_test is not None and len(X_test) > 0:
            row_i = st.slider("Pick a test instance", 0, len(X_test) - 1, 0)
            if st.button("Explain this instance"):
                try:
                    if audit.get("explain_engine") and audit["explain_engine"].status != "Unsupported":
                        X_eval = audit.get("X_eval", audit.get("X_test", X_test))
                        inst = audit["explain_engine"].explain_instance(X_eval.iloc[[row_i]])
                    else:
                        inst = explain_instance(audit["explainer"],
                                                X_test.loc[[row_i]] if hasattr(X_test, "loc") else X_test.iloc[[row_i]],
                                                audit["model"])

                    if inst.get("status") in ("Unsupported", "Error"):
                        st.warning(f"Could not explain instance: {inst.get('reason')}")
                    else:
                        classes = audit.get("classes") or []
                        pred_val = inst.get("prediction")
                        pred_label = classes[pred_val] if (isinstance(pred_val, int) and pred_val < len(classes)) else pred_val
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted class / value", pred_label)
                        prob_text = f"{inst['probability']:.3f}" if inst.get("probability") is not None else "N/A"
                        c2.metric("Probability (positive)", prob_text)
                        c3.metric("Base value", f"{inst['base_value']:.3f}" if inst.get("base_value") is not None else "N/A")
                        feat_df = pd.DataFrame(inst.get("top_features", []))
                        if not feat_df.empty and "feature" in feat_df.columns and "shap_value" in feat_df.columns:
                            st.bar_chart(feat_df.set_index("feature")["shap_value"])
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Explanation failed: {e}")

        st.divider()
        st.subheader("Global feature importance (SHAP)")
        if audit.get("shap_img") and os.path.exists(audit["shap_img"]):
            st.image(audit["shap_img"], use_container_width=True)
        if audit.get("top_features") is not None and hasattr(audit["top_features"], "head"):
            st.dataframe(audit["top_features"].head(15), use_container_width=True, hide_index=True)

# ---------------------- Tab 4: Fairness & Report ----------------------
with tab_fair:
    st.header("Fairness Audit & Report")
    audit = st.session_state.audit
    if not audit:
        st.info("Run the audit first (Model tab).")
    else:
        if not audit.get("fairness"):
            st.warning("No sensitive columns detected. Select them in tab 1 and re-run.")
        else:
            rows = []
            for col, r in audit["fairness"].items():
                if r.get("task") == "regression":
                    rows.append({"column": col,
                                 "mean prediction by group": r.get("mean_prediction_by_group"),
                                 "MAE by group": r.get("mae_by_group")})
                else:
                    rows.append({"column": col,
                                 "parity diff": r.get("demographic_parity_difference"),
                                 "parity ratio": r.get("demographic_parity_ratio"),
                                 "equalized odds diff": r.get("equalized_odds_difference"),
                                 "selection rates": r.get("selection_rate_by_group")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            for col, r in audit["fairness"].items():
                if r.get("chart_path") and os.path.exists(r["chart_path"]):
                    st.image(r["chart_path"], caption=f"Selection rate by {col}",
                             use_container_width=True)

        color = VERDICT_COLORS.get(audit.get("verdict", "N/A"), "#757575")
        worst_disp = audit.get("worst_disparity")
        disp_text = f" <small>(worst |parity difference| = {worst_disp})</small>" if worst_disp is not None else ""
        st.markdown(
            f"**Overall disparity verdict:** "
            f"<span style='background:{color};color:white;padding:4px 14px;"
            f"border-radius:12px;font-weight:bold'>{audit.get('verdict', 'N/A')}</span>"
            f"{disp_text}",
            unsafe_allow_html=True)
        st.caption("Note: Potential statistical disparity detected. Statistical disparity alone does not establish discrimination or legal non-compliance.")

        st.divider()
        st.subheader("📄 Audit Report")
        if audit.get("report_text"):
            st.markdown(audit["report_text"])
        rp = audit.get("report_paths")
        if rp and rp.get("markdown") and os.path.exists(rp["markdown"]):
            c1, c2 = st.columns(2)
            with open(rp["markdown"], "rb") as f:
                c1.download_button("⬇️ Download report (.md)", f, "audit_report.md")
            if rp.get("text") and os.path.exists(rp["text"]):
                with open(rp["text"], "rb") as f:
                    c2.download_button("⬇️ Download report (.txt)", f, "audit_report.txt")

