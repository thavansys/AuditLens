# 🔍 AuditLens — AI Model Audit & Explainability Platform

> *We don't just build AI models — we make them answer for their decisions.*

AuditLens is an end-to-end **model auditing tool**: upload any tabular CSV,
and it auto-trains the best of two classifiers, explains every decision with
**SHAP**, audits **fairness** across sensitive groups with **Fairlearn**,
and writes a plain-English **Audit Report** (LLM-powered, with a template
fallback) — all in one Streamlit dashboard.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Offline check (no UI needed):

```bash
python src/demo.py
```

Optional — AI-written reports (otherwise a template report is used):

```bash
export GEMINI_API_KEY="your-key"   # free key at aistudio.google.com
```

## Architecture

```
CSV upload ──> data_loader.py ──> model_runner.py ──> explainer.py
                                        │                    │
                                        v                    v
                              fairness.py <────────── (SHAP values)
                                        │
                                        v
                              reporter.py ──> Streamlit dashboard (app.py)
```

| Module | Role |
|---|---|
| `src/data_loader.py` | CSV load, auto-encoding, imputation, profiling |
| `src/model_runner.py` | LogReg vs XGBoost, 5-fold CV, auto-selection |
| `src/explainer.py` | SHAP local + global explanations |
| `src/fairness.py` | Fairlearn disparity metrics + verdict |
| `src/reporter.py` | Gemini LLM / template audit report |
| `src/pipeline.py` | Orchestrates the full audit |
| `app.py` | 4-tab Streamlit dashboard |

## Fairness methodology

| Verdict | Worst \|demographic parity difference\| |
|---|---|
| 🟢 LOW | < 0.05 |
| 🟡 MODERATE | 0.05 – 0.15 |
| 🔴 HIGH | > 0.15 |

Metrics: demographic parity difference/ratio, equalized odds difference,
selection rate by group (see `docs/METHODOLOGY.md`).

## Demo dataset

`data/adult_sample.csv` — 2,000 synthetic Adult-Census-style rows so the
whole app runs offline. Contains intentional sex-based income disparity
for demonstrating the fairness audit.

## Disclaimer

Decision-support tool for education and hackathon demonstration. Not a
substitute for human judgment or legal/regulatory compliance review.

## License

MIT — see LICENSE.
