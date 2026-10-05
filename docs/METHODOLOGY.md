# AuditLens Methodology

## 1. Data preparation
- Target column: last column by default (overridable in the UI).
- Columns with >30% missing values are dropped; remaining missing values
  are imputed (median for numeric after encoding).
- Categorical features are label-encoded; the encoding map is retained
  for reporting.

## 2. Model selection
Two candidate classifiers are trained:
- LogisticRegression (max_iter=1000)
- XGBClassifier (200 trees, depth 6, lr 0.1)

Each is scored with 5-fold cross-validation on accuracy and F1. The model
with the higher CV F1 is selected and evaluated on a stratified 25%
hold-out split (accuracy, precision, recall, F1, ROC AUC, confusion matrix).

## 3. Explainability (SHAP)
- **Global**: mean(|SHAP value|) per feature over a sample of up to 300
  rows + beeswarm summary plot.
- **Local**: per-instance SHAP values; the top-5 drivers are reported with
  direction ("increases/decreases prediction") and the base value.
- TreeExplainer is used for tree ensembles; KernelExplainer is the
  fallback for linear models.

## 4. Fairness audit (Fairlearn)
Sensitive columns are auto-detected by name (sex, gender, race, age,
marital, ethnicity, religion, nationality) or chosen manually.

Per sensitive column, on the hold-out predictions:
- **Demographic parity difference** = max group selection rate − min group
  selection rate.
- **Demographic parity ratio** = min / max selection rate.
- **Equalized odds difference** = max over groups of (TPR gap, FPR gap).
- **Selection rate by group** + bar chart.

### Verdict thresholds (documented, tunable in `compute_verdict`)
| Verdict | Worst |parity difference| |
|---|---|
| LOW | < 0.05 |
| MODERATE | 0.05 – 0.15 |
| HIGH | > 0.15 (recommend: do not deploy) |

## 5. Audit report generation
All statistics (profile, metrics, top SHAP drivers, instance explanations,
fairness metrics, verdict) are composed into a structured prompt for a
Gemini LLM, which writes a one-page markdown report (Executive Summary,
Model Performance, Key Decision Drivers, Fairness Findings,
Recommendations). Without an API key, an equivalent templated report is
generated so the system never fails.
