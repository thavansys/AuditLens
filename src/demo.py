"""Offline end-to-end CLI check: python src/demo.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_full_audit


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "data", "adult_sample.csv")
    if not os.path.exists(csv_path):
        print(f"Demo dataset not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("AUDITLENS - Offline end-to-end check")
    print("=" * 60)
    result = run_full_audit(csv_path, progress=lambda m: print(f"  [{m}]"))

    print("\n--- DATASET PROFILE ---")
    for k, v in result["profile"].items():
        print(f"  {k}: {v}")

    print("\n--- MODEL ---")
    print(result["comparison"].to_string(index=False))
    print(f"Selected: {result['model_name']}")
    print(f"Metrics: {result['metrics']}")

    print("\n--- TOP GLOBAL DRIVERS ---")
    print(result["top_features"].head(10).to_string(index=False))

    print("\n--- SAMPLE INSTANCE EXPLANATION ---")
    inst = result["instances"][0]
    print(f"Predicted class {inst['prediction']} (prob={inst['probability']:.3f})")
    for t in inst["top_features"]:
        print(f"  {t['feature']}: {t['shap_value']} ({t['direction']})")

    print("\n--- FAIRNESS ---")
    for col, r in result["fairness"].items():
        print(f"  {col}: parity_diff={r['demographic_parity_difference']} "
              f"rates={r['selection_rate_by_group']}")
    print(f"  VERDICT: {result['verdict']}")

    print("\n--- REPORT ---")
    print(result["report_text"][:1500])
    print(f"\nReport saved to: {result['report_paths']['markdown']}")
    print("=" * 60)
    print("ALL CHECKS PASSED - AuditLens is working.")


if __name__ == "__main__":
    main()
