"""
Two-Stage Ensemble: LR (recall-first) → TCN (FAR-first)

This experiment implements a cascaded ensemble combining:
1. Logistic Regression with high recall threshold (catches most stress)
2. TCN with low false alarm rate threshold (confirms with high specificity)

Decision Logic:
- If LR says "no stress" → Final prediction: NO STRESS
- If LR says "stress" → Ask TCN to confirm:
  - If TCN says "stress" → Final prediction: STRESS
  - If TCN says "no stress" → Final prediction: NO STRESS

Key Features:
- LOSO cross-validation
- Asymmetric threshold optimization:
  - LR: Recall-first (min 75% recall)
  - TCN: FAR-first (max 25% false alarm rate)
- Thresholds selected on TRAINING data only
- Subject-wise normalization
"""

__version__ = "1.0.0"

