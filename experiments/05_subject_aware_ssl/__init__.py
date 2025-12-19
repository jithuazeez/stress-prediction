"""
Experiment 05: Subject-Aware Contrastive Self-Supervised Learning.

Based on the Apple paper: "Self-supervised learning of electrodermal activity 
representations for stress classification."

Implements three training modes:
1. Base SSL: Standard contrastive learning (InfoNCE)
2. Subject-Invariant: Contrastive + Adversarial loss (for generalization)
3. Subject-Specific: Contrastive with same-subject negatives (for fine-tuning)
"""


