"""
Asymmetric Threshold Selection for Two-Stage Ensemble

Provides utilities for selecting thresholds with different optimization criteria:
- Recall-first: Find lowest threshold that meets minimum recall
- FAR-first: Find highest threshold that meets maximum false alarm rate
- LR decision + TCN confidence: LR makes decision, TCN provides confidence
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import confusion_matrix


def find_recall_first_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.75
) -> Tuple[float, Dict[str, float]]:
    """
    Find lowest threshold that achieves minimum recall.
    
    Strategy: Maximize recall while meeting constraint.
    Use case: First stage screening (LR) - catch most stress events.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        min_recall: Minimum required recall (default: 0.75)
    
    Returns:
        Tuple of (threshold, metrics_dict)
    """
    # Sort probabilities and find unique thresholds
    thresholds = np.unique(y_proba)
    thresholds = np.sort(thresholds)
    
    best_threshold = 0.5
    best_metrics = {}
    
    # Start from lowest threshold (highest recall)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Check if recall constraint is met
        if recall >= min_recall:
            # This threshold meets the constraint
            # We want the HIGHEST threshold that still meets it (for better precision)
            best_threshold = threshold
            best_metrics = {
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "false_alarm_rate": far,
                "threshold": threshold
            }
    
    # If no threshold meets constraint, use lowest threshold
    if not best_metrics:
        threshold = thresholds[0] if len(thresholds) > 0 else 0.0
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        best_threshold = threshold
        best_metrics = {
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "false_alarm_rate": far,
            "threshold": threshold,
            "warning": f"Could not meet min_recall={min_recall}, best={recall:.3f}"
        }
    
    return best_threshold, best_metrics


def find_far_first_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    max_far: float = 0.25
) -> Tuple[float, Dict[str, float]]:
    """
    Find highest threshold that achieves maximum false alarm rate.
    
    Strategy: Minimize false alarms while meeting constraint.
    Use case: Second stage confirmation (TCN) - high specificity filter.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        max_far: Maximum allowed false alarm rate (default: 0.25)
    
    Returns:
        Tuple of (threshold, metrics_dict)
    """
    # Sort probabilities and find unique thresholds
    thresholds = np.unique(y_proba)
    thresholds = np.sort(thresholds)[::-1]  # Start from highest (lowest FAR)
    
    best_threshold = 0.5
    best_metrics = {}
    
    # Start from highest threshold (lowest FAR)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Check if FAR constraint is met
        if far <= max_far:
            # This threshold meets the constraint
            # We want the LOWEST threshold that still meets it (for better recall)
            best_threshold = threshold
            best_metrics = {
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "false_alarm_rate": far,
                "threshold": threshold
            }
    
    # If no threshold meets constraint, use highest threshold
    if not best_metrics:
        threshold = thresholds[0] if len(thresholds) > 0 else 1.0
        y_pred = (y_proba >= threshold).astype(int)
        
        # Handle edge case where all predictions are 0
        if y_pred.sum() == 0:
            best_threshold = threshold
            best_metrics = {
                "recall": 0.0,
                "specificity": 1.0,
                "precision": 0.0,
                "false_alarm_rate": 0.0,
                "threshold": threshold,
                "warning": f"Could not meet max_far={max_far}, using threshold={threshold:.3f}"
            }
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            best_threshold = threshold
            best_metrics = {
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "false_alarm_rate": far,
                "threshold": threshold,
                "warning": f"Could not meet max_far={max_far}, best={far:.3f}"
            }
    
    return best_threshold, best_metrics


def apply_two_stage_decision(
    lr_proba: np.ndarray,
    tcn_proba: np.ndarray,
    lr_threshold: float,
    tcn_threshold: float
) -> np.ndarray:
    """
    Apply two-stage decision logic (HARD AND CASCADE).
    
    Rules:
    1. If LR < threshold → NO STRESS (skip TCN)
    2. If LR ≥ threshold → Check TCN:
       - If TCN ≥ threshold → STRESS
       - If TCN < threshold → NO STRESS
    
    Args:
        lr_proba: LR probabilities
        tcn_proba: TCN probabilities
        lr_threshold: LR decision threshold
        tcn_threshold: TCN decision threshold
    
    Returns:
        Binary predictions (0 or 1)
    """
    predictions = np.zeros(len(lr_proba), dtype=int)
    
    # Stage 1: LR screening
    lr_detects = (lr_proba >= lr_threshold)
    
    # Stage 2: TCN confirmation (only where LR detected)
    tcn_confirms = (tcn_proba >= tcn_threshold)
    
    # Final decision: LR detects AND TCN confirms
    predictions = (lr_detects & tcn_confirms).astype(int)
    
    return predictions


def apply_lr_decision_with_tcn_confidence(
    lr_proba: np.ndarray,
    tcn_proba: np.ndarray,
    lr_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LR makes all decisions, TCN provides confidence scores.
    
    Strategy: Maximize recall by using LR as sole decision maker.
    TCN's role is ONLY to provide confidence/reliability scores.
    
    Rules:
    1. Decision: LR probability ≥ threshold → STRESS, else NO STRESS
    2. Confidence: 
       - For STRESS predictions: confidence = TCN_proba (TCN agreement)
       - For NO STRESS predictions: confidence = 1 - TCN_proba (TCN agreement)
    
    This ensures:
    - LR's high recall is preserved (catches all stress)
    - TCN never vetoes stress detected by LR
    - TCN provides interpretable confidence for reporting
    
    Args:
        lr_proba: LR probabilities (decision maker)
        tcn_proba: TCN probabilities (confidence scorer)
        lr_threshold: LR decision threshold
    
    Returns:
        Tuple of (predictions, confidence_scores)
        - predictions: Binary (0 or 1)
        - confidence_scores: Float in [0, 1], higher = more confident
    """
    # LR makes the decision
    predictions = (lr_proba >= lr_threshold).astype(int)
    
    # TCN provides confidence
    # If we predict STRESS (1): confidence = TCN agreement (high TCN_proba = high confidence)
    # If we predict NO STRESS (0): confidence = TCN agreement (low TCN_proba = high confidence)
    confidence = np.where(predictions == 1, tcn_proba, 1 - tcn_proba)
    
    return predictions, confidence

