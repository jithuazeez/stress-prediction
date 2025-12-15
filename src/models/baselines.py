"""
Classical ML baseline models for stress prediction.

Models:
- Logistic Regression with L2 regularization
- Random Forest with feature importance
- XGBoost with class balancing
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available: {e}")
    print("Training will proceed with Logistic Regression and Random Forest only.")


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, model_name, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
    
    def fit(self, X, y, feature_names=None):
        """Train the model."""
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Extract feature importance if available
        self._extract_feature_importance()
    
    def predict(self, X):
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def _extract_feature_importance(self):
        """Extract feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (RF, XGBoost)
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        elif hasattr(self.model, 'coef_'):
            # Linear models (Logistic Regression)
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
    
    def get_feature_importance(self, top_n=20):
        """Get top N most important features."""
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        return None
    
    def save(self, filepath):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'model_name': self.model_name
            }, f)
    
    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
            self.model_name = data['model_name']


class LogisticRegressionModel(BaselineModel):
    """Logistic Regression with L2 regularization."""
    
    def __init__(self, C=1.0, class_weight='balanced', random_state=42):
        super().__init__("Logistic Regression", random_state)
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )


class RandomForestModel(BaselineModel):
    """Random Forest classifier."""
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 class_weight='balanced', random_state=42):
        super().__init__("Random Forest", random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )


class XGBoostModel(BaselineModel):
    """XGBoost classifier with class balancing."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 scale_pos_weight=None, random_state=42):
        super().__init__("XGBoost", random_state)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )


def create_baseline_models(pos_weight=None):
    """
    Create all baseline models.
    
    Args:
        pos_weight: Positive class weight for XGBoost
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {
        'logistic_regression': LogisticRegressionModel(),
        'random_forest': RandomForestModel(n_estimators=100, max_depth=10),
    }
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostModel(
            n_estimators=100, 
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight
        )
    
    return models


def evaluate_model(y_true, y_pred, y_proba):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'auroc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # Operational metrics
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


if __name__ == '__main__':
    # Test model creation
    print("Testing baseline models...")
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Test each model
    for name, model in create_baseline_models().items():
        print(f"\nTesting {name}...")
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        metrics = evaluate_model(y, y_pred, y_proba)
        print(f"  AUROC: {metrics['auroc']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        
        # Feature importance
        importance = model.get_feature_importance(top_n=5)
        if importance is not None:
            print(f"  Top 5 features:")
            for _, row in importance.iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
    
    print("\n✓ All models tested successfully!")

