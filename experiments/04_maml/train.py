"""
Training script for MAML (Model-Agnostic Meta-Learning) with LOSO evaluation.

Implements personalized stress prediction via meta-learning.

References:
- https://github.com/cbfinn/maml
- https://arxiv.org/pdf/1703.03400
- https://github.com/learnables/learn2learn
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import time
import random

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.raw_loader import load_raw_signals, get_all_subjects, get_experiment_time_range
from shared.alignment import align_to_1hz
from shared.windowing import create_labeled_windows, parse_stress_events
from shared.evaluation import (
    evaluate_predictions, aggregate_fold_metrics,
    save_results, save_predictions, plot_results
)
from shared.config import DEFAULT_CONFIG, Config
from shared.logging_utils import (
    setup_logger, log_experiment_start, log_experiment_end,
    log_data_summary, log_model_results
)

from model import StressClassifier
from meta_dataset import VitaStressMetaDataset

# Try to import learn2learn
try:
    import learn2learn as l2l
    LEARN2LEARN_AVAILABLE = True
except ImportError:
    LEARN2LEARN_AVAILABLE = False


class ManualMAML:
    """
    Manual MAML implementation (fallback when learn2learn not available).
    
    Implements the core MAML algorithm:
    1. Sample tasks (subjects)
    2. For each task, compute adapted parameters via gradient descent
    3. Evaluate adapted model on query set
    4. Update meta-parameters based on query loss
    """
    
    def __init__(self, 
                 model: nn.Module,
                 lr_inner: float = 0.01,
                 lr_outer: float = 0.001,
                 n_inner_steps: int = 5):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.n_inner_steps = n_inner_steps
        
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
    
    def clone_model(self) -> nn.Module:
        """Create a clone of the model with same weights."""
        clone = type(self.model)(
            input_dim=self.model.input_dim,
            hidden_dim=self.model.hidden_dim,
            n_classes=self.model.n_classes
        )
        clone.load_state_dict(self.model.state_dict())
        return clone.to(next(self.model.parameters()).device)
    
    def adapt(self, 
              support_x: torch.Tensor, 
              support_y: torch.Tensor) -> nn.Module:
        """
        Adapt model to support set using gradient descent.
        
        Returns:
            Adapted model clone
        """
        adapted = self.clone_model()
        adapted.train()
        
        for _ in range(self.n_inner_steps):
            logits = adapted(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=True)
            
            with torch.no_grad():
                for param, grad in zip(adapted.parameters(), grads):
                    param.sub_(self.lr_inner * grad)
        
        return adapted
    
    def meta_train_step(self, 
                        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """
        Perform one meta-training step.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        
        Returns:
            Average query loss
        """
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to support set
            adapted = self.adapt(support_x, support_y)
            
            # Evaluate on query set
            query_logits = adapted(query_x)
            query_loss = F.cross_entropy(query_logits, query_y)
            
            total_loss += query_loss
        
        # Average loss and backprop
        avg_loss = total_loss / len(tasks)
        avg_loss.backward()
        self.meta_optimizer.step()
        
        return avg_loss.item()


def load_all_windows(config: Config, logger) -> Dict[str, List[Dict]]:
    """Load all windows grouped by subject."""
    subjects = get_all_subjects(config.data_path)
    windows_by_subject = {}
    
    logger.info(f"Loading {len(subjects)} subjects...")
    
    pbar = tqdm(subjects, desc="Loading data", unit="subject")
    successful = 0
    total_windows = 0
    
    for subject_folder in pbar:
        signals = load_raw_signals(subject_folder)
        subject_id = signals["subject_id"]
        
        try:
            start, end = get_experiment_time_range(signals)
        except ValueError:
            continue
        
        aligned = align_to_1hz(signals, start, end)
        if aligned is None or len(aligned) == 0:
            continue
        
        event_info = parse_stress_events(
            signals.get("annotation"),
            config.stress_start_events,
            config.stress_stop_events,
            config.baseline_events
        )
        
        windows = create_labeled_windows(
            aligned,
            event_info,
            window_size_sec=config.window_size_sec,
            overlap_ratio=config.overlap_ratio,
            horizons_minutes=config.horizons_minutes,
            skip_first_minutes=config.skip_first_minutes
        )
        
        if windows:
            for w in windows:
                w["subject_id"] = subject_id
            windows_by_subject[subject_id] = windows
            successful += 1
            total_windows += len(windows)
            pbar.set_postfix({"OK": successful, "Windows": total_windows})
    
    pbar.close()
    logger.info(f"Loaded {successful} subjects with {total_windows} windows")
    
    return windows_by_subject


def prepare_window_data(windows: List[Dict], 
                        config: Config,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert windows to tensors."""
    X_list = []
    y_list = []
    
    for window in windows:
        df = window["window_data"]
        
        # Extract 4 channels
        channels = []
        
        # ACC magnitude
        if "acc_magnitude" in df.columns:
            channels.append(df["acc_magnitude"].values)
        else:
            channels.append(np.zeros(len(df)))
        
        # Skin temperature
        if "skin_temp" in df.columns:
            channels.append(df["skin_temp"].values)
        elif "skinTemperature" in df.columns:
            channels.append(df["skinTemperature"].values)
        else:
            channels.append(np.zeros(len(df)))
        
        # EDA
        if "eda_stress_skin" in df.columns:
            channels.append(df["eda_stress_skin"].values)
        elif "stressSkinConductance" in df.columns:
            channels.append(df["stressSkinConductance"].values)
        else:
            channels.append(np.zeros(len(df)))
        
        # PPG mean
        if "ppg_mean" in df.columns:
            channels.append(df["ppg_mean"].values)
        else:
            channels.append(np.zeros(len(df)))
        
        # Stack channels and flatten for MLP
        x = np.stack(channels, axis=0)  # (4, time)
        x = np.nan_to_num(x, nan=0.0)
        x = x.flatten()  # (4 * time,)
        
        X_list.append(x)
        y_list.append(window.get(config.target_label, 0))
    
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y_list), dtype=torch.long).to(device)
    
    return X, y


def meta_train(model: nn.Module,
               windows_by_subject: Dict[str, List[Dict]],
               test_subject: str,
               config: Config,
               device: torch.device,
               logger,
               n_epochs: int = 100,
               n_tasks_per_batch: int = 4,
               k_support: int = 5,
               k_query: int = 10,
               lr_inner: float = 0.01,
               lr_outer: float = 0.001,
               n_inner_steps: int = 5) -> nn.Module:
    """
    Meta-train on all subjects except test subject.
    
    Returns:
        Meta-trained model
    """
    train_subjects = [s for s in windows_by_subject.keys() if s != test_subject]
    
    if len(train_subjects) < 2:
        return model
    
    maml = ManualMAML(model, lr_inner=lr_inner, lr_outer=lr_outer, 
                      n_inner_steps=n_inner_steps)
    
    pbar = tqdm(range(n_epochs), desc="    Meta-training", leave=False, unit="epoch")
    
    for epoch in pbar:
        # Sample tasks
        task_subjects = random.sample(train_subjects, min(n_tasks_per_batch, len(train_subjects)))
        
        tasks = []
        for task_subject in task_subjects:
            windows = windows_by_subject[task_subject]
            
            if len(windows) < k_support + k_query:
                continue
            
            # Sample support and query
            indices = random.sample(range(len(windows)), k_support + k_query)
            support_windows = [windows[i] for i in indices[:k_support]]
            query_windows = [windows[i] for i in indices[k_support:]]
            
            support_x, support_y = prepare_window_data(support_windows, config, device)
            query_x, query_y = prepare_window_data(query_windows, config, device)
            
            tasks.append((support_x, support_y, query_x, query_y))
        
        if tasks:
            loss = maml.meta_train_step(tasks)
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
    
    pbar.close()
    return maml.model


def evaluate_on_subject(model: nn.Module,
                        windows: List[Dict],
                        config: Config,
                        device: torch.device,
                        n_adapt_steps: int = 5,
                        adapt_lr: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a subject with adaptation.
    
    Uses first few windows for adaptation, rest for evaluation.
    """
    if len(windows) < 5:
        X, y = prepare_window_data(windows, config, device)
        with torch.no_grad():
            logits = model(X)
            proba = torch.softmax(logits, dim=-1)[:, 1]
            pred = torch.argmax(logits, dim=-1)
        return y.cpu().numpy(), pred.cpu().numpy(), proba.cpu().numpy()
    
    # Split into adapt and eval sets
    n_adapt = min(5, len(windows) // 4)
    adapt_windows = windows[:n_adapt]
    eval_windows = windows[n_adapt:]
    
    # Clone model for adaptation
    adapted = type(model)(
        input_dim=model.input_dim,
        hidden_dim=model.hidden_dim,
        n_classes=model.n_classes
    )
    adapted.load_state_dict(model.state_dict())
    adapted = adapted.to(device)
    
    # Adapt to this subject
    adapt_x, adapt_y = prepare_window_data(adapt_windows, config, device)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=adapt_lr)
    
    adapted.train()
    for _ in range(n_adapt_steps):
        logits = adapted(adapt_x)
        loss = F.cross_entropy(logits, adapt_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    eval_x, eval_y = prepare_window_data(eval_windows, config, device)
    
    adapted.eval()
    with torch.no_grad():
        logits = adapted(eval_x)
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)
    
    return eval_y.cpu().numpy(), pred.cpu().numpy(), proba.cpu().numpy()


def loso_cross_validation(windows_by_subject: Dict[str, List[Dict]],
                          config: Config,
                          device: torch.device,
                          logger,
                          n_meta_epochs: int = 100) -> Dict:
    """Perform LOSO CV with MAML."""
    subjects = list(windows_by_subject.keys())
    n_subjects = len(subjects)
    
    # Calculate input dimension
    sample_window = list(windows_by_subject.values())[0][0]["window_data"]
    input_dim = 4 * len(sample_window)  # 4 channels * time steps
    
    logger.info(f"\nLOSO CV with {n_subjects} subjects")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Meta epochs per fold: {n_meta_epochs}")
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_subjects = []
    fold_metrics = []
    
    start_time = time.time()
    
    pbar = tqdm(enumerate(subjects), total=n_subjects, desc="LOSO folds", unit="fold")
    
    for fold_idx, test_subject in pbar:
        pbar.set_description(f"Fold {fold_idx+1}/{n_subjects} ({test_subject[:8]}...)")
        
        # Create fresh model for each fold
        model = StressClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            n_classes=2
        ).to(device)
        
        # Meta-train on other subjects
        model = meta_train(
            model,
            windows_by_subject,
            test_subject,
            config,
            device,
            logger,
            n_epochs=n_meta_epochs
        )
        
        # Evaluate on test subject with adaptation
        test_windows = windows_by_subject[test_subject]
        y_true, y_pred, y_proba = evaluate_on_subject(
            model, test_windows, config, device
        )
        
        # Store
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_subjects.extend([test_subject] * len(y_true))
        
        # Fold metrics
        fold_metric = evaluate_predictions(y_true, y_pred, y_proba, "maml")
        fold_metric["subject"] = test_subject
        fold_metrics.append(fold_metric)
        
        fold_auroc = fold_metric.get("auroc", float("nan"))
        pbar.set_postfix({"AUROC": f"{fold_auroc:.3f}", "Test": len(y_true)})
    
    pbar.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/60:.1f} minutes")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_subjects = np.array(all_subjects)
    
    # Aggregate metrics
    aggregate_metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba, "maml")
    fold_aggregated = aggregate_fold_metrics(fold_metrics)
    aggregate_metrics.update(fold_aggregated)
    
    return {
        "metrics": aggregate_metrics,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_proba": all_y_proba,
        "subjects": all_subjects,
        "fold_metrics": fold_metrics
    }


def main():
    """Main MAML training pipeline."""
    
    config = DEFAULT_CONFIG
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("maml", log_file=results_dir / "training.log")
    
    log_experiment_start(logger, "MAML META-LEARNING TRAINING")
    
    if LEARN2LEARN_AVAILABLE:
        logger.info("learn2learn: Available ✓")
    else:
        logger.warning("learn2learn: Not available - using manual MAML implementation")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Load all windows
    logger.info("\n" + "-"*50)
    logger.info("PHASE 1: Loading data")
    logger.info("-"*50)
    
    windows_by_subject = load_all_windows(config, logger)
    
    if not windows_by_subject:
        logger.error("No data loaded!")
        return
    
    total_windows = sum(len(w) for w in windows_by_subject.values())
    n_pos = sum(sum(1 for w in windows if w.get(config.target_label, 0) == 1)
                for windows in windows_by_subject.values())
    
    log_data_summary(
        logger,
        n_subjects=len(windows_by_subject),
        n_windows=total_windows,
        n_features=4,
        n_positive=n_pos,
        n_negative=total_windows - n_pos
    )
    
    # Run LOSO CV
    logger.info("\n" + "-"*50)
    logger.info("PHASE 2: Meta-Learning Training (LOSO CV)")
    logger.info("-"*50)
    
    results = loso_cross_validation(
        windows_by_subject,
        config,
        device,
        logger,
        n_meta_epochs=50
    )
    
    # Log results
    log_model_results(logger, "MAML", results["metrics"])
    
    # Save results
    save_results(results["metrics"], results_dir, "maml")
    save_predictions(
        results["y_true"], results["y_pred"], results["y_proba"],
        results["subjects"], results_dir, "maml"
    )
    
    if len(np.unique(results["y_true"])) > 1:
        plot_results(results["y_true"], results["y_proba"], "maml", results_dir)
    
    fold_df = pd.DataFrame(results["fold_metrics"])
    fold_df.to_csv(results_dir / "maml_fold_metrics.csv", index=False)
    
    log_experiment_end(logger, "MAML META-LEARNING TRAINING")
    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
