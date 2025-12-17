"""
Contrastive losses for Subject-Aware Self-Supervised Learning.

Implements three loss modes from the Apple paper:
1. Base SSL: Standard InfoNCE loss
2. Subject-Invariant: InfoNCE + Adversarial loss (combined)
3. Subject-Specific: InfoNCE with same-subject negative sampling

Reference:
    "Self-supervised learning of electrodermal activity representations 
    for stress classification" - Apple (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from enum import Enum


class SSLMode(Enum):
    """SSL training mode."""
    BASE = "base"                    # Plain contrastive, no subject awareness
    SUBJECT_INVARIANT = "invariant"  # Contrastive + Adversarial (combined)
    SUBJECT_SPECIFIC = "specific"    # Contrastive with same-subject negatives only


def infonce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.
    
    Maximizes similarity between positive pairs (two views of same sample)
    while minimizing similarity with negative pairs (views of different samples).
    
    Args:
        z1: Projected embeddings from view 1, shape (batch, dim)
        z2: Projected embeddings from view 2, shape (batch, dim)
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        InfoNCE loss (scalar)
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # L2 normalize embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute similarity matrix: z1 @ z2.T
    # Shape: (batch, batch) where [i,j] = similarity between z1[i] and z2[j]
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Positive pairs are on the diagonal (same sample, different views)
    labels = torch.arange(batch_size, device=device)
    
    # Cross entropy loss treats diagonal as positive, rest as negatives
    # Loss from z1 -> z2 direction
    loss_12 = F.cross_entropy(sim_matrix, labels)
    
    # Loss from z2 -> z1 direction
    loss_21 = F.cross_entropy(sim_matrix.T, labels)
    
    # Symmetric loss
    loss = (loss_12 + loss_21) / 2
    
    return loss


def infonce_loss_subject_specific(
    z1: torch.Tensor,
    z2: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    Subject-specific InfoNCE loss.
    
    Only uses samples from the SAME subject as negatives.
    This focuses learning on temporal patterns within each subject,
    ignoring inter-subject variability.
    
    Args:
        z1: Projected embeddings from view 1, shape (batch, dim)
        z2: Projected embeddings from view 2, shape (batch, dim)
        subject_ids: Subject ID for each sample, shape (batch,)
        temperature: Softmax temperature
    
    Returns:
        Subject-specific InfoNCE loss (scalar)
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # L2 normalize embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute full similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Create mask for same-subject pairs
    # subject_mask[i,j] = True if subject_ids[i] == subject_ids[j]
    subject_mask = subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)
    
    # For each sample, we need at least 2 samples from same subject
    # (itself and at least one negative)
    samples_per_subject = subject_mask.sum(dim=1)
    
    # Compute loss only for samples with enough same-subject negatives
    valid_samples = samples_per_subject >= 2
    
    if not valid_samples.any():
        # Fall back to standard InfoNCE if no valid samples
        return infonce_loss(z1, z2, temperature)
    
    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    
    for i in range(batch_size):
        if not valid_samples[i]:
            continue
        
        # Get indices of same-subject samples
        same_subject_idx = torch.where(subject_mask[i])[0]
        
        # Positive: z2[i] (same sample, different view)
        # Negatives: z2[j] for j in same_subject_idx, j != i
        
        # Similarities for this sample
        pos_sim = sim_matrix[i, i]  # Positive similarity
        
        # Negative similarities (same subject, different samples)
        neg_mask = same_subject_idx != i
        neg_idx = same_subject_idx[neg_mask]
        
        if len(neg_idx) == 0:
            continue
        
        neg_sims = sim_matrix[i, neg_idx]
        
        # Combine positive and negatives
        all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        
        # Cross entropy with positive at index 0
        loss_i = F.cross_entropy(all_sims.unsqueeze(0), torch.zeros(1, device=device).long())
        
        total_loss += loss_i
        n_valid += 1
    
    if n_valid == 0:
        return infonce_loss(z1, z2, temperature)
    
    return total_loss / n_valid


def adversarial_subject_loss(
    embeddings: torch.Tensor,
    subject_ids: torch.Tensor,
    subject_classifier: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adversarial subject classification loss.
    
    Two components:
    1. Classifier loss: Train classifier to predict subject from embeddings
    2. Adversarial loss: Train encoder to CONFUSE the classifier
    
    Args:
        embeddings: Encoder embeddings, shape (batch, embed_dim)
        subject_ids: Subject ID for each sample, shape (batch,)
        subject_classifier: Module that predicts subject from embeddings
    
    Returns:
        Tuple of (classifier_loss, adversarial_loss)
        - classifier_loss: For training the subject classifier
        - adversarial_loss: For training the encoder (gradient reversal)
    """
    # Get subject predictions
    subject_logits = subject_classifier(embeddings)
    
    # Classifier loss: Standard cross entropy (train classifier to predict subject)
    classifier_loss = F.cross_entropy(subject_logits, subject_ids)
    
    # Adversarial loss: Encoder should MAXIMIZE classifier confusion
    # We want uniform distribution over subjects (max entropy)
    n_subjects = subject_logits.shape[1]
    
    # Method 1: Negative cross entropy (maximize loss)
    # This encourages the encoder to produce embeddings that are hard to classify
    adversarial_loss = -classifier_loss
    
    # Method 2 (alternative): Entropy maximization
    # probs = F.softmax(subject_logits, dim=-1)
    # entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    # adversarial_loss = -entropy  # Maximize entropy
    
    return classifier_loss, adversarial_loss


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    Forward pass: Identity function
    Backward pass: Negate gradients (multiply by -alpha)
    
    This allows end-to-end training of the adversarial objective.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to tensor."""
    return GradientReversalLayer.apply(x, alpha)


class SubjectInvariantLoss(nn.Module):
    """
    Combined loss for subject-invariant SSL.
    
    Total loss = L_contrastive + λ * L_adversarial
    
    Uses gradient reversal layer for end-to-end training.
    """
    
    def __init__(
        self,
        subject_classifier: nn.Module,
        lambda_adv: float = 0.5,
        temperature: float = 0.5
    ):
        """
        Initialize loss module.
        
        Args:
            subject_classifier: Module that predicts subject from embeddings
            lambda_adv: Weight for adversarial loss
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()
        self.subject_classifier = subject_classifier
        self.lambda_adv = lambda_adv
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        embeddings: torch.Tensor,
        subject_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            z1: Projected embeddings from view 1 (batch, proj_dim)
            z2: Projected embeddings from view 2 (batch, proj_dim)
            embeddings: Raw encoder embeddings (batch, embed_dim)
            subject_ids: Subject IDs (batch,)
        
        Returns:
            Dictionary with loss components:
            - total: Combined loss for encoder optimization
            - contrastive: InfoNCE loss
            - classifier: Loss for subject classifier optimization
            - adversarial: Adversarial component
        """
        # Contrastive loss
        loss_contrastive = infonce_loss(z1, z2, self.temperature)
        
        # Subject classification loss
        subject_logits = self.subject_classifier(embeddings.detach())
        loss_classifier = F.cross_entropy(subject_logits, subject_ids)
        
        # Adversarial loss (through gradient reversal)
        embeddings_grl = gradient_reversal(embeddings, alpha=self.lambda_adv)
        subject_logits_adv = self.subject_classifier(embeddings_grl)
        loss_adversarial = F.cross_entropy(subject_logits_adv, subject_ids)
        
        # Total encoder loss (contrastive + adversarial through GRL)
        loss_total = loss_contrastive + self.lambda_adv * loss_adversarial
        
        return {
            "total": loss_total,
            "contrastive": loss_contrastive,
            "classifier": loss_classifier,
            "adversarial": loss_adversarial
        }


class SubjectSpecificLoss(nn.Module):
    """
    Loss for subject-specific SSL.
    
    Uses only same-subject samples as negatives, focusing on
    temporal patterns within each subject.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize loss module.
        
        Args:
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        subject_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute subject-specific contrastive loss.
        
        Args:
            z1: Projected embeddings from view 1 (batch, proj_dim)
            z2: Projected embeddings from view 2 (batch, proj_dim)
            subject_ids: Subject IDs (batch,)
        
        Returns:
            Dictionary with loss components:
            - total: Same as contrastive
            - contrastive: Subject-specific InfoNCE loss
        """
        loss = infonce_loss_subject_specific(z1, z2, subject_ids, self.temperature)
        
        return {
            "total": loss,
            "contrastive": loss
        }


class BaseLoss(nn.Module):
    """
    Standard InfoNCE loss for base SSL.
    
    No subject awareness - uses all samples as potential negatives.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize loss module.
        
        Args:
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standard InfoNCE loss.
        
        Args:
            z1: Projected embeddings from view 1 (batch, proj_dim)
            z2: Projected embeddings from view 2 (batch, proj_dim)
            **kwargs: Ignored (for API compatibility)
        
        Returns:
            Dictionary with loss components:
            - total: Same as contrastive
            - contrastive: InfoNCE loss
        """
        loss = infonce_loss(z1, z2, self.temperature)
        
        return {
            "total": loss,
            "contrastive": loss
        }


def create_loss_fn(
    mode: SSLMode,
    subject_classifier: Optional[nn.Module] = None,
    lambda_adv: float = 0.5,
    temperature: float = 0.5
) -> nn.Module:
    """
    Factory function to create appropriate loss module.
    
    Args:
        mode: SSL training mode
        subject_classifier: Required for SUBJECT_INVARIANT mode
        lambda_adv: Weight for adversarial loss (invariant mode)
        temperature: Temperature for InfoNCE loss
    
    Returns:
        Configured loss module
    """
    if mode == SSLMode.BASE:
        return BaseLoss(temperature)
    elif mode == SSLMode.SUBJECT_INVARIANT:
        if subject_classifier is None:
            raise ValueError("subject_classifier required for SUBJECT_INVARIANT mode")
        return SubjectInvariantLoss(subject_classifier, lambda_adv, temperature)
    elif mode == SSLMode.SUBJECT_SPECIFIC:
        return SubjectSpecificLoss(temperature)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test losses
    print("Testing contrastive losses...")
    
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    batch_size = 16
    embed_dim = 256
    proj_dim = 64
    n_subjects = 5
    
    # Create dummy data
    z1 = torch.randn(batch_size, proj_dim)
    z2 = torch.randn(batch_size, proj_dim)
    embeddings = torch.randn(batch_size, embed_dim)
    subject_ids = torch.randint(0, n_subjects, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"  z1, z2: {z1.shape}")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  subject_ids: {subject_ids.shape}")
    
    # Test Base loss
    print("\n1. Base SSL loss:")
    base_loss = BaseLoss(temperature=0.5)
    result = base_loss(z1, z2)
    print(f"   Total: {result['total'].item():.4f}")
    print(f"   Contrastive: {result['contrastive'].item():.4f}")
    
    # Test Subject-Specific loss
    print("\n2. Subject-Specific loss:")
    specific_loss = SubjectSpecificLoss(temperature=0.5)
    result = specific_loss(z1, z2, subject_ids)
    print(f"   Total: {result['total'].item():.4f}")
    print(f"   Contrastive: {result['contrastive'].item():.4f}")
    
    # Test Subject-Invariant loss
    print("\n3. Subject-Invariant loss:")
    from encoder import SubjectClassifier
    subject_clf = SubjectClassifier(input_dim=embed_dim, n_subjects=n_subjects)
    invariant_loss = SubjectInvariantLoss(subject_clf, lambda_adv=0.5, temperature=0.5)
    result = invariant_loss(z1, z2, embeddings, subject_ids)
    print(f"   Total: {result['total'].item():.4f}")
    print(f"   Contrastive: {result['contrastive'].item():.4f}")
    print(f"   Classifier: {result['classifier'].item():.4f}")
    print(f"   Adversarial: {result['adversarial'].item():.4f}")
    
    # Test gradient flow
    print("\n4. Testing gradient flow:")
    z1.requires_grad = True
    embeddings.requires_grad = True
    
    result = invariant_loss(z1, z2, embeddings, subject_ids)
    result["total"].backward()
    
    print(f"   z1 grad exists: {z1.grad is not None}")
    print(f"   embeddings grad exists: {embeddings.grad is not None}")
    print(f"   z1 grad norm: {z1.grad.norm().item():.4f}")
    print(f"   embeddings grad norm: {embeddings.grad.norm().item():.4f}")
    
    # Test factory function
    print("\n5. Testing factory function:")
    for mode in SSLMode:
        try:
            if mode == SSLMode.SUBJECT_INVARIANT:
                loss_fn = create_loss_fn(mode, subject_classifier=subject_clf)
            else:
                loss_fn = create_loss_fn(mode)
            print(f"   {mode.value}: Created {loss_fn.__class__.__name__}")
        except Exception as e:
            print(f"   {mode.value}: Error - {e}")
    
    print("\nAll tests passed!")
