"""
Training Module for 3DisoDeepPF

Handles model training with configurable loss functions,
optimizers, and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification.

    As described in the paper, this helps address the label imbalance
    inherent in protein function datasets.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            gamma: Focusing parameter that down-weights easy examples
            pos_weight: Optional positive class weights for BCE
        """
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels

        Returns:
            Loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss with class balance."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class Trainer:
    """
    Training loop for CrossModalGNN model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        labels: torch.Tensor,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            model: The GNN model to train
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            test_mask: Boolean mask for test nodes
            labels: Ground truth labels (num_nodes x num_labels)
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.labels = labels.to(self.device)

        # Compute positive weights for class imbalance
        num_pos = labels[train_mask].sum(dim=0)
        num_neg = train_mask.sum().item() - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=100)
        self.pos_weight = pos_weight.to(self.device)

        # Loss function
        self.loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=self.pos_weight)
        # Fallback to weighted BCE if pos_weight is problematic
        self.loss_fn = WeightedBCELoss(pos_weight=self.pos_weight, label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=300, eta_min=1e-5)

        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.history = {"train_loss": [], "val_f1": [], "test_f1": []}

    def train_epoch(self, node_indices: torch.Tensor) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        node_indices = node_indices.to(self.device)

        logits, _ = self.model(node_indices=node_indices)

        loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, node_indices: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on given nodes."""
        self.model.eval()

        node_indices = node_indices.to(self.device)
        logits, _ = self.model(node_indices=node_indices)
        probs = torch.sigmoid(logits[mask])

        targets = self.labels[mask].cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Compute metrics
        from sklearn.metrics import f1_score, precision_score, recall_score

        preds_binary = (probs_np > 0.5).astype(int)

        f1_macro = f1_score(targets, preds_binary, average="macro", zero_division=0)
        f1_micro = f1_score(targets, preds_binary, average="micro", zero_division=0)
        precision = precision_score(targets, preds_binary, average="macro", zero_division=0)
        recall = recall_score(targets, preds_binary, average="macro", zero_division=0)

        return {
            "loss": 0.0,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision": precision,
            "recall": recall,
        }

    def train(
        self,
        num_epochs: int,
        node_indices: torch.Tensor,
        print_every: int = 10,
    ) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs: Number of training epochs
            node_indices: All node indices
            print_every: Print progress every N epochs

        Returns:
            Training history dictionary
        """
        node_indices = node_indices.to(self.device)

        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Train
            train_loss = self.train_epoch(node_indices)

            # Evaluate
            val_metrics = self.evaluate(node_indices, self.val_mask)
            test_metrics = self.evaluate(node_indices, self.test_mask)

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["test_f1"].append(test_metrics["f1_macro"])

            # Save best model
            if val_metrics["f1_macro"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_macro"]
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % print_every == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
                print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")

            self.scheduler.step()

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    @torch.no_grad()
    def predict(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get predictions for given node indices."""
        self.model.eval()
        node_indices = node_indices.to(self.device)

        logits, _ = self.model(node_indices=node_indices)
        probs = torch.sigmoid(logits)

        return probs

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "best_val_f1": self.best_val_f1,
                "history": self.history,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.history = checkpoint.get("history", {})
        print(f"Model loaded from {path}")
