"""
Evaluation Metrics for Protein Function Prediction

Implements standard metrics including Fmax, AUPR, and per-label evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    hamming_loss,
)


def compute_fmax(
    targets: np.ndarray,
    predictions: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute maximum F-score (Fmax).

    Fmax is the maximum precision-recall balanced F-score achieved
    across different decision thresholds.

    Args:
        targets: Binary ground truth labels (num_samples x num_labels)
        predictions: Prediction probabilities (num_samples x num_labels)
        thresholds: Optional custom thresholds to evaluate

    Returns:
        Tuple of (fmax, best_threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    fmax = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        preds_binary = (predictions >= threshold).astype(int)

        # Compute precision and recall per label
        f1_per_label = []
        for i in range(targets.shape[1]):
            t = targets[:, i]
            p = preds_binary[:, i]

            tp = np.sum((p == 1) & (t == 1))
            fp = np.sum((p == 1) & (t == 0))
            fn = np.sum((p == 0) & (t == 1))

            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_per_label.append(f1)

        if f1_per_label:
            avg_f1 = np.mean(f1_per_label)
            if avg_f1 > fmax:
                fmax = avg_f1
                best_threshold = threshold

    return fmax, best_threshold


def compute_aupr(
    targets: np.ndarray,
    predictions: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Compute Area Under the Precision-Recall Curve (AUPR).

    Args:
        targets: Binary ground truth labels
        predictions: Prediction probabilities
        average: 'macro', 'micro', or 'samples'

    Returns:
        AUPR score
    """
    if targets.sum() == 0:
        return 0.0

    if average == "macro":
        aupr_scores = []
        for i in range(targets.shape[1]):
            t = targets[:, i]
            p = predictions[:, i]

            if t.sum() > 0 and t.sum() < len(t):
                precision, recall, _ = precision_recall_curve(t, p)
                aupr = auc(recall, precision)
                aupr_scores.append(aupr)

        return np.mean(aupr_scores) if aupr_scores else 0.0

    elif average == "micro":
        precision, recall, _ = precision_recall_curve(targets.ravel(), predictions.ravel())
        return auc(recall, precision)

    else:
        raise ValueError(f"Unknown average type: {average}")


def compute_per_label_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Compute metrics for each label individually.

    Args:
        targets: Binary ground truth
        predictions: Prediction probabilities
        threshold: Decision threshold
        label_names: Optional list of label names

    Returns:
        List of dictionaries with per-label metrics
    """
    preds_binary = (predictions >= threshold).astype(int)

    results = []
    for i in range(targets.shape[1]):
        t = targets[:, i]
        p = preds_binary[:, i]
        p_scores = predictions[:, i]

        tp = np.sum((p == 1) & (t == 1))
        fp = np.sum((p == 1) & (t == 0))
        fn = np.sum((p == 0) & (t == 1))
        tn = np.sum((p == 0) & (t == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if t.sum() > 0 and t.sum() < len(t):
            prec_curve, rec_curve, _ = precision_recall_curve(t, p_scores)
            aupr = auc(rec_curve, prec_curve)
        else:
            aupr = 0.0

        result = {
            "label_id": i,
            "label_name": label_names[i] if label_names else f"label_{i}",
            "num_positives": int(t.sum()),
            "num_predictions": int(p.sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "aupr": float(aupr),
        }
        results.append(result)

    return results


def evaluate_model(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        targets: Ground truth labels (numpy or torch tensor)
        predictions: Prediction probabilities
        threshold: Decision threshold
        label_names: Optional label names

    Returns:
        Dictionary with all metrics
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    preds_binary = (predictions >= threshold).astype(int)

    # Overall metrics
    metrics = {
        "f1_macro": f1_score(targets, preds_binary, average="macro", zero_division=0),
        "f1_micro": f1_score(targets, preds_binary, average="micro", zero_division=0),
        "precision_macro": precision_score(targets, preds_binary, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds_binary, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(targets, preds_binary),
        "aupr_macro": compute_aupr(targets, predictions, average="macro"),
        "aupr_micro": compute_aupr(targets, predictions, average="micro"),
    }

    # Compute Fmax
    fmax, best_thresh = compute_fmax(targets, predictions)
    metrics["fmax"] = fmax
    metrics["best_threshold"] = best_thresh

    # Per-label metrics
    metrics["per_label"] = compute_per_label_metrics(targets, predictions, threshold, label_names)

    return metrics


def print_evaluation_report(metrics: Dict, top_k: int = 10):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Fmax:                    {metrics['fmax']:.4f}")
    print(f"  F1 (macro):              {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):              {metrics['f1_micro']:.4f}")
    print(f"  AUPR (macro):            {metrics['aupr_macro']:.4f}")
    print(f"  AUPR (micro):            {metrics['aupr_micro']:.4f}")
    print(f"  Precision (macro):       {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):          {metrics['recall_macro']:.4f}")
    print(f"  Hamming Loss:            {metrics['hamming_loss']:.4f}")
    print(f"  Best Threshold:          {metrics['best_threshold']:.4f}")

    print(f"\nTop {top_k} Labels by AUPR:")
    per_label = metrics.get("per_label", [])
    if per_label:
        sorted_labels = sorted(per_label, key=lambda x: x["aupr"], reverse=True)[:top_k]
        print(f"  {'Label':<30} {'AUPR':>8} {'F1':>8} {'Positives':>10}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
        for item in sorted_labels:
            print(f"  {item['label_name']:<30} {item['aupr']:>8.4f} {item['f1']:>8.4f} {item['num_positives']:>10}")

    print("\n" + "=" * 60)
