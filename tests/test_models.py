# Test scripts for 3DisoDeepPF

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np

from 3disodeeppf.models import CrossModalGNN
from 3disodeeppf.utils import set_random_seed


def test_gnn_model():
    """Test the CrossModalGNN model."""
    set_random_seed(42)

    num_nodes = 100
    num_labels = 50

    model = CrossModalGNN(
        num_nodes=num_nodes,
        num_labels=num_labels,
        embedding_dim=64,
        esm_dim=128,
        motif_dim=32,
        hidden_dim=64,
        num_layers=2,
    )

    # Test forward pass
    batch_size = 10
    node_indices = torch.arange(batch_size)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)

    esm_features = torch.randn(batch_size, 128)
    motif_features = torch.randn(batch_size, 32)

    logits, edge_weights = model(
        node_indices=node_indices,
        edge_index=edge_index,
        motif_features=motif_features,
        esm_features=esm_features,
    )

    assert logits.shape == (batch_size, num_labels), f"Expected shape ({batch_size}, {num_labels}), got {logits.shape}"

    print("✓ CrossModalGNN test passed")


def test_edge_weighting():
    """Test adaptive edge weighting."""
    from 3disodeeppf.models.gnn import AdaptiveEdgeWeighting

    layer = AdaptiveEdgeWeighting()

    seq_sim = torch.tensor([0.8, 0.6, 0.9])
    struct_sim = torch.tensor([0.7, 0.8, 0.5])

    combined, lambda_val = layer(seq_sim, struct_sim)

    assert combined.shape == seq_sim.shape
    assert 0 <= lambda_val.item() <= 1

    print("✓ AdaptiveEdgeWeighting test passed")


def test_focal_loss():
    """Test focal loss."""
    from 3disodeeppf.training.trainer import FocalLoss

    loss_fn = FocalLoss(gamma=2.0)

    logits = torch.randn(10, 50)
    targets = torch.randint(0, 2, (10, 50)).float()

    loss = loss_fn(logits, targets)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("✓ FocalLoss test passed")


def test_metrics():
    """Test evaluation metrics."""
    from 3disodeeppf.evaluation import compute_fmax, compute_aupr

    targets = np.random.randint(0, 2, (100, 20)).astype(float)
    predictions = np.random.rand(100, 20)

    fmax, threshold = compute_fmax(targets, predictions)
    assert 0 <= fmax <= 1, "Fmax should be between 0 and 1"

    aupr = compute_aupr(targets, predictions)
    assert 0 <= aupr <= 1, "AUPR should be between 0 and 1"

    print("✓ Metrics test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 40)
    print("Running 3DisoDeepPF Tests")
    print("=" * 40 + "\n")

    test_gnn_model()
    test_edge_weighting()
    test_focal_loss()
    test_metrics()

    print("\n" + "=" * 40)
    print("All tests passed!")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_all_tests()
