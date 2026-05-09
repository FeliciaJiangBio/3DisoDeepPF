"""
3DisoDeepPF Demo: Protein Function Prediction

This script demonstrates the usage of the 3DisoDeepPF framework
on a synthetic small-scale dataset for illustration purposes.

For production use, please prepare your own data following the format
described in the documentation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from 3disodeeppf.models import CrossModalGNN
from 3disodeeppf.data import ProteinGraph
from 3disodeeppf.training import Trainer
from 3disodeeppf.evaluation import evaluate_model, print_evaluation_report
from 3disodeeppf.utils import set_random_seed, print_model_info

set_random_seed(42)


def create_synthetic_data(num_proteins: int = 500, num_labels: int = 50, avg_degree: int = 15):
    """
    Create synthetic protein data for demonstration.

    In a real use case, you would load this from your dataset files.

    Args:
        num_proteins: Number of synthetic proteins
        num_labels: Number of GO terms/Pfam domains
        avg_degree: Average graph connectivity

    Returns:
        Dictionary with synthetic data
    """
    print(f"Creating synthetic data: {num_proteins} proteins, {num_labels} labels")

    # Generate protein IDs
    protein_ids = [f"PROT_{i:05d}" for i in range(num_proteins)]

    # Generate random sequences (for demo purposes only)
    sequences = {
        pid: "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=100))
        for pid in protein_ids
    }

    # Generate label matrix with some structure (proteins sharing GO terms)
    label_matrix = np.zeros((num_proteins, num_labels), dtype=np.float32)

    # Create label clusters (simulating biological correlation)
    np.random.seed(42)
    for cluster_id in range(5):
        start_label = cluster_id * (num_labels // 5)
        end_label = start_label + (num_labels // 5)

        # Proteins in this cluster have higher probability of these labels
        cluster_proteins = np.random.choice(
            num_proteins, size=num_proteins // 5, replace=False
        )
        for prot_idx in cluster_proteins:
            for label_idx in range(start_label, end_label):
                if np.random.random() < 0.3:
                    label_matrix[prot_idx, label_idx] = 1.0

    # Ensure some labels are present
    for label_idx in range(num_labels):
        if label_matrix[:, label_idx].sum() < 1:
            # Add at least 2 positive examples per label
            positives = np.random.choice(num_proteins, size=2, replace=False)
            label_matrix[positives, label_idx] = 1.0

    # Generate graph edges
    edges = []
    edge_weights = []

    for i in range(num_proteins):
        num_neighbors = np.random.poisson(avg_degree)
        num_neighbors = min(num_neighbors, num_proteins - 1)

        neighbors = np.random.choice(
            [j for j in range(num_proteins) if j != i],
            size=num_neighbors,
            replace=False,
        )

        for j in neighbors:
            weight = np.random.uniform(0.3, 1.0)
            edges.append([i, j])
            edge_weights.append(weight)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)

    # Create random motif features (simulating Pfam domain encoding)
    motif_features = torch.randn(num_proteins, 128) * 0.1

    # Create ESM-like features (mean-pooled sequence embeddings)
    esm_features = torch.randn(num_proteins, 512) * 0.1

    data = {
        "protein_ids": protein_ids,
        "sequences": sequences,
        "labels": torch.from_numpy(label_matrix),
        "edge_index": edge_index,
        "edge_weights": edge_weights_tensor,
        "motif_features": motif_features,
        "esm_features": esm_features,
    }

    print(f"Created synthetic dataset:")
    print(f"  Proteins: {num_proteins}")
    print(f"  Labels: {num_labels}")
    print(f"  Edges: {len(edges)}")
    print(f"  Labeled proteins: {(label_matrix.sum(axis=1) > 0).sum()}")

    return data


def run_demo(
    num_proteins: int = 500,
    num_labels: int = 50,
    hidden_dim: int = 128,
    epochs: int = 50,
):
    """
    Run the demo pipeline.

    Args:
        num_proteins: Number of synthetic proteins
        num_labels: Number of synthetic labels
        hidden_dim: Hidden dimension for the model
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("3DisoDeepPF Demo: Protein Function Prediction")
    print("=" * 60)

    # Create synthetic data
    data = create_synthetic_data(num_proteins=num_proteins, num_labels=num_labels)

    protein_ids = data["protein_ids"]
    num_nodes = len(protein_ids)

    # Create train/val/test splits
    labeled_indices = [i for i in range(num_nodes) if data["labels"][i].sum() > 0]
    np.random.seed(42)
    np.random.shuffle(labeled_indices)

    n = len(labeled_indices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_indices = labeled_indices[:train_end]
    val_indices = labeled_indices[train_end:val_end]
    test_indices = labeled_indices[val_end:]

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = CrossModalGNN(
        num_nodes=num_nodes,
        num_labels=num_labels,
        num_aux_labels=0,
        embedding_dim=64,
        esm_dim=512,
        motif_dim=128,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        use_auxiliary=False,
        use_esm=True,
    )

    print_model_info(model, "CrossModalGNN")

    # Move data to device
    node_indices = torch.arange(num_nodes, device=device)
    edge_index = data["edge_index"].to(device)
    edge_weights = data["edge_weights"].to(device)
    labels = data["labels"].to(device)
    esm_features = data["esm_features"].to(device)
    motif_features = data["motif_features"].to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labels=labels,
        device=device,
        learning_rate=0.005,
        weight_decay=1e-4,
        focal_gamma=2.0,
    )

    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(
        num_epochs=epochs,
        node_indices=node_indices,
        print_every=10,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    predictions = trainer.predict(node_indices)

    test_targets = labels[test_mask].cpu().numpy()
    test_predictions = predictions[test_mask].cpu().numpy()

    metrics = evaluate_model(
        targets=test_targets,
        predictions=test_predictions,
        threshold=0.5,
    )

    print_evaluation_report(metrics, top_k=10)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "demo_predictions.tsv")
    with open(results_file, "w") as f:
        f.write("protein_id\tpredicted_labels\n")
        for i, prot_id in enumerate(protein_ids):
            pred_labels = torch.where(predictions[i] > 0.5)[0]
            label_str = ";".join([f"LABEL_{idx}" for idx in pred_labels.tolist()])
            f.write(f"{prot_id}\t{label_str}\n")

    print(f"\nResults saved to {results_file}")
    print("\nDemo completed successfully!")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3DisoDeepPF Demo")
    parser.add_argument("--num_proteins", type=int, default=500, help="Number of proteins")
    parser.add_argument("--num_labels", type=int, default=50, help="Number of labels")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    run_demo(
        num_proteins=args.num_proteins,
        num_labels=args.num_labels,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )
