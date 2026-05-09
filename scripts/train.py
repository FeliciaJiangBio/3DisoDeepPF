#!/usr/bin/env python
"""
Main training script for 3DisoDeepPF

This script demonstrates training the model on your own dataset.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import argparse
import numpy as np
from tqdm import tqdm

from 3disodeeppf.models import CrossModalGNN
from 3disodeeppf.models.esm_encoder import ESMEncoder
from 3disodeeppf.data import ProteinDataset
from 3disodeeppf.training import Trainer
from 3disodeeppf.evaluation import evaluate_model, print_evaluation_report
from 3disodeeppf.utils import set_random_seed, print_model_info


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3DisoDeepPF model")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing data files")
    parser.add_argument("--task", type=str, default="go_bp",
                       choices=["go_bp", "go_mf", "go_cc", "pfam"],
                       help="Target task")

    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=128,
                       help="Node embedding dimension")
    parser.add_argument("--esm_dim", type=int, default=512,
                       help="ESM feature dimension")
    parser.add_argument("--motif_dim", type=int, default=128,
                       help="Motif feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.4,
                       help="Dropout rate")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for training")

    # ESM arguments
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D",
                       choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                               "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"],
                       help="ESM model variant")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("3DisoDeepPF Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Data directory: {args.data_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ProteinDataset(
        data_dir=args.data_dir,
        task_name=args.task,
    )

    protein_ids = dataset.protein_ids
    num_nodes = len(protein_ids)
    num_labels = dataset.num_labels

    print(f"Proteins: {num_nodes}")
    print(f"Labels: {num_labels}")

    # Get splits
    train_mask, val_mask, test_mask = dataset.get_split_masks()
    labels = dataset.get_labels()

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = CrossModalGNN(
        num_nodes=num_nodes,
        num_labels=num_labels,
        embedding_dim=args.embedding_dim,
        esm_dim=args.esm_dim,
        motif_dim=args.motif_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_esm=True,
        use_auxiliary=False,
    )

    print_model_info(model, "CrossModalGNN")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labels=labels,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    node_indices = torch.arange(num_nodes, device=device)

    history = trainer.train(
        num_epochs=args.epochs,
        node_indices=node_indices,
        print_every=10,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    predictions = trainer.predict(node_indices)

    test_targets = labels[test_mask].cpu().numpy()
    test_predictions = predictions[test_mask].cpu().numpy()

    metrics = evaluate_model(
        targets=test_targets,
        predictions=test_predictions,
        threshold=0.5,
    )

    print_evaluation_report(metrics, top_k=10)

    # Save model
    model_path = os.path.join(args.output_dir, f"model_{args.task}.pt")
    trainer.save_model(model_path)

    print("\nTraining completed!")
    return metrics


if __name__ == "__main__":
    main()
