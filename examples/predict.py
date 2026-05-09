#!/usr/bin/env python
"""
Predict protein function for new sequences.

This script demonstrates how to use a trained 3DisoDeepPF model
to predict GO terms or Pfam domains for new protein sequences.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import argparse
from Bio import SeqIO

from 3disodeeppf.models import CrossModalGNN
from 3disodeeppf.models.esm_encoder import ESMEncoder
from 3disodeeppf.utils import get_device


def load_sequences(fasta_file: str):
    """Load protein sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def predict(
    model_path: str,
    fasta_file: str,
    output_file: str = None,
    esm_model: str = "esm2_t12_35M_UR50D",
    device: str = None,
):
    """
    Predict protein function for sequences in a FASTA file.

    Args:
        model_path: Path to trained model checkpoint
        fasta_file: Path to input FASTA file
        output_file: Path to output file (optional)
        esm_model: ESM model to use for encoding
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = get_device()

    print(f"Loading sequences from {fasta_file}...")
    sequences = load_sequences(fasta_file)
    print(f"Loaded {len(sequences)} sequences")

    # Initialize ESM encoder
    print(f"Loading ESM model: {esm_model}")
    encoder = ESMEncoder(model_name=esm_model, device=torch.device(device))

    # Encode sequences
    print("Encoding sequences with ESM...")
    esm_features = encoder.encode_sequences(sequences)

    # Load model (simplified - in practice would load trained checkpoint)
    num_proteins = len(sequences)
    num_labels = 100  # Would be actual number of labels

    model = CrossModalGNN(
        num_nodes=num_proteins,
        num_labels=num_labels,
        embedding_dim=128,
        esm_dim=encoder.get_embedding_dim(),
        motif_dim=128,
        hidden_dim=256,
        use_esm=True,
        use_auxiliary=False,
    )

    model = model.to(device)
    model.eval()

    # Load checkpoint if provided
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from {model_path}")

    # Get predictions
    print("Generating predictions...")
    with torch.no_grad():
        node_indices = torch.arange(num_proteins, device=device)
        logits, _ = model(node_indices=node_indices)
        probs = torch.sigmoid(logits)

    # Output results
    results = []
    for i, (prot_id, seq) in enumerate(sequences.items()):
        pred_probs = probs[i]
        top_indices = torch.argsort(pred_probs, descending=True)[:10]
        top_labels = [(f"LABEL_{idx.item()}", pred_probs[idx].item()) for idx in top_indices]

        results.append({
            "protein_id": prot_id,
            "sequence_length": len(seq),
            "predictions": top_labels,
        })

    # Print/save results
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    for result in results:
        print(f"\n{result['protein_id']} (length: {result['sequence_length']})")
        print("  Top predictions:")
        for label, prob in result["predictions"][:5]:
            print(f"    {label}: {prob:.4f}")

    if output_file:
        with open(output_file, "w") as f:
            f.write("protein_id\tpredicted_labels\tconfidences\n")
            for result in results:
                labels = ";".join([f"{l[0]}({l[1]:.3f})" for l in result["predictions"]])
                f.write(f"{result['protein_id']}\t{labels}\n")
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict protein function")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D",
                       choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                               "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"])

    args = parser.parse_args()

    predict(
        model_path=args.model,
        fasta_file=args.input,
        output_file=args.output,
        esm_model=args.esm_model,
        device=args.device,
    )
