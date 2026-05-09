"""
Protein Dataset Module

Handles loading and preprocessing of protein function prediction datasets.
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import numpy as np


class ProteinDataset:
    """
    Dataset class for protein function prediction.

    Manages loading of protein sequences, labels (GO terms/Pfam domains),
    and train/val/test splits.
    """

    def __init__(
        self,
        data_dir: str,
        task_name: str = "go_bp",
        min_label_count: int = 5,
    ):
        """
        Args:
            data_dir: Directory containing data files
            task_name: Task type ('go_bp', 'go_mf', 'go_cc', 'pfam')
            min_label_count: Minimum number of occurrences for a label to be included
        """
        self.data_dir = data_dir
        self.task_name = task_name

        self.protein_ids = []
        self.prot_id_to_idx = {}
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_labels = 0

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self._load_data(min_label_count)

    def _load_data(self, min_label_count: int):
        """Load and process protein data."""
        labels_file = os.path.join(self.data_dir, f"prot_id_label_{self.task_name}.tsv")

        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        df_labels = pd.read_csv(labels_file, sep="\t")
        self.protein_ids = df_labels["prot_id"].tolist()
        self.prot_id_to_idx = {pid: idx for idx, pid in enumerate(self.protein_ids)}

        # Parse labels
        labels_per_protein = []
        all_labels = []

        for _, row in df_labels.iterrows():
            if pd.isna(row.get("labels", "")) or str(row.get("labels", "")).strip() == "":
                labels_per_protein.append([])
            else:
                labels = str(row["labels"]).split(";")
                labels = [l.strip() for l in labels if l.strip()]
                labels_per_protein.append(labels)
                all_labels.extend(labels)

        # Count label frequencies and filter
        from collections import Counter
        label_counts = Counter(all_labels)
        valid_labels = {l for l, c in label_counts.items() if c >= min_label_count}

        # Create label mapping
        self.label_to_idx = {l: idx for idx, l in enumerate(sorted(valid_labels))}
        self.idx_to_label = {idx: l for l, idx in self.label_to_idx.items()}
        self.num_labels = len(self.label_to_idx)

        # Create binary label matrix
        self.labels = torch.zeros(len(self.protein_ids), self.num_labels, dtype=torch.float)
        self.has_labels = []

        for i, prot_labels in enumerate(labels_per_protein):
            valid_prot_labels = [l for l in prot_labels if l in self.label_to_idx]
            self.has_labels.append(len(valid_prot_labels) > 0)

            for label in valid_prot_labels:
                self.labels[i, self.label_to_idx[label]] = 1

        print(f"Loaded {len(self.protein_ids)} proteins with {self.num_labels} {self.task_name} labels")
        print(f"Proteins with at least one label: {sum(self.has_labels)}")

    def get_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Get train/val/test split indices."""
        labeled_indices = [i for i, has_label in enumerate(self.has_labels) if has_label]

        np.random.seed(42)
        np.random.shuffle(labeled_indices)

        n = len(labeled_indices)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        self.train_indices = labeled_indices[:train_end]
        self.val_indices = labeled_indices[train_end:val_end]
        self.test_indices = labeled_indices[val_end:]

        return self.train_indices, self.val_indices, self.test_indices

    def get_split_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get boolean masks for train/val/test splits."""
        if not self.train_indices:
            self.get_splits()

        num_nodes = len(self.protein_ids)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[self.train_indices] = True
        val_mask[self.val_indices] = True
        test_mask[self.test_indices] = True

        return train_mask, val_mask, test_mask

    def get_labels(self) -> torch.Tensor:
        """Get the label tensor."""
        return self.labels

    def get_label_names(self, indices: List[int]) -> List[List[str]]:
        """Get label names for given protein indices."""
        result = []
        for idx in indices:
            labels = torch.where(self.labels[idx] > 0.5)[0]
            result.append([self.idx_to_label[i.item()] for i in labels])
        return result


class ProteinDatasetFromFiles:
    """
    Dataset that loads from pre-processed files (for demo/small scale).
    """

    def __init__(
        self,
        proteins_file: str,
        labels_file: str,
        network_file: Optional[str] = None,
        splits_file: Optional[str] = None,
    ):
        """
        Args:
            proteins_file: TSV file with protein_id and sequence columns
            labels_file: TSV file with protein_id and labels (semicolon-separated)
            network_file: Optional TSV file with Source, Target, Weight columns
            splits_file: Optional TSV file with protein_id and split (train/val/test)
        """
        self.proteins_df = pd.read_csv(proteins_file, sep="\t")
        self.labels_df = pd.read_csv(labels_file, sep="\t")

        self.protein_ids = self.proteins_df["protein_id"].tolist()
        self.prot_id_to_idx = {pid: idx for idx, pid in enumerate(self.protein_ids)}

        # Load labels
        self._process_labels()

        # Load splits if provided
        if splits_file and os.path.exists(splits_file):
            self._load_splits(splits_file)
        else:
            self.get_splits()

        # Load network if provided
        if network_file and os.path.exists(network_file):
            self._load_network(network_file)

    def _process_labels(self):
        """Process label data."""
        label_counts = {}
        protein_labels = {}

        for _, row in self.labels_df.iterrows():
            prot_id = row["protein_id"]
            labels_str = str(row.get("labels", ""))
            labels = [l.strip() for l in labels_str.split(";") if l.strip()]

            protein_labels[prot_id] = labels
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

        # Filter labels appearing less than 5 times
        valid_labels = {l for l, c in label_counts.items() if c >= 5}
        self.label_to_idx = {l: idx for idx, l in enumerate(sorted(valid_labels))}
        self.idx_to_label = {idx: l for l, idx in self.label_to_idx.items()}
        self.num_labels = len(self.label_to_idx)

        # Create binary matrix
        self.labels = torch.zeros(len(self.protein_ids), self.num_labels, dtype=torch.float)
        self.has_labels = []

        for i, prot_id in enumerate(self.protein_ids):
            labels = protein_labels.get(prot_id, [])
            valid = [l for l in labels if l in self.label_to_idx]
            self.has_labels.append(len(valid) > 0)

            for label in valid:
                self.labels[i, self.label_to_idx[label]] = 1

    def _load_splits(self, splits_file: str):
        """Load train/val/test splits."""
        splits_df = pd.read_csv(splits_file, sep="\t")
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        for _, row in splits_df.iterrows():
            prot_id = row["protein_id"]
            if prot_id not in self.prot_id_to_idx:
                continue

            idx = self.prot_id_to_idx[prot_id]
            split = row.get("split", "train")

            if split == "train":
                self.train_indices.append(idx)
            elif split == "val":
                self.val_indices.append(idx)
            elif split == "test":
                self.test_indices.append(idx)

    def get_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Auto-generate train/val/test splits."""
        labeled_indices = [i for i, has_label in enumerate(self.has_labels) if has_label]
        np.random.seed(42)
        np.random.shuffle(labeled_indices)

        n = len(labeled_indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        self.train_indices = labeled_indices[:train_end]
        self.val_indices = labeled_indices[train_end:val_end]
        self.test_indices = labeled_indices[val_end:]

        return self.train_indices, self.val_indices, self.test_indices

    def _load_network(self, network_file: str):
        """Load protein similarity network."""
        self.network_df = pd.read_csv(network_file, sep="\t")

    def get_split_masks(self):
        """Get boolean masks for splits."""
        num_nodes = len(self.protein_ids)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[self.train_indices] = True
        val_mask[self.val_indices] = True
        test_mask[self.test_indices] = True

        return train_mask, val_mask, test_mask
