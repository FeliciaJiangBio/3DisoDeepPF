"""
Protein Similarity Graph Construction

This module handles construction of protein similarity graphs from
sequence and structure similarity data.
"""

import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data


class ProteinGraph:
    """
    Constructs protein similarity graphs from sequence and structure data.

    Edges are created based on:
    - Sequence similarity (BLAST e-value based)
    - Structure similarity (TM-score based)
    """

    def __init__(
        self,
        protein_ids: List[str],
        seq_similarity_file: Optional[str] = None,
        struct_similarity_file: Optional[str] = None,
        seq_threshold: float = 1e-3,
        struct_threshold: float = 0.5,
    ):
        """
        Args:
            protein_ids: List of protein IDs
            seq_similarity_file: Path to sequence similarity TSV (Source, Target, Score)
            struct_similarity_file: Path to structure similarity TSV
            seq_threshold: E-value threshold for sequence similarity edges
            struct_threshold: TM-score threshold for structure similarity edges
        """
        self.protein_ids = protein_ids
        self.prot_id_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
        self.num_nodes = len(protein_ids)

        self.seq_similarity_file = seq_similarity_file
        self.struct_similarity_file = struct_similarity_file
        self.seq_threshold = seq_threshold
        self.struct_threshold = struct_threshold

        self.edge_index = None
        self.edge_weights = None
        self.seq_sim_matrix = None
        self.struct_sim_matrix = None

    def build_graph(
        self,
        use_sequence: bool = True,
        use_structure: bool = True,
        lambda_seq: float = 0.5,
    ) -> Data:
        """
        Build the protein similarity graph.

        Args:
            use_sequence: Whether to use sequence similarity edges
            use_structure: Whether to use structure similarity edges
            lambda_seq: Weight for combining sequence (lambda) and structure (1-lambda)

        Returns:
            PyG Data object with edge_index, edge_weight, and num_nodes
        """
        edges = []
        edge_weights = []

        if use_sequence and self.seq_similarity_file:
            seq_edges, seq_weights = self._load_sequence_edges()
            edges.extend(seq_edges)
            edge_weights.extend(seq_weights)

        if use_structure and self.struct_similarity_file:
            struct_edges, struct_weights = self._load_structure_edges()
            edges.extend(struct_edges)
            edge_weights.extend(struct_weights)

        if not edges:
            raise ValueError("No edges generated. Provide similarity files.")

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

        data = Data(
            edge_index=edge_index,
            edge_attr=edge_weights,
            num_nodes=self.num_nodes,
        )

        print(f"Graph: {self.num_nodes} nodes, {len(edges)} edges")
        return data

    def _load_sequence_edges(self) -> Tuple[List[List[int]], List[float]]:
        """Load sequence similarity edges from BLAST results."""
        edges = []
        weights = []

        df = pd.read_csv(self.seq_similarity_file, sep="\t")

        for _, row in df.iterrows():
            src_idx = self.prot_id_to_idx.get(row["Source"])
            tgt_idx = self.prot_id_to_idx.get(row["Target"])

            if src_idx is None or tgt_idx is None:
                continue

            score = float(row["Score"]) if "Score" in row.columns else float(row.get("weight", 1.0))

            if "Evalue" in row.columns:
                evalue = float(row["Evalue"])
                if evalue > self.seq_threshold:
                    continue
                # Convert e-value to similarity (lower is better)
                similarity = 1.0 / (1.0 + evalue)
            else:
                similarity = score

            # Add undirected edges
            edges.append([src_idx, tgt_idx])
            weights.append(similarity)
            edges.append([tgt_idx, src_idx])
            weights.append(similarity)

        return edges, weights

    def _load_structure_edges(self) -> Tuple[List[List[int]], List[float]]:
        """Load structure similarity edges from TM-align results."""
        edges = []
        weights = []

        df = pd.read_csv(self.struct_similarity_file, sep="\t")

        for _, row in df.iterrows():
            src_idx = self.prot_id_to_idx.get(row["Source"])
            tgt_idx = self.prot_id_to_idx.get(row["Target"])

            if src_idx is None or tgt_idx is None:
                continue

            tmscore = float(row["TMScore"]) if "TMScore" in row.columns else float(row.get("score", 0.0))

            if tmscore < self.struct_threshold:
                continue

            # Add undirected edges
            edges.append([src_idx, tgt_idx])
            weights.append(tmscore)
            edges.append([tgt_idx, src_idx])
            weights.append(tmscore)

        return edges, weights

    def get_edge_index(self) -> Optional[torch.Tensor]:
        """Get the edge index tensor."""
        return self.edge_index

    def get_edge_weights(self) -> Optional[torch.Tensor]:
        """Get the edge weights tensor."""
        return self.edge_weights


def create_synthetic_graph(
    num_nodes: int,
    avg_degree: int = 20,
    seed: int = 42,
) -> Data:
    """
    Create a synthetic protein graph for testing/demo.

    Args:
        num_nodes: Number of nodes
        avg_degree: Average node degree
        seed: Random seed

    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    edges = []
    edge_weights = []

    for i in range(num_nodes):
        # Connect each node to some random neighbors
        num_neighbors = np.random.poisson(avg_degree)
        num_neighbors = min(num_neighbors, num_nodes - 1)

        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i],
            size=num_neighbors,
            replace=False,
        )

        for j in neighbors:
            weight = np.random.uniform(0.5, 1.0)
            edges.append([i, j])
            edge_weights.append(weight)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)

    return Data(
        edge_index=edge_index,
        edge_attr=edge_weights_tensor,
        num_nodes=num_nodes,
    )
