"""
3DisoDeepPF: An Isoform-Centric, Structure-Aware Framework for Protein Function Prediction

This package provides a deep learning framework for predicting protein function
at the isoform resolution, combining graph neural networks with multi-modal
representations.
"""

__version__ = "1.0.0"
__author__ = "Felicia T. Jiang et al."

from .models import GNN, ESMEncoder, MultiModalFusion
from .data import ProteinDataset, ProteinGraph
from .training import Trainer
from .evaluation import evaluate

__all__ = [
    "GNN",
    "ESMEncoder",
    "MultiModalFusion",
    "ProteinDataset",
    "ProteinGraph",
    "Trainer",
    "evaluate",
]
