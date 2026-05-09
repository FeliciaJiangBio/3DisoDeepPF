# 🧬 3DisoDeepPF

### An Isoform-Centric, Structure-Aware Framework for Protein Function Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898/2026.04.24.720502-orange.svg)](https://doi.org/10.64898/2026.04.24.720502)

> *"A novel computational framework that leverages deep learning to predict the functional impact of protein isoforms generated through alternative splicing events."*

---

## 📖 Abstract

Protein function prediction (PFP) is essential for mechanistic insight, disease biology, and therapeutic development. Most existing approaches assign function using a single reference protein form per gene, overlooking functionally important variations across proteoforms.

**3DisoDeepPF** (3D Isoform Deep Protein Function) addresses this gap by:
- Resolving protein function at the **isoform level** explicitly
- Integrating **multi-modal representations** (sequence, structure, motif, annotation)
- Leveraging **graph neural networks** over protein similarity networks
- Jointly predicting **GO terms and Pfam domains** with cross-modal enhancement

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Isoform-Centric Design** | Explicitly resolves protein function at the isoform resolution |
| 🔬 **Multi-Modal Fusion** | Combines ESM embeddings, structural features, motifs, and annotations |
| 🕸️ **Graph Neural Networks** | GCN/GAT architecture over sequence-structure similarity graphs |
| ⚖️ **Adaptive Edge Weighting** | Learned combination of sequence (BLAST) and structure (TM-align/Foldseek) similarity |
| 🔄 **Cross-Modal Learning** | Joint GO ↔ Pfam prediction with mutual enhancement |
| 🎯 **Evidence Tracing** | Decomposes predictions into supporting topological and modal evidence |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     3DisoDeepPF Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Sequence   │    │  Structure   │    │   Motif /    │      │
│  │   (ESM)     │    │  (TM-align) │    │   Pfam      │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │           Gated Multi-Modal Feature Fusion             │     │
│  │           (α_seq · ESM ⊕ α_str · Struct ⊕ α_motif)   │     │
│  └─────────────────────────┬───────────────────────────┘     │
│                            │                                   │
│                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Graph Convolution Layers (GCN/GAT)            │   │
│  │     Adaptive Edge Weighting: λ·Seq + (1-λ)·Struct     │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                   │
│                            ▼                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────┐ │
│  │   GO-MF    │  │   GO-BP    │  │   GO-CC    │  │  Pfam   │ │
│  │  Predictor │  │  Predictor │  │  Predictor │  │Predictor│ │
│  └────────────┘  └────────────┘  └────────────┘  └─────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance

Evaluated on **CAFA-aligned benchmarks** and **breast cancer isoform atlas**:

| Task | Metric | 3DisoDeepPF | Best Baseline |
|------|--------|:------------:|:-------------:|
| **GO-MF** | Fmax | ✅ Highest | DeepGOPlus |
| **GO-BP** | Fmax | ✅ Highest | DeepFRI |
| **GO-CC** | Fmax | ✅ Highest | FunFams |
| **Pfam** | Fmax | ✅ Highest | BLAST |
| **All Tasks** | AUPR | ✅ Best | Competitive |

> See our [bioRxiv paper](https://doi.org/10.64898/2026.04.24.720502) for detailed benchmark results.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FeliciaJiangBio/3DisoDeepPF.git
cd 3DisoDeepPF

# Create conda environment
conda env create -f environment.yml
conda activate 3disodeeppf

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Run Demo

```bash
# Navigate to examples and run the demo
cd examples
python demo.py --num_proteins 500 --epochs 50
```

### Basic Usage

```python
from 3disodeeppf.models import CrossModalGNN
from 3disodeeppf.training import Trainer

# Initialize model
model = CrossModalGNN(
    num_nodes=10000,
    num_labels=30000,
    embedding_dim=128,
    esm_dim=512,
    motif_dim=128,
    hidden_dim=256,
    num_layers=2,
)

# Train
trainer = Trainer(
    model=model,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    labels=labels,
    learning_rate=0.01,
)

trainer.train(num_epochs=100)

# Predict
predictions = trainer.predict(node_indices)
```

---

## 📁 Repository Structure

```
3DisoDeepPF/
├── src/3disodeeppf/              # Main source code
│   ├── models/
│   │   ├── gnn.py               # CrossModalGNN architecture
│   │   ├── esm_encoder.py       # ESM feature extraction
│   │   └── multi_modal_fusion.py # Gated fusion module
│   ├── data/
│   │   ├── dataset.py            # Dataset classes
│   │   └── protein_graph.py     # Graph construction
│   ├── training/
│   │   └── trainer.py            # Training loop
│   └── evaluation/
│       └── metrics.py            # Fmax, AUPR, per-label metrics
│
├── examples/
│   ├── demo.py                   # Demo with synthetic data
│   └── predict.py                # Prediction script
│
├── scripts/
│   └── train.py                  # Training script
│
├── tests/
│   └── test_models.py            # Unit tests
│
├── requirements.txt
├── environment.yml
├── setup.py
└── README.md
```

---

## 🔬 Methodology Highlights

### Multi-Modal Graph Construction

We construct protein similarity graphs integrating:
- **Sequence similarity**: BLASTP e-value < 1e-3
- **Structure similarity**: TM-score via Foldseek/TM-align

Combined using learned mixing parameter λ:

```
A_ij = λ · A_seq + (1 - λ) · A_struct
```

### Gated Multi-Modal Fusion

Feature contributions are dynamically weighted:

```
h_final = Σ_k α_k · σ(W_k · h_k)
```

where α_k are softmax-normalized gating weights.

### Focal Loss for Class Imbalance

Address label imbalance with focal loss:

```
L = -Σ_c w_c · [(1-p_c)^γ · log(p_c)]
```

### Evidence Tracing

Decompose predictions into:
- **Topological support**: Contributing graph neighbors
- **Modal evidence**: Gating weights per modality

---

## 📚 Datasets

| Dataset | Proteins | Description |
|---------|----------|-------------|
| **CAFA-Aligned Benchmark** | 76,804 | Swiss-Prot + PDB + AlphaFoldDB |
| **3DisoGalaxy Atlas** | 46,411 | Breast cancer isoforms |

> Visit our web portals: [3DisoDeepPF](http://3disodeeppf.com/) | [3DisoGalaxy](http://3disogalaxy.com/)

---

## 📄 Citation

If 3DisoDeepPF contributes to your research, please cite:

```bibtex
@article{3disodeeppf2026,
    title={An Isoform-Centric, Structure-Aware Framework for Protein Function
           Prediction and Evaluation, Instantiated in 3DisoDeepPF},
    author={Jiang, F.T. and Zhao, R. and Liang, F. and Cui, T. and
            Zhang, Y. and Zhao, X. and Xu, M. and Shuai, Y. and
            Luo, T. and Wang, X. and Tang, J. and Yao, H. and Xu, C.
            and Wang, Z. and Zeng, W. and Xu, J. and Tang, Z. and
            Zhang, W. and Heng, P.A. and Li, Y. and Wang, X.},
    journal={bioRxiv},
    year={2026},
    doi={10.64898/2026.04.24.720502}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This work was supported by the Faculty of Medicine at The Chinese University of Hong Kong and collaborative contributions from Peking University and the National University of Defense Technology.

---

## 📞 Contact

For questions, collaborations, or feedback:

| Person | Email | Role |
|--------|-------|------|
| **Felicia T. Jiang** | tjiang@surgery.cuhk.edu.hk | Lead Author |
| **Runhao Zhao** | runhaozhao@nudt.edu.cn | Co-Author |
| **Xin Wang** | xinwang@cuhk.edu.hk | Corresponding Author |

🐛 [Report Issues](https://github.com/FeliciaJiangBio/3DisoDeepPF/issues) • ✨ [Request Features](https://github.com/FeliciaJiangBio/3DisoDeepPF/issues)

---

<div align="center">

**⭐ Star us on GitHub if 3DisoDeepPF is useful to your research!**

</div>
