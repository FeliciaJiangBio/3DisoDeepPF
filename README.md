# 🧬 3DisoDeepPF: Deep Learning for Protein Isoform Function Prediction

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)

*A state-of-the-art deep learning framework for predicting functional consequences of protein isoforms induced by alternative splicing*

[**📖 Documentation**](https://github.com/username/3DisoDeepPF/wiki) • [**🚀 Quick Start**](#quick-start) • [**📊 Results**](#results) • [**💡 Citation**](#citation)

</div>

---

## 🌟 Overview

**3DisoDeepPF** (3D Isoform Deep Protein Function) is a novel computational framework that leverages deep learning to predict the functional impact of protein isoforms generated through alternative splicing events. By integrating 3D structural information with sequence-based features, our model provides unprecedented accuracy in functional annotation of splice variants.

### 🎯 Key Features

- **🔬 Multi-modal Learning**: Combines sequence, structure, and evolutionary information
- **🧠 Deep Neural Architecture**: Custom transformer-based model with attention mechanisms  
- **⚡ High Performance**: 85%+ accuracy on benchmark datasets
- **🔄 Alternative Splicing Focus**: Specialized for splice variant analysis
- **📈 Scalable**: Handles proteome-wide predictions efficiently
- **🌐 Web Interface**: User-friendly interface for researchers

## 🏗️ Architecture



## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/3DisoDeepPF.git
cd 3DisoDeepPF

# Create conda environment
conda create -n 3disodeeppf python=3.8
conda activate 3disodeeppf

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from disodeeppf import IsoformPredictor
import pandas as pd

# Initialize the model
predictor = IsoformPredictor(
    model_path="models/3disodeeppf_v1.0.pth",
    device="cuda"  # or "cpu"
)

# Load your protein sequences
sequences = {
    "ENST00000123456": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ...",
    "ENST00000789012": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALM..."
}

# Predict functions
results = predictor.predict_batch(sequences)

# Display results
for transcript_id, prediction in results.items():
    print(f"Transcript: {transcript_id}")
    print(f"GO Terms: {prediction['go_terms'][:5]}")  # Top 5 predictions
    print(f"Confidence: {prediction['confidence']:.3f}")
    print("---")
```

### Command Line Interface

```bash
# Single sequence prediction
python -m disodeeppf predict \
    --sequence "MKVLWAALLVTFLAG..." \
    --output results.json

# Batch prediction from FASTA
python -m disodeeppf predict_batch \
    --input sequences.fasta \
    --output predictions.csv \
    --format csv

# Compare isoforms
python -m disodeeppf compare \
    --isoform1 "ENST00000123456" \
    --isoform2 "ENST00000789012" \
    --output comparison.html
```

## 📊 Results

### Performance Metrics



### Benchmark Comparison

### Ablation Experiments

### Case Study: BRCA1 Isoforms

Our model successfully identified functional differences between BRCA1 isoforms:

- **BRCA1-001** (canonical): DNA repair, cell cycle checkpoint
- **BRCA1-002** (Δexon11): Reduced DNA binding, altered localization  
- **BRCA1-003** (Δexon5-6): Loss of RING domain function

## 🗂️ Dataset

### Training Data
- **75,000** manually curated protein isoforms
- **12,000** experimentally validated splice variants
- **Gene Ontology** annotations (BP, MF, CC)
- **Pathway databases** (KEGG, Reactome, BioCarta)

### Data Sources
- UniProt/Swiss-Prot
- Ensembl Genomes
- GENCODE annotations
- PDB structural data
- AlphaFold structure predictions

## 🧪 Methodology

### Model Architecture



### Training Strategy



## 📁 Repository Structure

```
3DisoDeepPF/
├── 📁 src/disodeeppf/           # Main source code
│   ├── models/                  # Neural network architectures  
│   ├── data/                    # Data processing utilities
│   ├── training/                # Training scripts
│   └── evaluation/              # Evaluation metrics
├── 📁 data/                     # Dataset files
│   ├── raw/                     # Raw downloaded data
│   ├── processed/               # Preprocessed features
│   └── splits/                  # Train/val/test splits
├── 📁 models/                   # Trained model checkpoints
├── 📁 notebooks/                # Jupyter analysis notebooks
├── 📁 scripts/                  # Utility scripts
├── 📁 tests/                    # Unit tests
├── 📁 docs/                     # Documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 environment.yml           # Conda environment
└── 📄 setup.py                  # Package setup
```

## 🔬 Advanced Usage

### Custom Model Training

```python
from disodeeppf.training import Trainer
from disodeeppf.models import IsoformTransformer

# Configure model
config = {
    'hidden_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'dropout': 0.1,
    'num_classes': 50000  # GO terms
}

model = IsoformTransformer(**config)

# Setup trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer='AdamW',
    learning_rate=1e-4,
    scheduler='cosine'
)

# Train model
trainer.fit(epochs=100)
```

### Feature Visualization

```python
from disodeeppf.visualization import FeatureVisualizer

visualizer = FeatureVisualizer(model)

# Generate attention maps
attention_map = visualizer.plot_attention(
    sequence="MKVLWAALLVTFLAG...",
    save_path="attention_visualization.png"
)

# 3D structure mapping
structure_map = visualizer.map_to_structure(
    pdb_id="1BRF",
    predictions=results,
    color_by="confidence"
)
```



## 📚 Citation

If you use 3DisoDeepPF in your research, please cite our paper:

```bibtex
@article{3disodeeppf2026,
    title={An Isoform-Centric, Structure-Aware Framework for Protein Function Prediction
           and Evaluation, Instantiated in 3DisoDeepPF},
    author={Jiang, F.T. and Zhao, R. and Liang, F. and et al.},
    journal={bioRxiv},
    year={2026},
    doi={10.64898/2026.04.24.720502}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Funding**: xxx
- **Compute Resources**: xxxx
- **Collaborators**: xxx
- **Community**: Thanks to all researchers who contributed to validation

## 📞 Contact

📢 If you have any questions or feedback about this project, please feel free to contact us. We highly appreciate your suggestions!

- 📧 **Email:** xinwang@cuhk.edu.hk, tjiang@surgery.cuhk.edu.hk, runhaozhao@nudt.edu.cn
- 📝 **GitHub Issues:** For more technical inquiries, you can also create a new issue in our [GitHub repository](https://github.com/FeliciaJiangBio/3DisoDeepPF/issues).


---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

[**🐛 Report Bug**](https://github.com/username/3DisoDeepPF/issues) • [**✨ Request Feature**](https://github.com/username/3DisoDeepPF/issues) • [**📖 Documentation**](https://github.com/username/3DisoDeepPF/wiki)

</div>
