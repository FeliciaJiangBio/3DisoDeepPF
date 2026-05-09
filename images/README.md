# 📁 Images Directory

This directory stores paper figures for the README.md.

## How to Add Figures

### Step 1: Extract Figures from PDF

1. Open the paper PDF (`3DisoDeepPF.pdf`) in a PDF viewer
2. Screenshot or export the following figures:
   - **Fig. 1a**: Main framework overview
   - **Fig. 1c**: Fmax comparison dot plots
   - **Fig. 1d**: AUPR bar plots
   - **Fig. 2a**: 2D embedding visualization
   - **Fig. 2b/c**: Isoform evaluation results

### Step 2: Save Images

Save screenshots as PNG files with these names:

```
images/
├── README.md                      # This file
├── fig1a_framework_overview.png   # Main framework diagram (Fig. 1a)
├── fig1c_fmax_comparison.png     # Fmax dot plots (Fig. 1c)
├── fig1d_aupr_comparison.png     # AUPR bars (Fig. 1d)
├── fig2a_embedding.png           # 2D embedding (Fig. 2a)
├── fig2b_isoform_aupr.png       # Isoform AUPR (Fig. 2b)
└── fig2c_isoform_fmax.png       # Isoform Fmax (Fig. 2c)
```

### Step 3: Recommended Image Size

- Width: 800-1200 pixels
- Format: PNG or JPG
- File size: < 500KB per image (optional)

### Step 4: Update README

Once images are added, update README.md to use markdown image syntax:

```markdown
![Figure 1a: Framework Overview](images/fig1a_framework_overview.png)
```

## Figure Sources

All figures are from the 3DisoDeepPF paper:
> Jiang, F.T. et al. (2026). An Isoform-Centric, Structure-Aware Framework for Protein Function Prediction. bioRxiv.
