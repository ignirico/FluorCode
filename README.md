# FluorCode: Predicting Fluorescent Protein Properties with LoRA-Fine-Tuned Protein Language Models

Code and data for the ICML 2026 AI4Science Workshop paper.

## Overview

FluorCode predicts six photophysical properties of fluorescent proteins (FPs) from amino acid sequence:

| Property | Unit | Description |
|----------|------|-------------|
| ex_max | nm | Excitation maximum wavelength |
| em_max | nm | Emission maximum wavelength |
| qy | 0-1 | Quantum yield |
| ext_coeff | M-1cm-1 | Molar extinction coefficient |
| pka | - | Acid dissociation constant |
| brightness | % | Relative brightness (ext_coeff x qy) |

We compare three approaches:
1. **FPredX (baseline)** - XGBoost on MSA one-hot encoding
2. **LoRA-ESM2 + XGBoost** - LoRA-fine-tuned ESM2-650M embeddings with XGBoost
3. **LoRA-ESM2 + MLP** - LoRA-fine-tuned ESM2-650M embeddings with a 2-layer MLP

## Repository Structure

```
data/
  fetch_fpbase.py           # Download raw data from FPbase API
  identify_chromophore.py   # Identify chromophore tripeptide positions
  fold_simplefold.py        # Fold sequences with SimpleFold
  parse_structures.py       # Parse PDB structures for feature extraction
  graft_chromophore.py      # Graft chromophore into predicted structures
  sequence/                 # Curated sequence data and metadata

model/
  Baseline_FPredX/          # FPredX baseline (one-hot + XGBoost)
  LoRA_ESM2/                # LoRA fine-tuning notebook + training results
  LoRA_ESM2_Structure/      # Structural feature ablation (pocket3d)

benchmark/
  BENCHMARK_REPORT.md       # Full benchmark methodology and results
  compare_models.py         # Head-to-head model comparison script
  clustered/                # MMseqs2-clustered cross-validation results

inference/                  # Standalone prediction from sequence
  model.py                  # Model architecture
  predict.py                # CLI + Python API for predictions

figures/                    # Paper figure generation scripts + outputs
data_visual/                # Exploratory data visualizations
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.9. ESM2 weights (~2.5 GB) are downloaded automatically on first run.

## Quick Start

### Predict properties for new sequences

```bash
cd inference
python predict.py \
    --sequence MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK \
    --checkpoint ../model/LoRA_ESM2/checkpoints/fold_0/best.pt
```

### Reproduce paper figures

```bash
# Figure 2: Dataset statistics
python figures/plot_fig2_data.py

# Figure 3: Random CV comparison
# Figure 4: Clustered CV comparison
python figures/plot_comparison.py

# Supplementary scatter plot
python figures/plot_scatter_em.py
```

### Reproduce benchmarks

The benchmark scripts require LoRA embeddings (`lora_embeddings_all_folds.npz`, ~84 MB).
Download from HuggingFace and place in `model/LoRA_ESM2/`:

```bash
# After downloading embeddings:
python figures/run_mlp_benchmark.py      # MLP benchmark (all targets, all schemes)
python figures/run_xgb_benchmark.py      # XGBoost benchmark (qy, ext_coeff, pka)
```

Pre-computed results are included in `figures/mlp_benchmark_results.csv` and `figures/xgb_extra_results.csv`.

## Model Checkpoints

This repository includes a single fold checkpoint (`model/LoRA_ESM2/checkpoints/fold_0/best.pt`, ~32 MB).
All 20 fold checkpoints for ensemble prediction are available on HuggingFace:

> Coming soon

Each checkpoint contains LoRA adapter weights, attention pooling parameters, prediction head weights, and target normalization statistics.

## Key Results

### Random Cross-Validation (Pearson R)

| Target | FPredX | LoRA+XGBoost | LoRA+MLP |
|--------|--------|--------------|----------|
| ex_max | 0.95 | 0.97 | 0.97 |
| em_max | 0.95 | 0.97 | 0.97 |
| qy | 0.89 | 0.96 | 0.96 |
| ext_coeff | 0.94 | 0.97 | 0.97 |
| pka | 0.72 | 0.90 | 0.91 |
| brightness | 0.94 | 0.96 | 0.96 |

### Clustered CV (50% Sequence Identity)

Under stringent clustered cross-validation, FPredX collapses (R < 0.2 for QY/ext_coeff/pKa) while LoRA-based models maintain R > 0.87 across all targets.

## Data Pipeline

The curated dataset is included in `data/sequence/`. To rebuild from scratch:

```bash
python data/fetch_fpbase.py          # Download raw data from FPbase API
python data/identify_chromophore.py  # Identify chromophore positions
```

## Requirements

See `requirements.txt`. Core dependencies:

```
torch>=2.0, fair-esm, numpy, pandas, scikit-learn, scipy, xgboost, matplotlib, biopython, optuna
```

## Citation

```
@inproceedings{fluorcode2026,
  title={FluorCode: Predicting Fluorescent Protein Properties with LoRA-Fine-Tuned Protein Language Models},
  author={Sou, Rico Chi Kit and Ziajowska, Alicja},
  booktitle={ICML 2026 Workshop on AI for Science},
  year={2026}
}
```

## License

MIT
