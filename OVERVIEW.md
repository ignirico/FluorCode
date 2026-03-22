# FluorCode_colab — Project Overview

This project is a **fluorescent protein (FP) property prediction** research codebase, working toward an ICML 2025 workshop paper on position-aware protein language model (PLM) features. The core goal is to predict photophysical properties (excitation/emission maxima, quantum yield, extinction coefficient, pKa) from FP sequences.

---

## Directory Structure

```
FluorCode_colab/
├── data/
│   ├── sequence/
│   └── structure/
│       ├── simplefold/
│       └── structures_overall/
├── model/
│   └── Baseline_FPredX/
│       └── benchmark_results/
└── reference/
    └── ICML_2025_workshop/
```

---

## `data/`

All raw and processed data for training and evaluation.

### `data/sequence/`

Sequence-level data and embeddings fetched from [FPbase](https://www.fpbase.org/), the fluorescent protein database.

| File | Description |
|------|-------------|
| `fpbase_raw.json` | Raw full export from FPbase API (all FP records with all fields) |
| `fpbase_basic_raw.json` | Lightweight FPbase export (basic fields only) |
| `fp_sequences_raw.fasta` | Raw sequences before any trimming, as fetched from FPbase |
| `fp_sequences_trimmed.fasta` | Sequences after trimming to the conserved FP barrel region |
| `fp_sequences_aligned.fasta` | Multiple sequence alignment (MAFFT) — 3,212-column alignment used for one-hot feature encoding |
| `fp_cleaned.csv` | Cleaned dataset: one row per FP, photophysical labels (ex_max, em_max, QY, ext_coeff, pKa), after deduplication and quality filtering |
| `chromophore_positions.csv` | Per-FP alignment column indices of the chromophore triad — used for position-aware features and attention bias |
| `fp_embeddings.npz` | Mean-pooled ESMC-300M embeddings (960-dim per FP) |
| `fp_embeddings_meta.csv` | Metadata (FP name, FPbase slug) corresponding to rows in `fp_embeddings.npz` |
| `pipeline.log` | Log from the full data pipeline run |
| `embed.log` | Log from the ESMC embedding step |

**Scripts that produced this data** (stored in the repo root, not in `data/`):
- `fetch_fpbase.py` — downloads FP records from FPbase
- `identify_chromophore.py` — identifies chromophore residue positions per FP
- `fold_simplefold.py` — runs SimpleFold structure prediction

### `data/structure/`

Predicted and curated 3D structures for fluorescent proteins.

#### `data/structure/simplefold/`

SimpleFold (100M parameter ESMFold-style model) structure predictions.

| Subfolder/File | Description |
|----------------|-------------|
| `simplefold_fastas/` | Per-FP FASTA files fed into SimpleFold |
| `predictions_simplefold_100M/` | Per-FP `.pdb` or `.npz` structure predictions from the 100M model |

#### `data/structure/structures_overall/`

Contains `.npz` files — one per fluorescent protein — storing parsed 3D structural features (coordinates, backbone angles, etc.). Named by FPbase slug (e.g., `egfp.npz`, `mcherry.npz`). ~600+ structures covering known FPs and engineered variants.

---

## `model/`

Trained models and training scripts.

### `model/Baseline_FPredX/`

An improved reimplementation of **FPredX** (Tam & Zhang, *Proteins* 2022), which predicts FP photophysical properties using alignment-based one-hot encoding + XGBoost.

**Key improvements over the original FPredX:**
- Larger dataset (1,040 vs. 738 FPbase records)
- Smarter feature filtering (≥2% frequency threshold → 1,271 features vs. 3,362)
- Optuna-based hyperparameter tuning (30 trials, 3-fold inner CV)
- 5 predicted targets instead of 2 (adds QY, extinction coefficient, pKa)

**Benchmark results (20-fold random CV):**

| Target | Pooled MAE | Best Fold MAE | FPredX Best Fold | Beat? |
|--------|:----------:|:-------------:|:----------------:|:-----:|
| ex_max | 14.76 nm | **9.33 nm** | 11.23 nm | YES |
| em_max | 9.38 nm | **4.48 nm** | 7.72 nm | YES |
| QY | 0.106 | 0.063 | N/A | — |
| ext_coeff | 17,837 | 12,833 | N/A | — |
| pKa | 0.696 | 0.360 | N/A | — |

| File | Description |
|------|-------------|
| `training_and_benchmark.py` | Main training script: builds one-hot features, runs Optuna tuning, 20-fold CV, outputs results |
| `README.md` | Detailed model documentation, hyperparameters, and what was tried |
| `benchmark_results/` | Output directory for all benchmark results |

#### `model/Baseline_FPredX/benchmark_results/`

| File | Description |
|------|-------------|
| `all_results.json` | Full per-fold metrics for all 5 targets (MAE, RMSE, Pearson r, R²) |
| `summary.csv` | Pooled and best-fold summary statistics across all targets |
| `MAE_comparison_vs_fprex.png` | Bar chart comparing our MAE vs. FPredX on ex_max and em_max |
| `scatter_comparison_vs_fprex.png` | Scatter plots (predicted vs. measured) for all targets |
| `my_result_only.png` | Our results only (no FPredX comparison overlay) |
| `others/` | Reference result files from the original FPredX paper for comparison |

---

## `reference/`

Literature and planning documents.

### `reference/` (root-level files)

| File | Description |
|------|-------------|
| `FPredX_ourbaseline.pdf` | The FPredX paper (Tam & Zhang 2022) — the baseline this project improves upon |
| `SimpleFold_paper.pdf` | SimpleFold paper — the structure predictor used to generate structures in `data/structure/` |
| `road_to_ICML.md` | Research roadmap and implementation plan for the ICML 2025 workshop submission. Describes the proposed *Alignment-Anchored PLM (AA-PLM)* method, ablation study design, fine-tuning strategy (LoRA on ESMC), and paper outline |

### `reference/ICML_2025_workshop/`

Papers collected from the ICML 2025 workshop on protein ML, used for related work and situating this project's contributions.

| File | Description |
|------|-------------|
| `32_Multimodal_Modeling_of_CRIS.pdf` | Workshop paper on multimodal modeling (likely CRISPR/protein related) |
| `48_NextGenPLM_A_Novel_Structur.pdf` | Workshop paper on next-generation protein language models with structural features |
| `55_Advancing_Knotted_Protein_D.pdf` | Workshop paper on knotted protein design |

---

## Research Direction Summary

The project is building toward the paper *"Position Matters: Alignment-Anchored PLM Features for Fluorescent Protein Property Prediction"*. The core finding motivating the work: mean-pooled ESM embeddings (26 nm MAE) lose to 2021-era one-hot features (15 nm MAE) because mean-pooling destroys position-specific residue identity. FP color is determined by ~3–5 key residues — averaging 240 residues dilutes this signal ~50×. The proposed fix extracts per-residue PLM embeddings at alignment-defined positions, preserving positional specificity.
