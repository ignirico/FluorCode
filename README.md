# FluorCode

Predicting fluorescent protein (FP) photophysical properties using position-aware protein language model features. This project accompanies the paper *"Position Matters: Alignment-Anchored PLM Features for Fluorescent Protein Property Prediction"*, targeting the ICML 2025 Workshop on Computational Biology.

## Motivation

Mean-pooled PLM embeddings (e.g., ESM) underperform classical one-hot alignment features for FP property prediction because averaging over ~240 residues dilutes the signal from the ~3-5 residues that actually determine color. FluorCode fixes this by extracting **per-residue PLM embeddings at alignment-defined positions**, preserving the positional specificity that matters.

## Predicted Properties

| Property | Description |
|----------|-------------|
| Excitation max (nm) | Peak excitation wavelength |
| Emission max (nm) | Peak emission wavelength |
| Quantum yield | Fluorescence efficiency |
| Extinction coefficient | Light absorption strength |
| pKa | pH sensitivity |

## Baseline Results (Improved FPredX)

Reimplementation of [FPredX](https://doi.org/10.1002/prot.26372) (Tam & Zhang, *Proteins* 2022) with a larger dataset (1,040 vs. 738 FPs), smarter feature filtering, and Optuna hyperparameter tuning.

**20-fold cross-validation:**

| Target | Pooled MAE | Best Fold MAE | Original FPredX | Improved? |
|--------|:----------:|:-------------:|:----------------:|:---------:|
| Ex max | 14.76 nm | **9.33 nm** | 11.23 nm | Yes |
| Em max | 9.38 nm | **4.48 nm** | 7.72 nm | Yes |
| QY | 0.106 | 0.063 | N/A | -- |
| Ext coeff | 17,837 | 12,833 | N/A | -- |
| pKa | 0.696 | 0.360 | N/A | -- |

## Project Structure

```
FluorCode/
├── data/
│   ├── sequence/              # FPbase sequences, alignments, embeddings
│   │   ├── fp_cleaned.csv             # Cleaned dataset with photophysical labels
│   │   ├── fp_sequences_aligned.fasta # MSA (MAFFT, 3212 columns)
│   │   ├── fp_embeddings.npz          # ESMC-300M embeddings (960-dim)
│   │   └── chromophore_positions.csv  # Chromophore triad positions per FP
│   ├── structure/
│   │   ├── simplefold/        # SimpleFold (100M) predicted PDB structures
│   │   └── structures_overall/# Parsed 3D structural features (.npz)
│   ├── fetch_fpbase.py        # Download FP records from FPbase
│   ├── identify_chromophore.py# Identify chromophore residue positions
│   ├── fold_simplefold.py     # Run SimpleFold structure prediction
│   └── parse_structures.py    # Parse PDB into structural features
├── model/
│   └── Baseline_FPredX/
│       ├── training_and_benchmark.py  # Training, Optuna tuning, 20-fold CV
│       └── benchmark_results/         # Metrics, plots, comparisons
└── reference/                 # Papers and research roadmap
```

## Data Pipeline

1. **Fetch** -- Download FP records from [FPbase](https://www.fpbase.org/) (`fetch_fpbase.py`)
2. **Clean** -- Deduplicate, quality-filter, extract photophysical labels
3. **Align** -- Trim to conserved barrel region, run MAFFT MSA
4. **Embed** -- Generate ESMC-300M mean-pooled embeddings
5. **Structure** -- Predict 3D structures with SimpleFold, parse into features
6. **Chromophore** -- Identify chromophore triad positions per FP (`identify_chromophore.py`)

## Getting Started

```bash
git clone https://github.com/ignirico/FluorCode.git
cd FluorCode
```

### Run the baseline model

```bash
python model/Baseline_FPredX/training_and_benchmark.py
```

### Set up SimpleFold

SimpleFold must be deployed before running structure prediction. Follow the installation instructions at [apple/ml-simplefold](https://github.com/apple/ml-simplefold).

```bash
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
# Follow the repo's setup instructions
```

### Regenerate data from scratch

```bash
python data/fetch_fpbase.py
python data/identify_chromophore.py
python data/fold_simplefold.py
python data/parse_structures.py
```

## Key References

- Tam & Zhang. *FPredX: Predicting fluorescent protein properties using alignment-based features.* Proteins, 2022.
- Lin et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 2023. (ESMFold / SimpleFold)

## License

This project is for research purposes.
