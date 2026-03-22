# Improved FPredX — Fluorescent Protein Property Prediction

## Overview

This model improves upon **FPredX** (Tam & Zhang, *Proteins* 2022, [DOI:10.1002/prot.26270](https://doi.org/10.1002/prot.26270)) for predicting fluorescent protein photophysical properties from sequence.

Both FPredX and this model use the same core approach: **alignment-based one-hot encoding + XGBoost**. Our improvements come from a larger dataset, smarter feature filtering, and automated hyperparameter tuning.

---

## Results Summary (20-fold random CV)

| Target | Our Pooled MAE | Our Best Fold | FPredX Best Fold | Beat? | R² |
|--------|:--------------:|:------------:|:----------------:|:-----:|:--:|
| ex_max | 14.76 nm | **9.33 nm** | 11.23 nm | **YES** | 0.807 |
| em_max | 9.38 nm | **4.48 nm** | 7.72 nm | **YES** | 0.886 |
| QY | 0.106 | 0.063 | N/A | — | 0.639 |
| ext_coeff | 17,837 | 12,833 | N/A | — | 0.513 |
| pKa | 0.696 | 0.360 | N/A | — | 0.276 |

> **Note**: FPredX only reports their best single fold MAE across 20 folds, not pooled MAE. Our best fold beats theirs on both spectral targets.

---

## What Changed vs FPredX

### 1. Larger Dataset

| | FPredX (March 2021) | Ours (March 2026) |
|--|:---:|:---:|
| FPbase records | 738 | 1,040 |
| After trimAl | 672 | 986 |
| ex_max labels | 544 | 872 |
| em_max labels | 542 | 833 |

FPbase has grown significantly since 2021. More training data = better generalization.

### 2. Smarter Feature Engineering

| | FPredX | Ours |
|--|---|---|
| Alignment length | 401 positions | 3,212 positions |
| One-hot features | 3,362 | **1,271** |
| Feature filtering | All position-residue pairs | Only pairs present in ≥2% of sequences, zero-variance removed |

FPredX kept all position-residue pairs including very rare ones. We filter to only informative features (present in ≥2% of FPs), which reduces noise and overfitting. Fewer features, better performance.

### 3. Optuna Hyperparameter Tuning

FPredX used default or manually tuned XGBoost parameters. We use **Optuna** (30 trials, 3-fold CV) to automatically find optimal hyperparameters per target.

**Tuned hyperparameters:**

| Parameter | FPredX (likely defaults) | Ours (ex_max) | Ours (em_max) |
|-----------|:---:|:---:|:---:|
| n_estimators | ~100-500 | 1,109 | 1,193 |
| max_depth | 6 | 8 | 8 |
| learning_rate | 0.1-0.3 | 0.013 | 0.011 |
| subsample | 1.0 | 0.82 | 0.90 |
| colsample_bytree | 1.0 | 0.69 | 0.46 |
| reg_alpha (L1) | 0 | 2.48 | 0.014 |
| reg_lambda (L2) | 1 | 2.69 | 0.57 |
| min_child_weight | 1 | 1 | 1 |

Key pattern: Optuna found that **more trees + lower learning rate + stronger regularization** works best. This is a classic boosting finding — many weak learners with slow learning and regularization generalize better than fewer aggressive trees.

**All Optuna-tuned parameters by target:**

| Parameter | ex_max | em_max | QY | ext_coeff | pKa |
|-----------|:------:|:------:|:--:|:---------:|:---:|
| n_estimators | 1,109 | 1,193 | 381 | 1,013 | 432 |
| max_depth | 8 | 8 | 7 | 8 | 7 |
| learning_rate | 0.013 | 0.011 | 0.099 | 0.014 | 0.018 |
| subsample | 0.82 | 0.90 | 0.94 | 0.97 | 0.77 |
| colsample_bytree | 0.69 | 0.46 | 0.40 | 0.31 | 0.42 |
| reg_alpha | 2.48 | 0.014 | 0.084 | 1.94 | 0.53 |
| reg_lambda | 2.69 | 0.57 | 0.53 | 4.68 | 4.81 |

### 4. Additional Target Properties

FPredX predicted: excitation max, emission max, brightness, oligomeric state.

We predict: **excitation max, emission max, quantum yield (QY), extinction coefficient, pKa** — three continuous properties not covered by FPredX.

### 5. What We Tried That Didn't Help

- **ESMC-300M embeddings** (960-dim protein language model): pooled MAE = 26.39 nm for ex_max — much worse than one-hot (14.76 nm). The alignment explicitly encodes position-specific residue identity, which is what determines spectral properties.
- **Combining OneHot + ESMC**: adding ESMC features (even PCA-compressed to 64 dims) to one-hot features **increased** MAE. The extra dimensions add noise.
- **GVP-GNN structural features**: with only ~300 structures (many predicted by SimpleFold), structural features hurt rather than helped.

---

## Evaluation Protocol

Identical to FPredX for fair comparison:

1. **20-fold cross-validation**: randomly split data into 20 folds, train on 19, test on 1, rotate
2. **Metrics**: MAE (primary), RMSE, Pearson r, R²
3. **Per-fold reporting**: report pooled MAE across all folds and best single fold MAE

---

## How to Run

```bash
cd FluorCode
python3 ML_Algorithms/Improved_FPredX/benchmark_vs_fprex.py
```

**Runtime**: ~5 minutes (30 Optuna trials × 5 targets)

**Dependencies**: numpy, pandas, biopython, scikit-learn, xgboost, optuna

**Output**: `ML_Algorithms/Improved_FPredX/benchmark_results/`

---

## Reference

- FPredX paper: Tam & Zhang, "FPredX: Interpretable models for the prediction of spectral maxima, brightness, and oligomeric states of fluorescent proteins", *Proteins* 2022;90:732-746, [DOI:10.1002/prot.26270](https://doi.org/10.1002/prot.26270)
- FPredX code: [github.com/johnnytam100/FPredX](https://github.com/johnnytam100/FPredX)
