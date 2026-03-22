# Plan: ICML Workshop Paper — Position-Aware PLM Features for FP Property Prediction

## Context

**Problem**: Our improved FPredX (alignment one-hot + Optuna XGBoost) already beats the 2021 FPredX paper. But for an ICML workshop submission, we need a novel ML contribution — not just "better hyperparameters on a 2021 method."

**Key insight**: Mean-pooled ESMC embeddings (26nm MAE) lose to 2021 one-hot features (15nm MAE) because mean-pooling destroys position-specific residue identity. FP color is determined by ~3-5 key residues — averaging 240 residues dilutes this signal ~50×.

**Fix**: Extract per-residue PLM embeddings at alignment-defined positions ("Alignment-Anchored PLM features" / AA-PLM). This preserves position-specificity while retaining PLM contextual richness.

**Paper title (working)**: *"Position Matters: Alignment-Anchored PLM Features for Fluorescent Protein Property Prediction"*

---

## Implementation Plan

### Phase 1: AA-PLM Feature Extraction

**New script**: `ML_Algorithms/Improved_FPredX/extract_aa_plm.py`

1. Modify `embed_esmc.py` approach: run ESMC-300M but save **per-residue** embeddings (L × 960) instead of mean-pooling
2. Use MAFFT alignment to map alignment columns → sequence positions for each FP
3. For each of the 1271 informative alignment columns (same as one-hot), extract the 960-dim ESMC embedding of the residue at that position (zero vector for gaps)
4. PCA per alignment position: 960 → 16 dims (fit within each CV fold to prevent leakage)
5. Output: `(986, 1271, 16)` → flatten to `(986, 20336)` for XGBoost

**Also extract**: Chromophore-local variant — only positions within ±20 alignment columns of chromophore (~80 columns × 16 = ~1280 features)

**Memory strategy**: Process sequences one-by-one, immediately project to alignment positions, discard full per-residue matrix. Peak memory ~1GB.

### Phase 2: Ablation Study — The Core Table

**Script**: `ML_Algorithms/Improved_FPredX/ablation_pooling.py`

All methods evaluated with same protocol: Optuna-tuned XGBoost, 20-fold random CV, 30 trials.

| Method | Features | Description |
|--------|:--------:|-------------|
| One-Hot (FPredX-style) | 1,271 | Current SOTA baseline |
| ESMC Mean-Pool | 960 | Standard PLM approach |
| ESMC Max-Pool | 960 | Alternative pooling |
| ESMC Attention-Pool (frozen) | 960 | Learned attention, frozen backbone |
| ESMC Chromophore-Window | 960 | Mean-pool only ±20 residues around chromophore |
| **AA-PLM (full)** | ~20K | Per-position PLM features, all 1271 columns, PCA-16 |
| **AA-PLM (chromophore-local)** | ~1,280 | Per-position PLM, ±20 columns around chromophore |
| One-Hot + AA-PLM | ~21K | Combined |

### Phase 3: Fine-tuned ESMC with Chromophore-Aware Attention

**Script**: `ML_Algorithms/Finetune_ESM/finetune_chromophore.py` (extend existing `extract_attention.py`)

Modifications to existing `ESMCMultiTask`:
- **LoRA** (rank=8) on q_proj/v_proj in last 6 ESMC layers (~150K trainable params vs 300M frozen)
- **Chromophore position bias**: learnable scalar added to attention scores at chromophore residue indices
- **5-target multi-task**: extend from 2 heads to 5 with masked loss
- **Multi-head attention pooling**: 4 heads × 240-dim → 960-dim pooled

Training: AdamW lr=5e-4 (LoRA) / 1e-4 (heads), CosineAnnealing 100 epochs, patience=20, batch=4 with grad accumulation=4. Evaluated with 20-fold CV.

### Phase 4: Figures and Visualization

**Script**: `ML_Algorithms/Improved_FPredX/paper_figures.py`

1. **The pooling problem**: UMAP of per-residue ESMC embeddings for EGFP, colored by distance to chromophore — show chromophore residues aren't separated, mean-pooling dilutes them
2. **Attention maps**: Fine-tuned model attention weights along sequence for EGFP, mCherry, mTurquoise2 — show model discovers chromophore positions
3. **Scatter plots**: Predicted vs measured for One-Hot, Mean-Pool, AA-PLM across all targets
4. **Feature importance**: XGBoost feature importance for AA-PLM, mapped back to GFP structure positions — should cluster around chromophore

---

## Paper Structure (4-6 pages)

1. **Introduction** (0.75p): FPs are uniquely position-sensitive. 2021 one-hot features beat modern PLMs. Why?
2. **Background** (0.5p): FPredX, ESM/ESMC, protein property prediction
3. **Why Mean-Pooling Fails** (0.75p): Information-theoretic argument — chromophore signal is ~1/240 of mean-pooled vector
4. **AA-PLM Method** (0.75p): Alignment-anchored per-residue PLM features
5. **Experiments** (1.5p): Main ablation table, FPredX comparison, attention visualization, feature importance
6. **Discussion** (0.5p): When does mean-pooling fail? Implications for other position-sensitive protein families

---

## Critical Files

| File | Role |
|------|------|
| `embed_esmc.py` | Adapt for per-residue extraction |
| `ML_Algorithms/Improved_FPredX/benchmark_vs_fprex.py` | One-hot builder + 20-fold CV protocol to reuse |
| `ML_Algorithms/Finetune_ESM/extract_attention.py` | Existing attention + fine-tuning code to extend |
| `data/fp_sequences_aligned.fasta` | 3212-column alignment for position anchoring |
| `data/chromophore_positions.csv` | Chromophore positions for local features + attention bias |

## Execution Order

1. `extract_aa_plm.py` — extract alignment-anchored PLM features (~30 min compute)
2. `ablation_pooling.py` — run all pooling variants through 20-fold CV (~2-3 hours)
3. `finetune_chromophore.py` — LoRA fine-tuning with 20-fold CV (~4-6 hours on MPS)
4. `paper_figures.py` — generate all figures
5. Write paper

## Verification

- AA-PLM pooled MAE should beat mean-pooled ESMC by >30% on ex_max/em_max
- AA-PLM best fold should approach or beat one-hot (9.33nm ex_max, 4.48nm em_max)
- Attention maps should show peaks at known chromophore positions without explicit supervision
- Feature importance should concentrate on alignment positions near chromophore
