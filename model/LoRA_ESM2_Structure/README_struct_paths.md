# LoRA_ESM2_Structure — design decision log

**Last update:** 2026-04-16
**Owner:** ekko
**Active path:** Pocket-3D (chromophore-HETATM-anchored 3D-spatial features, ~80 dims)

This file is the *why* behind the structural-feature directory. Code lives next to it; this log explains why **this particular** structural block was authored when diagnostic and [clustered benchmark](../../benchmark/BENCHMARK_REPORT.md) both already concluded "structural features hurt — ship `B_lora` alone."

---

## Files in this directory

| File | Role | Status |
|---|---|---|
| [pocket3d_atom_tables.py](pocket3d_atom_tables.py) | Pure-data atom/residue dictionaries | Active |
| [build_pocket3d_features.py](build_pocket3d_features.py) | Chromophore-anchored 3D-spatial extractor (~80 dims) | Active |
| [build_pocket3d_features.ipynb](build_pocket3d_features.ipynb) | Colab CPU driver, ~3 min for 913 PDBs | Active |
| [validate_pocket3d.ipynb](validate_pocket3d.ipynb) | Sanity + mini-XGBoost gate, ~20 min on H100 | Active |
| pocket3d_features.npz | Pre-computed features for all 913 structures | Active |

---

## The three structural-integration paths considered

### Path 1 — Pocket-3D (chosen)

Anchor at chromophore HETATM heavy atoms (phenol-OH, imidazolinone centroid, phenol-ring centroid). Compute 3D-spatial shells (3.5 / 5 / 8 Å), τ/φ dihedrals, ring planarity RMSDs, named-atom nearest distances (E222, H148, T203 proxies), signed electrostatic proxy `Σ qᵢ/dᵢ²`. **~80 dims, four blocks (A+B+C+D).**

**Why chosen.** Every prior structural design (diagnostic Tier 1) was anchored at backbone CA and *blind to the HETATM ligand*. ESM-IF1 explicitly drops HETATM during parsing; the hand-crafted block computed shells at triad-CA, not at the phenol ring atoms where photophysics lives. Pocket-3D extracts ligand chemistry that **no sequence model and no prior structural block ever saw**. If it still fails, we ship the principled negative result.

### Path 2 — GearNet / ProteinMPNN pocket GNN (rejected)

Train or fine-tune a structural GNN (GearNet, ProteinMPNN, GVP-GNN) on the chromophore pocket, pool the per-residue node embeddings inside a 10 Å radius, project to ≤256 dims.

**Why rejected.**
- ESM-IF1 is itself a graph model that already learned protein-structure embeddings on millions of structures. The diagnostic (Tier 1b PCA sweep) showed *no* compressed version of ESM-IF1 helps. A second GNN trained on 913 PDBs is unlikely to extract signal that a 12M-param pre-trained one missed.
- Diagnostic analysis explicitly advises against the "swap the structural featurizer" strategy.
- Same gain-importance trap risk (Tier 2a): a dense ≥256-dim block will claim XGBoost gain regardless of predictive value.
- Fine-tuning compute is non-trivial (~6-12 h on H100) and eats the 5-day budget.

### Path 3 — APBS / PDB2PQR electrostatic surface (rejected)

Compute the Poisson-Boltzmann electrostatic potential map around the chromophore, project onto a uniform 3D grid centered at phenol-OH, downsample to ≤64 features.

**Why rejected.**
- The CRO chromophore (and tripeptide variants TYG/SYG/etc.) **lacks standard AMBER force-field parameters**. PDB2PQR will refuse the HETATM or assign default zero charges, defeating the purpose.
- Custom parameter generation (Antechamber + GAFF) is multi-day work with debug risk.
- The signed `Σ qᵢ/dᵢ²` proxy in Block B+C captures the same physics with two orders of magnitude less compute.

---

## Fallback ladder if Pocket-3D fails the mini-XGBoost gate

The validation notebook ([validate_pocket3d.ipynb](validate_pocket3d.ipynb), Cell 10) runs a 5-fold random-KFold mini-benchmark on `B_lora` vs `B_lora+pocket3d` on `ex_max`. Decision rule:

| `mae(B_lora+pocket3d) − mae(B_lora)` | Verdict | Action |
|---|---|---|
| ≤ −0.3 nm | STRONG POSITIVE | Proceed to full clustered benchmark immediately |
| < 0 nm | WEAK POSITIVE | Proceed; expect modest clustered-CV gain |
| < +0.3 nm | NEUTRAL | Proceed but lower expectations; test blue-cliff secondary win |
| ≥ +0.3 nm | NEGATIVE | **Abort full benchmark.** Enter ladder below |

**Ladder (in order, stop at first pass):**

1. **Drop block D (barrel architecture, 10 dims).** D is the most likely block to leak family/size signal across clusters because chromophore-protein-centroid distance and CA covariance eigenvalues correlate with overall fold size. Re-run mini-gate with A+B+C only (~70 dims).
2. **Keep only blocks A+B (~50 dims).** Chromophore chemistry + phenol-OH environment only — the minimal-hypothesis version. This is the version most directly tied to the Tier 2b blue-cliff finding.
3. **Regularize harder.** Re-run mini-gate with `reg_alpha` × 2 and `reg_lambda` × 2 on the full 80-dim block. XGBoost over-splits on dense blocks (Tier 2a finding).
4. **Principled negative result.** If steps 1–3 all fail, the paper structural-features section becomes: *"three increasingly physics-aware structural featurizations (ESM-IF1, hand-crafted CA, chromophore-HETATM-anchored 3D) all fail to improve over LoRA-ESM2 on 986-sample clustered CV — with mechanistic explanation via the gain-importance trap and PCA sweep."* Still a publishable contribution.

**Secondary win condition (independent of overall MAE).** Diagnostic Tier 2b flagged a blue-cohort cliff: 41 blue FPs (em_max < 470 nm) carry +11 to +14 nm bias under `B_lora` because BFP/Sapphire/Sirius differ from green via tripeptide substitution and phenol-OH protonation chemistry — exactly what blocks A+B encode. If overall clustered-50 MAE is flat **but** blue-cohort MAE drops by ≥ 2 nm, that alone is a publishable family-fairness contribution.

---

## Data-source ladder

| Priority | Source | Notes |
|---|---|---|
| 1 | `data/structure/minimized/{slug}_minimized.pdb` (913) | OpenMM vacuum minimization. **Default.** |
| 2 | `data/structure/grafted_fixed/{slug}_grafted.pdb` (913) | Pre-minimization (chromophore grafted from RCSB donor). Use if minimization distorts side chains > 0.5 Å. |
| 3 | `data/structure/esmfold_predictions/` (742) | ESMFold predicted backbones. Last resort — backbone only, weaker chromophore geometry. |
| 4 | Zero-vector + `has_pocket = 0` | Same convention as the existing `has_structure` flag. |

A day-2 spot check compares OH↔H148 distance across sources 1/2/3 on five known FPs (egfp, mcherry, dsred, sirius, mtagbfp). Systematic shift > 0.5 Å between minimized and grafted_fixed → switch primary to source 2.

---

## What this directory will *not* try

- **LoRA retraining with clustered folds.** ~30 h compute. Soft-leak already bounded ≤ 2 nm per [BENCHMARK_REPORT.md](../../benchmark/BENCHMARK_REPORT.md) §LoRA leak caveat. Out of 5-day scope.
- **Attention pooling on per-residue ESM-IF1.** Diagnostic analysis explicitly advised against this.
- **Re-including the hand-crafted 55-dim block** ("Block E"). Tier 1 showed `F_onehot+struct` is +2 nm worse than `A_onehot`. If Pocket-3D fails, we don't re-contaminate by adding back a known-bad block.
- **ChromaFormer changes.** Frozen per project memory.
