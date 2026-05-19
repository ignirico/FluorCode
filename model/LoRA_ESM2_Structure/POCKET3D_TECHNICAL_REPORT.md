# Pocket-3D Structural Features — Technical Report

**Module:** `model/LoRA_ESM2_Structure/`
**Author:** Rico Chi Kit Sou
**Date:** 2026-04-29

---

## 1. Overview

Pocket-3D is a hand-crafted, chromophore-HETATM-anchored 3D structural feature extractor for fluorescent protein (FP) property prediction. It produces **~95 dimensions** organized into 4 blocks (A/B/C/D), extracted from energy-minimized PDB structures using only NumPy — no neural network, no pre-trained structural encoder.

The key design principle: **anchor all spatial queries at the chromophore ligand atoms**, not at backbone CA. This is the fundamental difference from all prior structural featurizations attempted in this project.

---

## 2. Motivation — Why Not ESM-IF1

Two prior structural approaches were tested during the diagnostic phase and **both failed to improve over LoRA-ESM2 alone** (subset `B_lora`):

### 2.1 ESM-IF1 Embeddings (2048 dims) — Rejected

ESM-IF1 is a pre-trained inverse-folding GVP-GNN (12M parameters) that produces structure-conditioned sequence likelihoods. We pooled per-residue ESM-IF1 embeddings as `[mean | max | chrom±5 mean | chrom±10 mean]` to produce a 2048-dim vector per FP.

**Why it failed:**
- ESM-IF1 **explicitly drops all HETATM records** during PDB parsing. The chromophore — the molecule whose electronic structure determines all photophysical properties — is invisible to the model.
- The diagnostic PCA sweep (Tier 1b) showed no compressed version of ESM-IF1 helps: sweeping from 2048 → 512 → 256 → 128 → 64 dims, every configuration either matched or worsened `B_lora` MAE.
- Under clustered cross-validation (MMseqs2 ≥50% identity), adding ESM-IF1 to the feature set (`I_all` vs `E_onehot+lora`) produced differences < 0.5 nm — within noise.
- **Verdict (BENCHMARK_REPORT.md):** *"ESM-IF1 is noise — confirmed harmful under both random and clustered CV. Drop the 2048-dim structural block with confidence."*

### 2.2 Hand-Crafted CA Features (55 dims) — Rejected

A hand-crafted block computing shell composition, radius of gyration, and contact order, anchored at the **backbone CA atoms** of the chromophore triad (residues 65-66-67 in GFP numbering).

**Why it failed:**
- Anchored at CA, not at the phenol ring where photophysics lives. Key residues like E222, H148, T203, and R96 sit 3–6 Å from phenol-OH in 3D but are 30–160 residues away in sequence. CA-anchored shells at 6/8/10 Å capture bulk barrel composition, not the chemically relevant pocket.
- Adding this block to the one-hot baseline (`F_onehot+struct`) was **+2 nm worse** than one-hot alone (`A_onehot`).
- The gain-importance trap (Tier 2a diagnostic finding): XGBoost assigns high feature importance to any dense block regardless of predictive value, because dense blocks offer more split candidates.

### 2.3 The Gap Pocket-3D Addresses

Both failed approaches share one flaw: **blindness to the HETATM chromophore ligand**. ESM-IF1 drops it during parsing; the hand-crafted block ignores it by anchoring at backbone CA.

Pocket-3D anchors at the chromophore's own atoms — phenol-OH, imidazolinone centroid, phenol-ring centroid — and computes spatial features in the ligand's local coordinate frame. These features encode the hydrogen-bond network, electrostatic environment, and steric packing that directly modulate excitation/emission wavelengths, quantum yield, and pKa.

---

## 3. Structure Source Pipeline

Structures are **not** experimental crystal structures (only ~60 unique GFP-family structures exist in the PDB). Instead, they are computationally generated for all 913 GFP-family FPs:

| Step | Method | Output |
|---|---|---|
| 1. Backbone prediction | SimpleFold (ESMFold-based) | 742 predicted backbones |
| 2. Chromophore grafting | Superpose RCSB donor chromophore onto predicted backbone via triad CA alignment | 913 structures with HETATM chromophore |
| 3. Energy minimization | OpenMM vacuum minimization (AMBER ff14SB) | 913 minimized PDBs |

The grafting step is critical: ESMFold (and all PLM-based structure predictors) cannot model the post-translational chromophore maturation reaction. The chromophore HETATM must be inserted from an experimental donor structure and relaxed in context.

**Primary source:** `data/structure/minimized/{slug}_minimized.pdb`

---

## 4. Feature Blocks — Detailed Specification

### 4.1 Block A — Chromophore Chemistry & Geometry (20 dims)

Encodes the intrinsic chemical identity and internal geometry of the chromophore itself.

| # | Feature | Dims | Description |
|---|---|---|---|
| 1–8 | `is_first_{M,Q,G,T,S,H,C,other}` | 8 | One-hot encoding of the chromophore tripeptide's first residue (e.g., TYG→T, SYG→S). The first residue determines side-chain chemistry at position 65, which influences protonation equilibrium and π-system extension. Top 7 residues cover ~91% of the dataset; rarer ones collapse into `other`. |
| 9 | `phenol_planarity_rmsd` | 1 | RMSD of the 6 phenol ring atoms (CG2, CD1, CD2, CE1, CE2, CZ) from their best-fit plane via SVD. Non-planar = twisted chromophore = reduced quantum yield. |
| 10 | `imid_planarity_rmsd` | 1 | Same computation for the 5-membered imidazolinone ring (CA2, C1, N2, C2, N3). |
| 11–12 | `tau_cos`, `tau_sin` | 2 | τ dihedral angle (CA2–CB2–CG2–CD1): the exocyclic bond rotation connecting the two rings. This is the **primary geometric determinant of quantum yield** — a planar (τ ≈ 0°) chromophore fluoresces; a twisted one undergoes non-radiative decay. Encoded as cos/sin to avoid angle wraparound. |
| 13–14 | `phi_cos`, `phi_sin` | 2 | φ dihedral (N2–CA2–CB2–CG2): secondary torsion along the bridge. |
| 15 | `inter_ring_dihedral_cos` | 1 | Dot product of phenol-plane normal and imidazolinone-plane normal. Measures coplanarity of the two-ring π-system. Coplanar = maximal conjugation = red-shifted absorption. |
| 16 | `cz_oh_vs_phenol_normal_angle` | 1 | CZ→OH bond direction relative to phenol ring plane normal. Sanity dimension. |
| 17 | `oh_to_n2_dist` | 1 | Intra-chromophore OH↔N2 distance (Å). Encodes internal H-bond geometry. |
| 18 | `oh_to_ca3_dist` | 1 | OH↔CA3 distance (Å). |
| 19 | `chrom_heavy_rg` | 1 | Radius of gyration of all chromophore heavy atoms. Larger Rg indicates extended π-system (red FPs). |
| 20 | `atom_completeness` | 1 | Fraction of expected HETATM atoms actually found vs. the reference set for that tripeptide code. Quality flag — low completeness indicates grafting/minimization artifacts. |

### 4.2 Block B — Phenol-OH 3D Environment (35 dims)

Anchored at the **phenol hydroxyl oxygen (OH)** — the protonatable group whose ionization state (neutral vs. anionic) is the primary determinant of excitation/emission wavelength.

#### 4.2.1 Shell Composition (27 dims = 3 shells × 9 features)

For each distance cutoff (3.5 Å, 5.0 Å, 8.0 Å) from phenol-OH:

| Feature | Description |
|---|---|
| `oh_shell_{r}A_count` | Number of protein residues with any heavy atom within cutoff |
| `oh_shell_{r}A_{cat}_frac` | Fraction of those residues belonging to each of 8 categories: hydrophobic, polar, positive, negative, aromatic, H-bond donor, H-bond acceptor, Gly |

**Rationale:** The 3.5 Å shell captures the first coordination sphere (direct H-bond partners). The 5.0 Å shell captures the secondary shell that modulates electrostatics. The 8.0 Å shell captures the broader pocket environment. Category fractions encode the chemical character of the pocket without requiring residue identity.

#### 4.2.2 Named-Atom Nearest Distances (13 dims)

Minimum distance from phenol-OH to specific functional-group atoms on surrounding residues. Each feature returns the nearest instance; if none found within 20 Å, returns 20.0.

| Feature | Target atoms | Biological motivation |
|---|---|---|
| `oh_nearest_carboxylate_o` | Asp/Glu OD1, OD2, OE1, OE2 | E222 — the conserved proton acceptor in the GFP excited-state proton transfer (ESPT) relay |
| `oh_nearest_his_nitrogen` | His ND1, NE2 | H148 — stabilizes anionic phenolate via H-bond in EGFP (expected ~3.0–3.5 Å). Absence correlates with neutral chromophore (blue FPs) |
| `oh_nearest_tyr_oh` | Tyr OH | Adjacent Tyr residues in the barrel that may H-bond to chromophore OH |
| `oh_nearest_arg_guanidinium` | Arg NE, NH1, NH2 | R96 — conserved positive charge near chromophore in most GFP variants |
| `oh_nearest_lys_nz` | Lys NZ | **K163 — the hallmark red-shift residue in mCherry/DsRed family.** K163 NZ forms a direct H-bond to the chromophore phenolate, stabilizing the extended π-system |
| `oh_nearest_thr_og1` | Thr OG1 | **T203 — anionic phenolate stabilizer in EGFP/EYFP.** T203Y mutation in EYFP produces the yellow shift via π-stacking |
| `oh_nearest_ser_og` | Ser OG | **S205 — proton-wire relay residue in wild-type avGFP** (Chattoraj et al., 1996). Part of the OH→water→S205→E222 ESPT pathway |
| `oh_nearest_asn_od1` | Asn OD1 | N146 — backbone-amide H-bond acceptor partner near chromophore |
| `oh_nearest_asn_nd2` | Asn ND2 | N146 — donor nitrogen |
| `oh_nearest_gln_oe1` | Gln OE1 | Q69/Q94 — acceptor in several engineered variants |
| `oh_nearest_gln_ne2` | Gln NE2 | Q69/Q94 — donor |
| `oh_nearest_trp_ne1` | Trp NE1 | W57 — indole NH donor + π-stacking partner in blue FPs |

#### 4.2.3 Electrostatic Proxy (1 dim)

```
oh_electrostatic_proxy_8A = Σ qᵢ / dᵢ²
```

Summed over all side-chain polar atoms within 8 Å of phenol-OH, where qᵢ is the approximate formal charge at pH 7.0:

| Residue | Charge |
|---|---|
| Arg, Lys | +1.0 |
| His | +0.5 (partially protonated) |
| Asp, Glu | −1.0 |
| All others | 0.0 |

Charge is distributed equally across the side-chain polar atoms of each residue.

**Rationale:** The chromophore's excitation wavelength is sensitive to the local electrostatic field via the Stark effect. A full Poisson-Boltzmann calculation was rejected (Path 3) because the CRO chromophore lacks standard AMBER force-field parameters. This proxy captures the same physics at negligible compute cost.

### 4.3 Block C — Imidazolinone-Centroid 3D Environment (20 dims)

Same architecture as Block B but anchored at the **imidazolinone ring centroid** (mean coordinate of 5 ring atoms: CA2, C1, N2, C2, N3). The imidazolinone is the electron-accepting ring of the chromophore π-system; its environment modulates emission wavelength.

#### Shell Composition (18 dims = 2 shells × 9 features)
- Cutoffs: 5.0 Å, 8.0 Å (no 3.5 Å shell — fewer direct contacts to imidazolinone than to OH)

#### Named-Atom Distances (7 dims)

| Feature | Target | Motivation |
|---|---|---|
| `imid_nearest_cationic` | Arg/Lys NH1/NH2/NE/NZ | Positive charge near imidazolinone modulates emission |
| `imid_nearest_backbone_O_4A` | Any backbone O within 4 Å | Returns 1/d² (not distance). Backbone carbonyl H-bonds to imidazolinone N–H |
| `imid_electrostatic_proxy_8A` | (same Σ qᵢ/dᵢ² formula) | Electrostatic environment of the acceptor ring |
| `imid_nearest_thr_og1` | Thr OG1 | T/S hydroxyl H-bonds to imidazolinone N2/N3 |
| `imid_nearest_ser_og` | Ser OG | Same |
| `imid_nearest_trp_ring` | Any of 9 Trp ring heavy atoms | π-π stacking distance to tryptophan indole |
| `imid_nearest_phe_ring` | Any of 6 Phe ring heavy atoms | π-π stacking distance to phenylalanine |

### 4.4 Block D — Barrel Architecture (10 dims)

Global structural context — how the chromophore sits within the β-barrel scaffold.

| Feature | Description |
|---|---|
| `chrom_to_protein_centroid_dist` | Distance from chromophore centroid to protein CA centroid. Measures buriedness — deeply buried chromophores are shielded from solvent quenching. |
| `ca_cov_eigratio_1` | λ₂/λ₁ of the CA coordinate covariance matrix. Encodes barrel shape anisotropy. |
| `ca_cov_eigratio_2` | λ₃/λ₁. Together with eigratio_1, distinguishes prolate vs. oblate vs. spherical barrel geometry. |
| `ca_count_{3,5,7,10}A_phenol_centroid` | Number of CA atoms within 3/5/7/10 Å of phenol ring centroid. Local packing density at 4 spatial scales. |
| `pi_stacking_candidates_4A` | Aromatic residues (Phe/Tyr/Trp/His) whose CA is within 7 Å lateral and ≤4 Å perpendicular to the phenol ring plane. Counts potential π-stacking partners. |
| `chrom_sasa_proxy` | Fraction of chromophore heavy atoms whose nearest protein heavy atom is >5 Å. Proxy for solvent-accessible surface area without explicit solvent calculation. |
| `chrom_max_reach` | Maximum nearest-protein-atom distance across all chromophore heavy atoms. Identifies the most exposed part of the chromophore. |

**Caveat (from fallback ladder):** Block D is the most likely to leak family/size signal across clustered CV folds, because `chrom_to_protein_centroid_dist` and the CA covariance eigenvalues correlate with overall fold size. If Pocket-3D fails the mini-XGBoost gate, Block D is dropped first.

---

## 5. Implementation Details

### 5.1 PDB Parser

A minimal two-pass PDB parser (`parse_pdb()` in `build_pocket3d_features.py`):

- **Pass 1:** Identify the chain containing the chromophore HETATM (fallback: first chain).
- **Pass 2:** Collect per-residue atom coordinates + chromophore HETATM coordinates. Drops alternative conformers (altloc ≠ ' ' or 'A'). Only keeps the first chromophore encountered (avoids double-counting in multi-chromophore PDBs).

Output: `PDBStructure` with `ca_coords`, `ca_resnames`, `residue_atoms` (per-residue dict), `chrom_atoms` (chromophore dict), `chrom_resname`.

### 5.2 Chromophore Recognition

The module recognizes **42 HETATM residue names** covering:
- Matured GFP-like generic codes: CRO, CR2, CR8, CR7, CRU, CRQ, CRF
- Explicit tripeptide codes: TYG, SYG, GYG, MYG, CYG, HYG, AYG, EYG, QYG, etc.
- Red-FP variants: NRQ, NRP (mCherry/mScarlet/DsRed family)
- Blue/cyan/yellow variants: SWG, SHG, CH6, CH7, CFY
- Newer PDB codes: 4M9, 5SQ, 0WZ, 7R0, BJO, BJF, PIA, etc.

Self-policing: any HETATM that lacks phenol-OH + ≥3 phenol ring atoms gets `has_pocket=0` and a zero-vector.

### 5.3 Atom Naming Conventions

Chromophore HETATM atom names follow the RCSB Chemical Component Dictionary for CRO:
- Phenol ring: CG2, CD1, CD2, CE1, CE2, CZ
- Phenol hydroxyl: OH
- Bridge: CA2, CB2
- Imidazolinone ring: CA2, C1, N2, C2, N3 (alt: CA3 for C2)

### 5.4 Missing Data Handling

- Missing atoms → feature defaults to 0.0 (distances, dihedrals) or is skipped
- Missing structure entirely → full zero-vector + `has_pocket=0`
- `atom_completeness` computed per tripeptide code against reference expected-atom sets
- NaN/Inf sanitized to 0.0 before downstream use

---

## 6. Validation Protocol

The `validate_pocket3d.ipynb` notebook runs 12 cells of checks before committing to full benchmarking:

### 6.1 Integrity Gate
- No NaN or Inf in feature matrix
- `has_pocket` coverage ≥ 850 out of 913 (93%)
- `atom_completeness` mean ≥ 0.90 across `has_pocket=1` structures

### 6.2 Literature Spot Check (5 canonical FPs)

| FP | Expected | What we check |
|---|---|---|
| EGFP | OH↔H148 ND1 ≈ 3.0–3.5 Å | Anionic chromophore H-bond to His148 |
| Sirius | OH↔His > 6 Å | No nearby His → neutral chromophore → 355 nm excitation |
| DsRed | phenol↔M163 < 5 Å | Characteristic red-FP packing |
| mCherry | OH↔K163 NZ ~ 3.5 Å | Lys163 proton acceptor in DsRed family |
| mTagBFP | Modified chromophore | `atom_completeness` should flag unusual ring |

### 6.3 Correlation Sanity
- `tau_cos` vs QY: expect non-zero r (planarity governs quantum yield)
- `oh_electrostatic_proxy_8A` vs ex_max: expect non-zero r (electrostatic Stark effect)
- Any |r| > 0.10 on at least one pair = evidence features carry signal

### 6.4 Blue-Cliff Hypothesis Test
- Blue FPs (em_max < 470 nm, n≈41) had +11–14 nm prediction bias under `B_lora` alone
- Test whether Pocket-3D Block A+B features can separate blue vs. green FPs (protonated vs. anionic phenol chemistry)

### 6.5 Mini-XGBoost Gate (Decision Point)

5-fold random KFold on ex_max with three feature subsets:

| Subset | Features |
|---|---|
| `L_pocket3d_only` | Pocket-3D ~95 dims alone |
| `B_lora` | LoRA-ESM2 1280 dims (baseline) |
| `B_lora+pocket3d` | LoRA-ESM2 + Pocket-3D concatenated |

**Decision rule:**

| Δ MAE (B+P − B) | Verdict | Action |
|---|---|---|
| ≤ −0.3 nm | STRONG POSITIVE | Proceed to full clustered benchmark |
| < 0 nm | WEAK POSITIVE | Proceed; outcome uncertain |
| < +0.3 nm | NEUTRAL | Proceed but test secondary wins |
| ≥ +0.3 nm | NEGATIVE | Abort. Enter fallback ladder |

**Secondary win condition:** Even if overall MAE is flat, a blue-cohort MAE drop ≥ 2 nm is independently publishable as a family-fairness contribution.

---

## 7. Comparison Table: Pocket-3D vs ESM-IF1

| Property | ESM-IF1 | Pocket-3D |
|---|---|---|
| **Type** | Pre-trained GVP-GNN (12M params) | Hand-crafted physics-based features |
| **Dimensionality** | 2048 | ~95 |
| **Chromophore awareness** | None — drops all HETATM records | Anchored at chromophore HETATM atoms |
| **Anchor points** | Backbone N, CA, C, O only | Phenol-OH, imidazolinone centroid, phenol-ring centroid |
| **Physics encoded** | General protein fold topology | Chromophore-specific: τ/φ dihedrals, ring planarity, H-bond network, electrostatic environment, shell composition, packing density |
| **Compute** | GPU forward pass through 12M-param model | CPU NumPy, ~3 min for 913 PDBs |
| **Dependencies** | `esm-if1` model weights (~200 MB) | NumPy + PDB text parsing only |
| **Known failure mode** | Gain-importance trap: 2048 dense dims claim XGBoost splits regardless of signal | Block D may leak family-size signal; mitigated by fallback ladder |
| **Benchmark verdict** | "Confirmed harmful" — dropped | Under evaluation |

---

## 8. File Index

| File | Role |
|---|---|
| `build_pocket3d_features.py` | Feature extraction module (~775 lines). Contains PDB parser, 4 block compute functions, and `extract_all()` driver. |
| `pocket3d_atom_tables.py` | Pure-data module (~187 lines). Chromophore HETATM name registry (42 codes), expected atom sets per tripeptide, side-chain polar atom tables, formal charge table, residue category membership functions. |
| `build_pocket3d_features.ipynb` | Colab driver notebook. Mounts Drive, runs extraction, saves `pocket3d_features.npz`. ~3 min on CPU. |
| `validate_pocket3d.ipynb` | Validation notebook (12 cells). Integrity checks, literature spot check, correlation sanity, blue-cliff test, mini-XGBoost gate. ~20 min on H100. |
| `pocket3d_features.npz` | Pre-computed features for all 913 structures. Keys: `features` (N×~95), `feature_names`, `slugs`, `has_pocket`, `atom_completeness`. |
| `README_struct_paths.md` | Design decision log: why Pocket-3D was chosen over GearNet/GNN (Path 2) and APBS electrostatics (Path 3). |

---

## 9. References

- Remington, S. J. (2011). Green fluorescent protein: a perspective. *Protein Science*, 20(9), 1509–1519. — T203, H148, E222 roles in chromophore environment.
- Chattoraj, M. et al. (1996). Ultra-fast excited state dynamics in green fluorescent protein. *PNAS*, 93(16), 8362–8367. — S205→E222 proton-wire relay.
- Chudakov, D. M. et al. (2010). Fluorescent proteins and their applications in imaging living cells and tissues. *Physiol. Rev.*, 90(3), 1103–1163. — K163 in red FPs, general chromophore photophysics.
- Hsu, C. et al. (2022). Learning inverse folding from millions of predicted structures. *ICML 2022*. — ESM-IF1 architecture and HETATM exclusion.
