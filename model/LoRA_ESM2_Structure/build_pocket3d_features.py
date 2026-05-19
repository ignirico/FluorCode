"""
Pocket-3D chromophore-anchored structural features for FP property prediction.

Motivation:
every prior structural block (ESM-IF1 2048-dim, hand-crafted CA 55-dim) was
anchored at backbone-CA and blind to the HETATM chromophore ligand itself.
Photophysically relevant residues (E222, T203, H148, R96) sit 3-6 Å from the
phenol-ring in 3D but 30-160 residues away in sequence, so sequence-window
pooling misses them entirely. This module anchors at HETATM heavy atoms
(phenol-OH, imidazolinone centroid, ring centroid) and computes 3D-spatial
shells — features that neither ESM-IF1 nor the prior hand-crafted block ever
saw.

Output: `pocket3d_features.npz` with (v2 — Lever 1 expansion, ~95 dims):
    features          (N, ~95)  float32
    feature_names     (~95,)    str
    slugs             (N,)      str          (matches fp_embeddings_meta.csv order)
    has_pocket        (N,)      int8         (1 iff chromophore HETATM parsed OK)
    atom_completeness (N,)      float32      (fraction of expected HETATM atoms found)

v2 changes vs v1 (2026-04-17, see plans/lucky-inventing-mist.md Lever 1):
  - Block B +8 dims: oh_nearest_{lys_nz, thr_og1, ser_og, asn_od1, asn_nd2,
    gln_oe1, gln_ne2, trp_ne1} — K163/T203/S205/N146/Q69 chromophore-contact
    residues that the v1 block never queried.
  - Block C +4 dims: imid_nearest_{thr_og1, ser_og, trp_ring, phe_ring} —
    T/S hydroxyl H-bonds to imidazolinone, TRP/PHE π-π stacking.

Run:
    python -m model.LoRA_ESM2_Structure.build_pocket3d_features
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from pocket3d_atom_tables import (
    CHROM_HETATM_NAMES,
    PHENOL_RING_ATOMS,
    PHENOL_OH_ATOM,
    BRIDGE_ATOMS,
    IMID_RING_ATOMS,
    IMID_RING_ATOMS_ALT,
    CHROM_EXPECTED_ATOMS,
    SIDECHAIN_POLAR_ATOMS,
    FORMAL_CHARGE,
    CATEGORIES,
    aa_is,
)

ROOT       = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = ROOT / "data"
SEQ_DIR    = DATA_DIR / "sequence"
STRUCT_DIR = DATA_DIR / "structure" / "minimized"
OUT_DIR    = Path(__file__).resolve().parent

EPS = 1e-8


# ─── PDB parser ──────────────────────────────────────────────────────────────
class PDBStructure:
    """Minimal per-residue + HETATM atom store for one chain of one PDB.

    Attributes
    ----------
    ca_coords : (L, 3) float32      — CA coord per residue (in residue order)
    ca_resnames : list[str]         — 3-letter residue names aligned to ca_coords
    residue_atoms : list[dict]      — per-residue {atom_name: (x,y,z)}, aligned
    chrom_atoms : dict              — {atom_name: (x,y,z)} for the chromophore HETATM
    chrom_resname : str | None      — the 3-letter code that was parsed (CRO/SYG/...)
    """

    __slots__ = ("ca_coords", "ca_resnames", "residue_atoms",
                 "chrom_atoms", "chrom_resname", "chain")

    def __init__(self):
        self.ca_coords: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.ca_resnames: list[str] = []
        self.residue_atoms: list[dict] = []
        self.chrom_atoms: dict = {}
        self.chrom_resname: str | None = None
        self.chain: str | None = None


def parse_pdb(pdb_path: Path) -> PDBStructure | None:
    """Parse a PDB file and return a PDBStructure for the chain containing the
    chromophore HETATM (or the first chain if none). Returns None on failure.

    Follows the multi-chain dispatch pattern of build_structural_features.py.
    """
    # Pass 1: decide target chain (prefer chain containing chromophore HETATM).
    chains_seen: list[str] = []
    chrom_chain: str | None = None
    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                chain = line[21]
                if chain not in chains_seen:
                    chains_seen.append(chain)
                if line.startswith("HETATM") and line[17:20].strip() in CHROM_HETATM_NAMES:
                    if chrom_chain is None:
                        chrom_chain = chain
    except OSError:
        return None
    target_chain = chrom_chain if chrom_chain is not None else (chains_seen[0] if chains_seen else None)
    if target_chain is None:
        return None

    # Pass 2: collect per-residue atoms + chromophore HETATM atoms.
    struct = PDBStructure()
    struct.chain = target_chain

    per_res_atoms: dict[int, dict] = {}   # resid -> {atom_name: coord}
    per_res_name: dict[int, str] = {}     # resid -> 3-letter name (last-seen wins; altlocs unlikely on this field)
    residue_order: list[int] = []         # order in which resids first appear
    chrom_atoms: dict = {}
    chrom_resname: str | None = None

    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if line[21] != target_chain:
                    continue
                altloc = line[16]
                if altloc not in (" ", "A"):   # drop alternative conformers
                    continue
                atom_name = line[12:16].strip()
                resname   = line[17:20].strip()
                try:
                    resid = int(line[22:26])
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                coord = (x, y, z)

                if line.startswith("HETATM") and resname in CHROM_HETATM_NAMES:
                    # Only keep the first chromophore encountered (avoids double-count
                    # in weird multi-chromophore PDBs).
                    if chrom_resname is None:
                        chrom_resname = resname
                    if resname == chrom_resname:
                        chrom_atoms[atom_name] = coord
                    continue

                # Standard protein residue (or non-chrom HETATM like ions — skip).
                if line.startswith("HETATM"):
                    continue
                if resid not in per_res_atoms:
                    per_res_atoms[resid] = {}
                    residue_order.append(resid)
                per_res_name[resid] = resname
                per_res_atoms[resid][atom_name] = coord
    except OSError:
        return None

    if not residue_order:
        return None

    # Build aligned arrays.
    ca_coords_list = []
    ca_resnames = []
    residue_atoms = []
    for rid in residue_order:
        atoms = per_res_atoms[rid]
        if "CA" not in atoms:
            # No CA → skip (incomplete residue, e.g. truncated backbone)
            continue
        ca_coords_list.append(atoms["CA"])
        ca_resnames.append(per_res_name[rid])
        residue_atoms.append(atoms)

    if not ca_coords_list:
        return None

    struct.ca_coords    = np.asarray(ca_coords_list, dtype=np.float32)
    struct.ca_resnames  = ca_resnames
    struct.residue_atoms = residue_atoms
    struct.chrom_atoms  = chrom_atoms
    struct.chrom_resname = chrom_resname
    return struct


# ─── Geometry helpers ────────────────────────────────────────────────────────
def _get(struct: PDBStructure, *atom_names: str) -> np.ndarray | None:
    """Return the coordinates of the first chromophore atom matching any of the
    given names; None if no match."""
    for a in atom_names:
        if a in struct.chrom_atoms:
            return np.asarray(struct.chrom_atoms[a], dtype=np.float32)
    return None


def _plane_rmsd(points: np.ndarray) -> float:
    """RMSD of points from their best-fit plane (via SVD). 0 = coplanar."""
    if points.shape[0] < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]  # smallest-singular-value direction
    dist = centered @ normal
    return float(np.sqrt(np.mean(dist ** 2)))


def _plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    return n / (np.linalg.norm(n) + EPS)


def _dihedral(p1, p2, p3, p4) -> float:
    """Signed dihedral angle (radians) defined by 4 points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= (np.linalg.norm(n1) + EPS)
    n2 /= (np.linalg.norm(n2) + EPS)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + EPS))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


# ─── Block A: chromophore chemistry & geometry (20 dims) ─────────────────────
# Tripeptide first-residue one-hot. The HETATM resname in PDB files is usually
# the *matured* generic code (CRO / NRQ / etc.), not the pre-maturation
# tripeptide (TYG / SYG / etc.), so we read the tripeptide identity from
# chromophore_positions.csv (`chrom_residues` column) and one-hot the first
# residue. Coverage of the top 7 first-residue identities is ~91% of the dataset
# (M, Q, G, T, S, H, C); rarer ones (A, K, L, D, E, N, F, R) collapse into 'other'.
_FIRST_RES_ONEHOT = ("M", "Q", "G", "T", "S", "H", "C")     # ordered by frequency
BLOCK_A_NAMES = (
    [f"is_first_{r}" for r in _FIRST_RES_ONEHOT]
    + ["is_first_other"]
    + [
        "phenol_planarity_rmsd", "imid_planarity_rmsd",
        "tau_cos", "tau_sin", "phi_cos", "phi_sin",
        "inter_ring_dihedral_cos",
        "cz_oh_vs_phenol_normal_angle",
        "oh_to_n2_dist", "oh_to_ca3_dist",
        "chrom_heavy_rg",
        "atom_completeness",
    ]
)
_N_FIRST_ONEHOT = len(_FIRST_RES_ONEHOT) + 1  # +1 for 'other'


def compute_block_a(struct: PDBStructure, tripeptide: str | None = None) -> list[float]:
    feats = [0.0] * len(BLOCK_A_NAMES)
    if not struct.chrom_atoms or struct.chrom_resname is None:
        return feats

    # First-residue one-hot from chromophore_positions.csv (e.g., 'TYG' → T → is_first_T=1).
    # Falls through to is_first_other when tripeptide is None, has '-' placeholders,
    # or first residue is outside the top-7 set.
    if tripeptide and len(tripeptide) >= 1:
        first = tripeptide[0]
        if first in _FIRST_RES_ONEHOT:
            feats[_FIRST_RES_ONEHOT.index(first)] = 1.0
        elif first != "-":
            feats[len(_FIRST_RES_ONEHOT)] = 1.0  # is_first_other
    # else: leave all 8 first-residue features at 0 (signals tripeptide unknown)

    # Numeric-feature offset starts immediately after the 8-dim first-residue one-hot.
    j = _N_FIRST_ONEHOT

    # Phenol planarity RMSD (6 ring atoms).
    ring = []
    for a in PHENOL_RING_ATOMS:
        c = _get(struct, a)
        if c is not None:
            ring.append(c)
    ring_arr = np.stack(ring) if len(ring) >= 3 else np.zeros((0, 3), dtype=np.float32)
    feats[j + 0] = _plane_rmsd(ring_arr) if ring_arr.shape[0] >= 3 else 0.0

    # Imidazolinone planarity RMSD (try primary then alt naming).
    imid = []
    for a in IMID_RING_ATOMS:
        c = _get(struct, a)
        if c is not None:
            imid.append(c)
    if len(imid) < 3:
        imid = []
        for a in IMID_RING_ATOMS_ALT:
            c = _get(struct, a)
            if c is not None:
                imid.append(c)
    imid_arr = np.stack(imid) if len(imid) >= 3 else np.zeros((0, 3), dtype=np.float32)
    feats[j + 1] = _plane_rmsd(imid_arr) if imid_arr.shape[0] >= 3 else 0.0

    # τ dihedral CA2-CB2-CG2-CD1 (exocyclic bond rotation; governs QY).
    ca2 = _get(struct, "CA2"); cb2 = _get(struct, "CB2")
    cg2 = _get(struct, "CG2"); cd1 = _get(struct, "CD1")
    if all(x is not None for x in (ca2, cb2, cg2, cd1)):
        tau = _dihedral(ca2, cb2, cg2, cd1)
        feats[j + 2] = float(np.cos(tau)); feats[j + 3] = float(np.sin(tau))

    # φ dihedral N2-CA2-CB2-CG2.
    n2 = _get(struct, "N2")
    if all(x is not None for x in (n2, ca2, cb2, cg2)):
        phi = _dihedral(n2, ca2, cb2, cg2)
        feats[j + 4] = float(np.cos(phi)); feats[j + 5] = float(np.sin(phi))

    # Inter-ring dihedral angle between phenol-plane normal and imid-plane normal.
    if ring_arr.shape[0] >= 3 and imid_arr.shape[0] >= 3:
        np_ring = _plane_normal(ring_arr)
        np_imid = _plane_normal(imid_arr)
        if np_ring is not None and np_imid is not None:
            feats[j + 6] = float(np.clip(np.dot(np_ring, np_imid), -1.0, 1.0))

    # CZ-OH bond angle vs phenol-ring normal (sanity-dim; should be ≈ 90°).
    cz = _get(struct, "CZ"); oh = _get(struct, PHENOL_OH_ATOM)
    if cz is not None and oh is not None and ring_arr.shape[0] >= 3:
        bond = oh - cz
        bond /= (np.linalg.norm(bond) + EPS)
        n_ring = _plane_normal(ring_arr)
        if n_ring is not None:
            feats[j + 7] = float(np.clip(np.dot(bond, n_ring), -1.0, 1.0))

    # OH ↔ N2 distance (intra-chromophore H-bond possibility).
    if oh is not None and n2 is not None:
        feats[j + 8] = float(np.linalg.norm(oh - n2))

    # OH ↔ CA3 distance.
    ca3 = _get(struct, "CA3")
    if oh is not None and ca3 is not None:
        feats[j + 9] = float(np.linalg.norm(oh - ca3))

    # Chromophore heavy-atom Rg.
    if struct.chrom_atoms:
        heavy = np.asarray(list(struct.chrom_atoms.values()), dtype=np.float32)
        centroid = heavy.mean(axis=0)
        feats[j + 10] = float(np.sqrt(np.mean(np.sum((heavy - centroid) ** 2, axis=1))))

    # atom_completeness = fraction of expected HETATM atoms actually found.
    expected = CHROM_EXPECTED_ATOMS.get(struct.chrom_resname, set())
    if expected:
        found = len(set(struct.chrom_atoms.keys()) & expected)
        feats[j + 11] = float(found) / float(len(expected))

    return feats


# ─── Block B: phenol-OH 3D environment (35 dims) ─────────────────────────────
_B_SHELLS = (3.5, 5.0, 8.0)    # Å

_B_NAMES_BASE = []
for s in _B_SHELLS:
    tag = f"{s:g}A"
    _B_NAMES_BASE.append(f"oh_shell_{tag}_count")
    for cat in CATEGORIES:
        _B_NAMES_BASE.append(f"oh_shell_{tag}_{cat}_frac")

BLOCK_B_NAMES = _B_NAMES_BASE + [
    "oh_nearest_carboxylate_o",
    "oh_nearest_his_nitrogen",
    "oh_nearest_tyr_oh",
    "oh_nearest_arg_guanidinium",
    "oh_electrostatic_proxy_8A",
    # Lever-1 additions (v2): atom-pair queries for key chromophore-contact residues
    # that photophysics literature (Remington 2011; Chudakov 2010) implicates in
    # ex_max / em_max shifts but that the v1 block never queried.
    "oh_nearest_lys_nz",          # K163 — mCherry/DsRed red-shift
    "oh_nearest_thr_og1",         # T203 — anionic phenolate stabilizer (EGFP/EYFP)
    "oh_nearest_ser_og",          # S205 — proton-wire relay (wild-type GFP)
    "oh_nearest_asn_od1",         # N146 backbone-amide acceptor partner
    "oh_nearest_asn_nd2",         # N146 donor
    "oh_nearest_gln_oe1",         # Q69/Q94 acceptor
    "oh_nearest_gln_ne2",         # Q69/Q94 donor
    "oh_nearest_trp_ne1",         # W57/W97 π-stacking + indole-NH donor
]


def _shell_categorical(struct: PDBStructure, anchor: np.ndarray, cutoff: float,
                       exclude_chrom_chain_neighbors: bool = True) -> tuple[int, dict]:
    """Return (count, {category_name: frac}) for residues with ANY heavy atom
    within `cutoff` Å of `anchor`. Category fractions sum to residue count
    (NOT 1.0) — they are per-category per-residue indicator sums normalized by
    count at the end.
    """
    counts_per_cat = {cat: 0 for cat in CATEGORIES}
    n_res = 0
    for res_atoms, resname in zip(struct.residue_atoms, struct.ca_resnames):
        # Minimum-heavy-atom distance (exclude backbone-only residues? no — use all).
        min_d = None
        for coord in res_atoms.values():
            d = float(np.linalg.norm(np.asarray(coord, dtype=np.float32) - anchor))
            if min_d is None or d < min_d:
                min_d = d
        if min_d is None or min_d > cutoff:
            continue
        n_res += 1
        for cat in CATEGORIES:
            counts_per_cat[cat] += aa_is(resname, cat)

    fracs = {cat: (counts_per_cat[cat] / n_res if n_res > 0 else 0.0) for cat in CATEGORIES}
    return n_res, fracs


def _nearest_named_atom_distance(struct: PDBStructure, anchor: np.ndarray,
                                  resnames: set[str], atom_names: Iterable[str],
                                  max_dist: float = 20.0) -> float:
    """Return the distance to the nearest heavy atom `atom_name` in any residue
    of the given 3-letter name set. If nothing found within max_dist, return max_dist.
    """
    atom_set = set(atom_names)
    best = max_dist
    for res_atoms, resname in zip(struct.residue_atoms, struct.ca_resnames):
        if resname not in resnames:
            continue
        for a_name, coord in res_atoms.items():
            if a_name not in atom_set:
                continue
            d = float(np.linalg.norm(np.asarray(coord, dtype=np.float32) - anchor))
            if d < best:
                best = d
    return best


def _electrostatic_proxy(struct: PDBStructure, anchor: np.ndarray, cutoff: float = 8.0) -> float:
    """Signed Σ q_i / d_i² over side-chain polar atoms within `cutoff` Å.

    Uses the approximate formal-charge table (His=+0.5, Arg/Lys=+1, Asp/Glu=−1)
    spread equally across the atoms listed in SIDECHAIN_POLAR_ATOMS for that residue.
    """
    acc = 0.0
    for res_atoms, resname in zip(struct.residue_atoms, struct.ca_resnames):
        q_total = FORMAL_CHARGE.get(resname, 0.0)
        if q_total == 0.0:
            continue
        polar_names = SIDECHAIN_POLAR_ATOMS.get(resname, ())
        if not polar_names:
            continue
        q_per_atom = q_total / len(polar_names)
        for a_name in polar_names:
            coord = res_atoms.get(a_name)
            if coord is None:
                continue
            d = float(np.linalg.norm(np.asarray(coord, dtype=np.float32) - anchor))
            if d > cutoff or d < EPS:
                continue
            acc += q_per_atom / (d * d)
    return acc


def compute_block_b(struct: PDBStructure) -> list[float]:
    feats = [0.0] * len(BLOCK_B_NAMES)
    oh = _get(struct, PHENOL_OH_ATOM)
    if oh is None:
        return feats

    # Shell categorical fractions.
    idx = 0
    for s in _B_SHELLS:
        n_res, fracs = _shell_categorical(struct, oh, s)
        feats[idx] = float(n_res); idx += 1
        for cat in CATEGORIES:
            feats[idx] = fracs[cat]; idx += 1

    # Named-atom nearest distances (idx now at end of shell block).
    feats[idx]     = _nearest_named_atom_distance(struct, oh, {"ASP", "GLU"}, ("OD1", "OD2", "OE1", "OE2"))
    feats[idx + 1] = _nearest_named_atom_distance(struct, oh, {"HIS"},        ("ND1", "NE2"))
    feats[idx + 2] = _nearest_named_atom_distance(struct, oh, {"TYR"},        ("OH",))
    feats[idx + 3] = _nearest_named_atom_distance(struct, oh, {"ARG"},        ("NH1", "NH2", "NE"))
    feats[idx + 4] = _electrostatic_proxy(struct, oh, cutoff=8.0)
    # Lever-1 additions.
    feats[idx + 5]  = _nearest_named_atom_distance(struct, oh, {"LYS"}, ("NZ",))
    feats[idx + 6]  = _nearest_named_atom_distance(struct, oh, {"THR"}, ("OG1",))
    feats[idx + 7]  = _nearest_named_atom_distance(struct, oh, {"SER"}, ("OG",))
    feats[idx + 8]  = _nearest_named_atom_distance(struct, oh, {"ASN"}, ("OD1",))
    feats[idx + 9]  = _nearest_named_atom_distance(struct, oh, {"ASN"}, ("ND2",))
    feats[idx + 10] = _nearest_named_atom_distance(struct, oh, {"GLN"}, ("OE1",))
    feats[idx + 11] = _nearest_named_atom_distance(struct, oh, {"GLN"}, ("NE2",))
    feats[idx + 12] = _nearest_named_atom_distance(struct, oh, {"TRP"}, ("NE1",))

    return feats


# ─── Block C: imidazolinone-centroid 3D environment (20 dims) ────────────────
_C_SHELLS = (5.0, 8.0)

_C_NAMES_BASE = []
for s in _C_SHELLS:
    tag = f"{s:g}A"
    _C_NAMES_BASE.append(f"imid_shell_{tag}_count")
    for cat in CATEGORIES:
        _C_NAMES_BASE.append(f"imid_shell_{tag}_{cat}_frac")

BLOCK_C_NAMES = _C_NAMES_BASE + [
    "imid_nearest_cationic",
    "imid_nearest_backbone_O_4A",
    "imid_electrostatic_proxy_8A",
    # Lever-1 additions (v2): atom-pair queries for imidazolinone-centroid
    # environment. T/S hydroxyls H-bond to the imidazolinone N2 / N3 in
    # multiple structurally-solved FPs; TRP/PHE ring-centroid distances
    # encode π-π stabilization of the imidazolinone ring.
    "imid_nearest_thr_og1",
    "imid_nearest_ser_og",
    "imid_nearest_trp_ring",          # min dist to any TRP ring heavy atom
    "imid_nearest_phe_ring",          # min dist to any PHE ring heavy atom
]

# Aromatic-ring heavy atoms for ring-ring distance queries.
_TRP_RING_ATOMS = ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")
_PHE_RING_ATOMS = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")


def _imid_centroid(struct: PDBStructure) -> np.ndarray | None:
    atoms = []
    for a in IMID_RING_ATOMS:
        c = _get(struct, a)
        if c is not None:
            atoms.append(c)
    if len(atoms) < 3:
        atoms = []
        for a in IMID_RING_ATOMS_ALT:
            c = _get(struct, a)
            if c is not None:
                atoms.append(c)
    if len(atoms) < 3:
        return None
    return np.mean(np.stack(atoms), axis=0)


def _nearest_backbone_carbonyl_O(struct: PDBStructure, anchor: np.ndarray,
                                  cutoff: float = 4.0) -> float:
    """Return 1/d² to nearest backbone carbonyl O within cutoff, else 0."""
    best = None
    for res_atoms in struct.residue_atoms:
        o = res_atoms.get("O")
        if o is None:
            continue
        d = float(np.linalg.norm(np.asarray(o, dtype=np.float32) - anchor))
        if d > cutoff:
            continue
        if best is None or d < best:
            best = d
    return float(1.0 / (best * best)) if best is not None else 0.0


def compute_block_c(struct: PDBStructure) -> list[float]:
    feats = [0.0] * len(BLOCK_C_NAMES)
    centroid = _imid_centroid(struct)
    if centroid is None:
        return feats

    idx = 0
    for s in _C_SHELLS:
        n_res, fracs = _shell_categorical(struct, centroid, s)
        feats[idx] = float(n_res); idx += 1
        for cat in CATEGORIES:
            feats[idx] = fracs[cat]; idx += 1

    feats[idx]     = _nearest_named_atom_distance(struct, centroid, {"ARG", "LYS"}, ("NH1", "NH2", "NE", "NZ"))
    feats[idx + 1] = _nearest_backbone_carbonyl_O(struct, centroid, cutoff=4.0)
    feats[idx + 2] = _electrostatic_proxy(struct, centroid, cutoff=8.0)
    # Lever-1 additions.
    feats[idx + 3] = _nearest_named_atom_distance(struct, centroid, {"THR"}, ("OG1",))
    feats[idx + 4] = _nearest_named_atom_distance(struct, centroid, {"SER"}, ("OG",))
    feats[idx + 5] = _nearest_named_atom_distance(struct, centroid, {"TRP"}, _TRP_RING_ATOMS)
    feats[idx + 6] = _nearest_named_atom_distance(struct, centroid, {"PHE"}, _PHE_RING_ATOMS)
    return feats


# ─── Block D: barrel architecture (10 dims) ─────────────────────────────────
BLOCK_D_NAMES = [
    "chrom_to_protein_centroid_dist",
    "ca_cov_eigratio_1",
    "ca_cov_eigratio_2",
    "ca_count_3A_phenol_centroid",
    "ca_count_5A_phenol_centroid",
    "ca_count_7A_phenol_centroid",
    "ca_count_10A_phenol_centroid",
    "pi_stacking_candidates_4A",
    "chrom_sasa_proxy",
    "chrom_max_reach",
]


def compute_block_d(struct: PDBStructure) -> list[float]:
    feats = [0.0] * len(BLOCK_D_NAMES)
    if len(struct.ca_coords) < 4 or not struct.chrom_atoms:
        return feats

    phenol = []
    for a in PHENOL_RING_ATOMS:
        c = _get(struct, a)
        if c is not None:
            phenol.append(c)
    if len(phenol) < 3:
        return feats
    phenol_arr = np.stack(phenol)
    phenol_centroid = phenol_arr.mean(axis=0)

    # Full chromophore heavy atoms.
    heavy = np.asarray(list(struct.chrom_atoms.values()), dtype=np.float32)
    chrom_centroid = heavy.mean(axis=0)

    # 1. Buriedness: chromophore centroid ↔ protein (CA) centroid.
    protein_centroid = struct.ca_coords.mean(axis=0)
    feats[0] = float(np.linalg.norm(chrom_centroid - protein_centroid))

    # 2-3. CA covariance anisotropy (eigvals sorted desc).
    ca_c = struct.ca_coords - protein_centroid
    cov = (ca_c.T @ ca_c) / max(len(ca_c) - 1, 1)
    eig = np.linalg.eigvalsh(cov.astype(np.float64))
    eig = np.sort(eig)[::-1]  # descending
    l1, l2, l3 = float(eig[0] + EPS), float(eig[1] + EPS), float(eig[2] + EPS)
    feats[1] = l2 / l1
    feats[2] = l3 / l1

    # 4-7. CA counts in shells around phenol centroid.
    d_ca = np.linalg.norm(struct.ca_coords - phenol_centroid, axis=1)
    for i, r in enumerate((3.0, 5.0, 7.0, 10.0)):
        feats[3 + i] = float((d_ca <= r).sum())

    # 8. π-stacking candidates: aromatic residues whose CA is within 4 Å of phenol plane.
    n_ring = _plane_normal(phenol_arr)
    n_stack = 0
    if n_ring is not None:
        for ca, resname in zip(struct.ca_coords, struct.ca_resnames):
            if resname not in ("PHE", "TYR", "TRP", "HIS"):
                continue
            if float(np.linalg.norm(ca - phenol_centroid)) > 7.0:
                continue
            plane_dist = abs(float(np.dot(ca - phenol_centroid, n_ring)))
            if plane_dist <= 4.0:
                n_stack += 1
    feats[7] = float(n_stack)

    # 9. SASA proxy: fraction of chromophore heavy atoms whose nearest protein
    #    heavy atom is > 5 Å (exposed).
    # Gather protein heavy-atom coords (subsample to CA + selected side-chain polar
    # atoms to keep this fast; good enough as a proxy).
    prot_heavy = []
    for res_atoms in struct.residue_atoms:
        for coord in res_atoms.values():
            prot_heavy.append(coord)
    prot_heavy_arr = np.asarray(prot_heavy, dtype=np.float32)
    if len(prot_heavy_arr):
        diffs = heavy[:, None, :] - prot_heavy_arr[None, :, :]
        min_d_per_chrom = np.sqrt((diffs ** 2).sum(axis=2)).min(axis=1)
        feats[8] = float((min_d_per_chrom > 5.0).mean())
        # 10. Max reach (largest chromophore→protein heavy atom min-distance).
        feats[9] = float(min_d_per_chrom.max())
    return feats


# ─── Driver ──────────────────────────────────────────────────────────────────
ALL_FEATURE_NAMES: list[str] = (
    [f"A__{n}" for n in BLOCK_A_NAMES]
    + [f"B__{n}" for n in BLOCK_B_NAMES]
    + [f"C__{n}" for n in BLOCK_C_NAMES]
    + [f"D__{n}" for n in BLOCK_D_NAMES]
)


def _load_tripeptide_map(seq_dir: Path = SEQ_DIR) -> dict[str, str]:
    """Load slug → tripeptide string (e.g., 'TYG') from chromophore_positions.csv.
    Missing slugs / dashed placeholders return empty string.
    """
    path = seq_dir / "chromophore_positions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(r["slug"]): str(r["chrom_residues"]) for _, r in df.iterrows()}


def extract_for_slug(slug: str, pdb_dir: Path = STRUCT_DIR,
                     tripeptide: str | None = None) -> tuple[np.ndarray, int, float]:
    """Extract the concatenated Pocket-3D feature vector for one slug.

    Parameters
    ----------
    tripeptide : str | None
        Pre-maturation tripeptide string from chromophore_positions.csv
        (e.g., 'TYG'). Used for the first-residue one-hot in Block A.

    Returns
    -------
    feats : (n_dim,) float32
    has_pocket : int (1 iff chromophore HETATM + phenol OH both parsed)
    atom_completeness : float (0.0 if no chromophore found)
    """
    n_dim = len(ALL_FEATURE_NAMES)
    zero = np.zeros(n_dim, dtype=np.float32)

    pdb = pdb_dir / f"{slug}_minimized.pdb"
    if not pdb.exists():
        return zero, 0, 0.0

    struct = parse_pdb(pdb)
    if struct is None or not struct.chrom_atoms:
        return zero, 0, 0.0

    a = compute_block_a(struct, tripeptide=tripeptide)
    b = compute_block_b(struct)
    c = compute_block_c(struct)
    d = compute_block_d(struct)
    vec = np.asarray(a + b + c + d, dtype=np.float32)

    # atom_completeness is the last numeric feature of Block A
    # (after the 8-dim first-residue one-hot, then 11 numeric features).
    completeness = float(a[_N_FIRST_ONEHOT + 11])

    # has_pocket: phenol-OH present AND at least 3 phenol ring atoms.
    phenol_atoms_found = sum(1 for name in PHENOL_RING_ATOMS if name in struct.chrom_atoms)
    has_pocket = 1 if (PHENOL_OH_ATOM in struct.chrom_atoms and phenol_atoms_found >= 3) else 0

    return vec, has_pocket, completeness


def extract_all(meta_csv: Path, pdb_dir: Path = STRUCT_DIR,
                verbose: bool = True) -> dict:
    """Run extraction across every slug in the meta file.

    Returns dict with keys: features, feature_names, slugs, has_pocket,
    atom_completeness. Save with `np.savez(path, **result)`.
    """
    meta = pd.read_csv(meta_csv)
    n = len(meta)
    n_dim = len(ALL_FEATURE_NAMES)

    # Load pre-maturation tripeptide map (drives Block A first-residue one-hot).
    trip_map = _load_tripeptide_map(seq_dir=meta_csv.parent)
    if verbose:
        print(f"  loaded tripeptide map for {len(trip_map)} slugs from chromophore_positions.csv")

    X = np.zeros((n, n_dim), dtype=np.float32)
    has_pocket = np.zeros(n, dtype=np.int8)
    atom_completeness = np.zeros(n, dtype=np.float32)
    slugs = meta["slug"].astype(str).values

    for i, slug in enumerate(slugs):
        vec, hp, ac = extract_for_slug(slug, pdb_dir=pdb_dir, tripeptide=trip_map.get(slug))
        X[i] = vec
        has_pocket[i] = hp
        atom_completeness[i] = ac
        if verbose and (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{n}  (has_pocket so far: {int(has_pocket.sum())})")

    if verbose:
        print(f"\nTotal: {int(has_pocket.sum())}/{n} FPs with valid chromophore pocket")
        print(f"Mean atom_completeness (over has_pocket=1): "
              f"{float(atom_completeness[has_pocket == 1].mean()) if has_pocket.sum() else 0.0:.3f}")

    return {
        "features": X,
        "feature_names": np.asarray(ALL_FEATURE_NAMES),
        "slugs": slugs.astype(object),
        "has_pocket": has_pocket,
        "atom_completeness": atom_completeness,
    }


def main() -> Path:
    out = extract_all(SEQ_DIR / "fp_embeddings_meta.csv", pdb_dir=STRUCT_DIR, verbose=True)
    out_path = OUT_DIR / "pocket3d_features.npz"
    np.savez(out_path, **out)
    print(f"\nSaved {out['features'].shape} → {out_path}")
    return out_path


if __name__ == "__main__":
    main()
