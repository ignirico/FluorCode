"""
Structure Parsing Pipeline
===========================
Parse PDB structures (RCSB experimental + SimpleFold predictions)
and extract per-residue features for graph construction.

Handles:
  - RCSB PDB files (multi-chain, HETATM records) → extract chain A
  - SimpleFold PDB files (single chain, clean ATOM records)

Usage:
    python3 parse_structures.py

Output:
    data/structure_features.pkl — dict[slug] → {
        "ca_coords": np.array (N, 3),
        "residue_names": list[str],
        "residue_indices": np.array (N,),
        "is_experimental": bool,
    }
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
STRUCT_DIR = DATA_DIR / "structure"
SF_PRED    = DATA_DIR / "simplefold_output" / "predictions_simplefold_100M"
META_CSV   = DATA_DIR / "sequence" / "fp_cleaned.csv"
OUT_PKL    = DATA_DIR / "structure_features.pkl"

# RCSB structures are in STRUCT_DIR, named {slug}.pdb
# SimpleFold predictions are in SF_PRED, named {slug}_sampled_0.pdb


def parse_pdb_ca(pdb_path: Path, chain_id: str = "A") -> dict | None:
    """Extract CA atom coordinates and residue info from a PDB file.

    For multi-chain PDBs, only the specified chain is used.
    Returns None if parsing fails or no CA atoms found.
    """
    ca_coords = []
    residue_names = []
    residue_indices = []
    seen_residues = set()

    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                chain = line[21].strip()
                # For multi-chain, filter to target chain
                if chain and chain != chain_id:
                    continue

                res_name = line[17:20].strip()
                res_seq  = line[22:27].strip()  # includes insertion code

                # Skip duplicate residues (alt conformations)
                res_key = (chain, res_seq)
                if res_key in seen_residues:
                    continue
                seen_residues.add(res_key)

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                ca_coords.append([x, y, z])
                residue_names.append(res_name)
                residue_indices.append(int(res_seq.rstrip("ABCDEFGHIJ ")))

    except Exception as e:
        print(f"  ERROR parsing {pdb_path.name}: {e}")
        return None

    if len(ca_coords) == 0:
        return None

    return {
        "ca_coords": np.array(ca_coords, dtype=np.float32),
        "residue_names": residue_names,
        "residue_indices": np.array(residue_indices, dtype=np.int32),
    }


# Standard 3-letter to 1-letter mapping (for validation)
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Modified residues that appear in RCSB PDBs
    "MSE": "M", "SEC": "C", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CRO": "X",  # Chromophore — not a standard residue
}


def main():
    meta = pd.read_csv(META_CSV)
    slugs = set(meta["slug"].tolist())
    print(f"Total FPs: {len(slugs)}")

    # Load existing results for incremental parsing
    if OUT_PKL.exists():
        with open(OUT_PKL, "rb") as f:
            features = pickle.load(f)
        print(f"Loaded {len(features)} existing parsed structures")
    else:
        features = {}

    # Collect all available PDB paths
    pdb_sources = {}

    # RCSB experimental structures (priority)
    for pdb in STRUCT_DIR.glob("*.pdb"):
        slug = pdb.stem
        if slug in slugs:
            pdb_sources[slug] = (pdb, True)  # (path, is_experimental)

    # SimpleFold predictions (fill in remaining)
    if SF_PRED.exists():
        for pdb in SF_PRED.glob("*_sampled_0.pdb"):
            slug = pdb.stem.replace("_sampled_0", "")
            if slug in slugs and slug not in pdb_sources:
                pdb_sources[slug] = (pdb, False)

    print(f"PDB files available: {len(pdb_sources)} "
          f"(RCSB: {sum(1 for _, e in pdb_sources.values() if e)}, "
          f"SimpleFold: {sum(1 for _, e in pdb_sources.values() if not e)})")

    # Parse new structures
    n_new, n_fail = 0, 0
    to_parse = {s: v for s, v in pdb_sources.items() if s not in features}
    print(f"To parse: {len(to_parse)} new structures")

    for i, (slug, (pdb_path, is_exp)) in enumerate(to_parse.items()):
        result = parse_pdb_ca(pdb_path)
        if result is None:
            # Try other chains for RCSB structures
            if is_exp:
                for alt_chain in ["B", "C", " "]:
                    result = parse_pdb_ca(pdb_path, chain_id=alt_chain)
                    if result is not None:
                        break
            if result is None:
                print(f"  SKIP [{slug}] — no CA atoms found")
                n_fail += 1
                continue

        result["is_experimental"] = is_exp
        features[slug] = result
        n_new += 1

        if (i + 1) % 50 == 0:
            print(f"  Parsed {i+1}/{len(to_parse)}...")

    # Save
    with open(OUT_PKL, "wb") as f:
        pickle.dump(features, f)

    n_exp = sum(1 for v in features.values() if v["is_experimental"])
    n_pred = len(features) - n_exp
    print(f"\nDone. Parsed {n_new} new, {n_fail} failed")
    print(f"Total in {OUT_PKL.name}: {len(features)} "
          f"(experimental: {n_exp}, predicted: {n_pred})")


if __name__ == "__main__":
    main()
