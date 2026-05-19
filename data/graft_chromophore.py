"""
Chromophore Grafting Pipeline
==============================
For predicted structures (no chromophore), find the most similar
protein among the RCSB PDB structures, then graft the chromophore HETATM
records from that donor onto the predicted structure.

Uses SimpleFold predictions as primary structure source (better quality,
0% placeholder atoms), with ESMFold as fallback.

Steps:
  0. Fetch experimental PDB files from RCSB for FPs with known PDB IDs
  1. Build a donor library: scan RCSB PDBs for chromophore HETATM records
  2. Match each target to the best donor by sequence identity (via MSA)
  3. Superimpose local backbone (chromophore ± flanking) and graft HETATM coords
  4. Validate grafts (RMSD, clashes, identity thresholds)

Usage:
    python3 graft_chromophore.py

Inputs:
    data/sequence/fp_cleaned.csv
    data/sequence/chromophore_positions.csv
    data/sequence/fp_sequences_aligned.fasta
    data/structure/simplefold/predictions_simplefold_100M/*.pdb  (primary)
    data/structure/esmfold_predictions/*.pdb                     (fallback)

Outputs:
    data/structure/rcsb_donors/{slug}.pdb (downloaded experimental PDBs)
    data/structure/grafted/{slug}_grafted.pdb
    data/structure/graft_summary.csv
"""

import time
import shutil
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.SVDSuperimposer import SVDSuperimposer

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SEQ_DIR    = ROOT / "sequence"
META_CSV   = SEQ_DIR / "fp_cleaned.csv"
CHROM_CSV  = SEQ_DIR / "chromophore_positions.csv"
ALN_FILE   = SEQ_DIR / "fp_sequences_aligned.fasta"

SIMPLEFOLD_DIR = ROOT / "structure" / "simplefold" / "predictions_simplefold_100M"
ESMFOLD_DIR    = ROOT / "structure" / "esmfold_predictions"  # fallback
RCSB_DIR       = ROOT / "structure" / "rcsb_donors"
GRAFTED_DIR  = ROOT / "structure" / "grafted"
SUMMARY_CSV  = ROOT / "structure" / "graft_summary.csv"

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Known chromophore HETATM residue names across FP families
CHROM_HETATM_NAMES = {
    # GFP-type autocatalytic chromophores
    "CRO", "CR2", "GYS", "SYG", "TYG", "66A", "NRQ", "CH6", "CH7",
    "CFP", "GFP", "MYG", "QYG",
    # Biliverdin / phytochrome cofactors
    "BLA", "BV", "BPB", "BV1", "BV2",
    # Flavin
    "FMN", "FAD", "RBF",
    # Generic / catch-all
    "HBI", "CHR",
}

# Thresholds
RMSD_WARN    = 1.5   # Angstrom — flag grafts above this
CLASH_DIST   = 2.0   # Angstrom — atom pairs closer than this are clashes
CLASH_MAX    = 5      # flag if more than this many clashes
IDENT_WARN   = 0.30   # flag donors below 30% identity
FLANK_SIZE   = 5      # residues on each side of chromophore triad for alignment

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── PDB Parsing Helpers ──────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: Path, chain_id: str = "A") -> list[dict]:
    """Parse all ATOM/HETATM records from a PDB file for a given chain.

    Returns list of dicts with keys:
        record_type, atom_name, res_name, chain, res_seq, x, y, z, line
    """
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip()
            if chain and chain != chain_id:
                continue
            atoms.append({
                "record_type": line[:6].strip(),
                "atom_name": line[12:16].strip(),
                "res_name": line[17:20].strip(),
                "chain": chain,
                "res_seq": int(line[22:26].strip()),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "line": line,
            })
    return atoms


def get_ca_coords_by_resseq(atoms: list[dict], res_seqs: list[int]) -> np.ndarray:
    """Extract CA coordinates for specific residue sequence numbers.

    Returns (N, 3) array. Raises ValueError if any residue is missing.
    """
    ca_map = {}
    for a in atoms:
        if a["atom_name"] == "CA" and a["res_seq"] not in ca_map:
            ca_map[a["res_seq"]] = [a["x"], a["y"], a["z"]]

    coords = []
    for rs in res_seqs:
        if rs not in ca_map:
            raise ValueError(f"Residue {rs} not found in CA atoms")
        coords.append(ca_map[rs])
    return np.array(coords, dtype=np.float64)


def get_hetatm_lines(atoms: list[dict], res_names: set[str]) -> list[dict]:
    """Extract all HETATM records matching given residue names."""
    return [a for a in atoms if a["record_type"] == "HETATM" and a["res_name"] in res_names]


def find_best_chain(pdb_path: Path, chrom_names: set[str]) -> str | None:
    """Find the chain that contains chromophore HETATM records."""
    chains_with_chrom = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name in chrom_names:
                    chains_with_chrom.add(line[21].strip())
    # Prefer chain A, then alphabetical
    for ch in ["A", "B", "C", "D", " "]:
        if ch in chains_with_chrom:
            return ch
    return chains_with_chrom.pop() if chains_with_chrom else None


# ── Step 0: Fetch RCSB Donor PDBs ────────────────────────────────────────────

def fetch_rcsb_donors(meta: pd.DataFrame):
    """Download experimental PDB files from RCSB for FPs with known PDB IDs."""
    RCSB_DIR.mkdir(parents=True, exist_ok=True)

    has_pdb = meta["pdb"].notna() & (meta["pdb"].astype(str).str.strip() != "")
    meta_pdb = meta[has_pdb].copy()
    meta_pdb["pdb_id"] = meta_pdb["pdb"].astype(str).str.split("|").str[0].str.strip()

    done = {p.stem for p in RCSB_DIR.glob("*.pdb")}
    todo = meta_pdb[~meta_pdb["slug"].isin(done)]

    log.info(f"[RCSB] FPs with PDB ID: {len(meta_pdb)}  |  "
             f"Already fetched: {len(done & set(meta_pdb['slug']))}  |  "
             f"To fetch: {len(todo)}")

    n_ok, n_fail = 0, 0
    for _, row in todo.iterrows():
        slug, pdb_id = row["slug"], row["pdb_id"].upper()
        try:
            resp = requests.get(RCSB_URL.format(pdb_id=pdb_id), timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                (RCSB_DIR / f"{slug}.pdb").write_text(resp.text)
                n_ok += 1
            else:
                log.warning(f"  SKIP [{slug}] {pdb_id} — HTTP {resp.status_code}")
                n_fail += 1
        except Exception as e:
            log.warning(f"  ERROR [{slug}] {pdb_id} — {e}")
            n_fail += 1
        time.sleep(0.1)

    log.info(f"[RCSB] Done. Fetched: {n_ok}  |  Failed: {n_fail}")


# ── Step 1: Build Donor Library ──────────────────────────────────────────────

def build_donor_library(chrom_df: pd.DataFrame) -> dict:
    """Scan RCSB PDBs and extract chromophore HETATM records.

    Returns dict[slug] → {
        chain, hetatm_name, hetatm_atoms: list[dict],
        chrom_res_seqs: [int, int, int],   # 1-indexed PDB residue numbers
        flanking_res_seqs: list[int],       # triad ± FLANK_SIZE
    }
    """
    donors = {}
    pdb_files = list(RCSB_DIR.glob("*.pdb"))
    log.info(f"[Donor Library] Scanning {len(pdb_files)} RCSB PDB files...")

    for pdb_path in pdb_files:
        slug = pdb_path.stem

        # Get chromophore positions for this FP (0-indexed → 1-indexed for PDB)
        row = chrom_df[chrom_df["slug"] == slug]
        if row.empty or row.iloc[0]["status"] != "ok":
            continue
        row = row.iloc[0]
        chrom_pos_0idx = [int(row["chrom_pos0"]), int(row["chrom_pos1"]), int(row["chrom_pos2"])]
        chrom_res_seqs = [p + 1 for p in chrom_pos_0idx]  # 1-indexed

        # Find chain with chromophore
        chain = find_best_chain(pdb_path, CHROM_HETATM_NAMES)
        if chain is None:
            log.debug(f"  [{slug}] No chromophore HETATM found — skipping")
            continue

        # Parse atoms from that chain
        atoms = parse_pdb_atoms(pdb_path, chain_id=chain)
        hetatm = get_hetatm_lines(atoms, CHROM_HETATM_NAMES)
        if not hetatm:
            continue

        hetatm_name = hetatm[0]["res_name"]

        # Flanking residues for alignment
        center = chrom_res_seqs[1]  # middle of triad
        flanking = list(range(center - FLANK_SIZE, center + FLANK_SIZE + 1))
        # Exclude the triad itself from flanking (we align on flanking only)
        flanking_only = [r for r in flanking if r not in chrom_res_seqs]

        # Verify we can extract CA coords for flanking residues
        try:
            get_ca_coords_by_resseq(atoms, flanking_only)
        except ValueError:
            log.debug(f"  [{slug}] Missing flanking CA atoms — skipping")
            continue

        donors[slug] = {
            "chain": chain,
            "hetatm_name": hetatm_name,
            "hetatm_atoms": hetatm,
            "chrom_res_seqs": chrom_res_seqs,
            "flanking_res_seqs": flanking_only,
            "pdb_path": pdb_path,
        }

    log.info(f"[Donor Library] Built library with {len(donors)} donors "
             f"({sum(1 for d in donors.values() if d['hetatm_name'] in {'BLA','BV','BPB','FMN','FAD'})} cofactor)")
    return donors


# ── Step 2: Sequence-Based Donor Matching ────────────────────────────────────

def compute_msa_identity(seq1_aln: str, seq2_aln: str) -> float:
    """Compute pairwise identity from two aligned sequences (MSA columns)."""
    matches = 0
    aligned = 0
    for a, b in zip(seq1_aln, seq2_aln):
        if a != "-" and b != "-":
            aligned += 1
            if a == b:
                matches += 1
    return matches / aligned if aligned > 0 else 0.0


def find_best_donors(donor_slugs: set[str], target_slugs: set[str],
                     aln: dict[str, str]) -> dict[str, tuple[str, float]]:
    """For each target, find the highest-identity donor via MSA.

    Returns dict[target_slug] → (donor_slug, identity)
    """
    donor_list = [s for s in donor_slugs if s in aln]
    matches = {}

    for target in target_slugs:
        if target not in aln:
            continue
        target_aln = aln[target]

        best_donor = None
        best_ident = -1.0
        for donor in donor_list:
            ident = compute_msa_identity(target_aln, aln[donor])
            if ident > best_ident:
                best_ident = ident
                best_donor = donor

        if best_donor is not None:
            matches[target] = (best_donor, best_ident)

    return matches


# ── Step 3: Structural Alignment + Grafting ──────────────────────────────────

def superimpose_and_graft(
    target_pdb: Path,
    donor_info: dict,
    target_chrom_pos_0idx: list[int],
) -> dict:
    """Align donor flanking CAs to target, transform chromophore HETATM coords.

    Returns dict with graft result:
        status, rmsd, n_clashes, grafted_lines: list[str]
    """
    # Target chromophore positions (1-indexed for PDB)
    target_chrom_res_seqs = [p + 1 for p in target_chrom_pos_0idx]
    target_center = target_chrom_res_seqs[1]
    target_flanking = [r for r in range(target_center - FLANK_SIZE, target_center + FLANK_SIZE + 1)
                       if r not in target_chrom_res_seqs]

    # Parse target atoms
    target_atoms = parse_pdb_atoms(target_pdb, chain_id="A")
    if not target_atoms:
        # Try no chain ID (ESMFold sometimes uses empty chain)
        target_atoms = parse_pdb_atoms(target_pdb, chain_id=" ")
    if not target_atoms:
        return {"status": "no_target_atoms", "rmsd": -1, "n_clashes": -1, "grafted_lines": []}

    # Extract flanking CA coords from target
    try:
        target_ca = get_ca_coords_by_resseq(target_atoms, target_flanking)
    except ValueError as e:
        return {"status": f"missing_target_ca: {e}", "rmsd": -1, "n_clashes": -1, "grafted_lines": []}

    # Extract flanking CA coords from donor
    donor_atoms = parse_pdb_atoms(donor_info["pdb_path"], chain_id=donor_info["chain"])
    try:
        donor_ca = get_ca_coords_by_resseq(donor_atoms, donor_info["flanking_res_seqs"])
    except ValueError as e:
        return {"status": f"missing_donor_ca: {e}", "rmsd": -1, "n_clashes": -1, "grafted_lines": []}

    # Ensure same number of atoms for superposition
    n = min(len(target_ca), len(donor_ca))
    if n < 3:
        return {"status": "insufficient_flanking", "rmsd": -1, "n_clashes": -1, "grafted_lines": []}
    target_ca = target_ca[:n]
    donor_ca = donor_ca[:n]

    # Superimpose: rotate donor onto target
    sup = SVDSuperimposer()
    sup.set(target_ca, donor_ca)  # fixed=target, moving=donor
    sup.run()
    rmsd = sup.get_rms()

    # Transform all donor chromophore HETATM coordinates
    rot, tran = sup.get_rotran()
    grafted_lines = []
    grafted_coords = []

    for ha in donor_info["hetatm_atoms"]:
        coord = np.array([ha["x"], ha["y"], ha["z"]])
        new_coord = np.dot(coord, rot) + tran
        grafted_coords.append(new_coord)

        # Rewrite the PDB HETATM line with new coordinates
        old_line = ha["line"]
        new_line = (
            old_line[:30]
            + f"{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}"
            + old_line[54:]
        )
        grafted_lines.append(new_line)

    # Clash check: grafted HETATM atoms vs all target ATOM atoms
    target_all_coords = np.array([[a["x"], a["y"], a["z"]] for a in target_atoms], dtype=np.float64)
    grafted_coords = np.array(grafted_coords, dtype=np.float64)

    n_clashes = 0
    if len(grafted_coords) > 0 and len(target_all_coords) > 0:
        # Pairwise distances
        for gc in grafted_coords:
            dists = np.linalg.norm(target_all_coords - gc, axis=1)
            n_clashes += int(np.sum(dists < CLASH_DIST))

    return {
        "status": "ok",
        "rmsd": rmsd,
        "n_clashes": n_clashes,
        "grafted_lines": grafted_lines,
    }


def write_grafted_pdb(target_pdb: Path, grafted_lines: list[str],
                      out_path: Path, donor_slug: str, seq_ident: float,
                      rmsd: float):
    """Write a grafted PDB: original ATOM records + appended HETATM."""
    with open(target_pdb) as f:
        original_lines = f.readlines()

    with open(out_path, "w") as f:
        # REMARK header
        f.write(f"REMARK  CHROMOPHORE GRAFT\n")
        f.write(f"REMARK  Donor: {donor_slug}\n")
        f.write(f"REMARK  Sequence identity: {seq_ident:.3f}\n")
        f.write(f"REMARK  Local RMSD: {rmsd:.3f} A\n")
        f.write(f"REMARK\n")

        # Original ATOM records
        for line in original_lines:
            if line.startswith(("ATOM  ", "HETATM", "TER", "MODEL", "ENDMDL")):
                f.write(line)

        # Grafted chromophore HETATM
        for line in grafted_lines:
            f.write(line)

        f.write("END\n")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    GRAFTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(META_CSV)
    chrom = pd.read_csv(CHROM_CSV)
    log.info(f"Loaded {len(meta)} FPs, {len(chrom)} chromophore records")

    # ── Step 0: Fetch RCSB donors ─────────────────────────────────────────────
    fetch_rcsb_donors(meta)

    # ── Step 1: Build donor library ───────────────────────────────────────────
    donors = build_donor_library(chrom)
    if not donors:
        log.error("No donors found. Cannot graft.")
        return

    # ── Step 2: Load MSA and find best donors ─────────────────────────────────
    log.info(f"Loading MSA from {ALN_FILE}...")
    aln = {rec.id: str(rec.seq) for rec in SeqIO.parse(ALN_FILE, "fasta")}
    log.info(f"  {len(aln)} sequences in alignment")

    # Identify targets: GFP-family FPs without PDB that have ESMFold predictions
    # Exclude cofactor-dependent FPs (biliverdin, flavin, etc.) — GFP family only
    has_pdb = set(meta[meta["pdb"].notna() & (meta["pdb"].astype(str).str.strip() != "")]["slug"])
    chrom_ok = set(chrom[chrom["status"] == "ok"]["slug"])
    cofactor_slugs = set(chrom[chrom["is_cofactor"] == True]["slug"])
    log.info(f"Excluding {len(cofactor_slugs)} cofactor-dependent FPs (GFP family only)")

    # Collect all predicted structure slugs (SimpleFold primary, ESMFold fallback)
    predicted_slugs = set()
    if SIMPLEFOLD_DIR.exists():
        predicted_slugs = {p.stem.replace("_sampled_0", "") for p in SIMPLEFOLD_DIR.glob("*_sampled_0.pdb")}
        log.info(f"SimpleFold predictions: {len(predicted_slugs)}")
    else:
        log.warning(f"SimpleFold directory not found: {SIMPLEFOLD_DIR}")
    if ESMFOLD_DIR.exists():
        esm_slugs = {p.stem for p in ESMFOLD_DIR.glob("*.pdb")}
        new = esm_slugs - predicted_slugs
        predicted_slugs |= esm_slugs
        log.info(f"ESMFold predictions: {len(esm_slugs)} ({len(new)} unique beyond SimpleFold)")
    else:
        log.warning(f"ESMFold directory not found: {ESMFOLD_DIR}")

    target_slugs = (predicted_slugs - has_pdb) & chrom_ok - cofactor_slugs
    log.info(f"Targets to graft: {len(target_slugs)}")

    donor_matches = find_best_donors(set(donors.keys()), target_slugs, aln)
    log.info(f"Donor matches found: {len(donor_matches)}")

    # ── Step 3: Graft chromophores ────────────────────────────────────────────
    summary_rows = []

    # 3a: Graft targets
    n_grafted, n_failed = 0, 0
    for i, target_slug in enumerate(sorted(target_slugs)):
        if target_slug not in donor_matches:
            summary_rows.append({
                "target_slug": target_slug, "donor_slug": "", "seq_identity": 0,
                "local_rmsd": -1, "n_clashes": -1, "chrom_hetatm_name": "",
                "status": "no_donor_match",
            })
            n_failed += 1
            continue

        donor_slug, seq_ident = donor_matches[target_slug]
        donor_info = donors[donor_slug]

        # Find target PDB (SimpleFold first, ESMFold fallback)
        target_pdb = SIMPLEFOLD_DIR / f"{target_slug}_sampled_0.pdb"
        if not target_pdb.exists():
            target_pdb = ESMFOLD_DIR / f"{target_slug}.pdb"
        if not target_pdb.exists():
            summary_rows.append({
                "target_slug": target_slug, "donor_slug": donor_slug,
                "seq_identity": seq_ident, "local_rmsd": -1, "n_clashes": -1,
                "chrom_hetatm_name": donor_info["hetatm_name"],
                "status": "target_pdb_missing",
            })
            n_failed += 1
            continue

        # Get target chromophore positions
        crow = chrom[chrom["slug"] == target_slug].iloc[0]
        target_chrom_pos = [int(crow["chrom_pos0"]), int(crow["chrom_pos1"]), int(crow["chrom_pos2"])]

        # Graft
        result = superimpose_and_graft(target_pdb, donor_info, target_chrom_pos)

        # Determine status
        status = result["status"]
        if status == "ok":
            if seq_ident < IDENT_WARN:
                status = "low_confidence"
            elif result["rmsd"] > RMSD_WARN:
                status = "high_rmsd"
            elif result["n_clashes"] > CLASH_MAX:
                status = "clashes"

        if result["grafted_lines"]:
            out_path = GRAFTED_DIR / f"{target_slug}_grafted.pdb"
            write_grafted_pdb(target_pdb, result["grafted_lines"], out_path,
                              donor_slug, seq_ident, result["rmsd"])
            n_grafted += 1

        summary_rows.append({
            "target_slug": target_slug,
            "donor_slug": donor_slug,
            "seq_identity": round(seq_ident, 4),
            "local_rmsd": round(result["rmsd"], 3) if result["rmsd"] >= 0 else -1,
            "n_clashes": result["n_clashes"],
            "chrom_hetatm_name": donor_info["hetatm_name"],
            "status": status,
        })

        if (i + 1) % 50 == 0:
            log.info(f"  Grafted {i+1}/{len(target_slugs)}...")

    log.info(f"[Graft] Done. Grafted: {n_grafted}  |  Failed: {n_failed}")

    # 3b: Copy RCSB PDBs directly (they already have chromophores) — GFP family only
    n_copied = 0
    for slug in sorted(has_pdb - cofactor_slugs):
        src = RCSB_DIR / f"{slug}.pdb"
        if src.exists():
            dst = GRAFTED_DIR / f"{slug}_grafted.pdb"
            shutil.copy2(src, dst)
            n_copied += 1
            summary_rows.append({
                "target_slug": slug, "donor_slug": slug,
                "seq_identity": 1.0, "local_rmsd": 0.0, "n_clashes": 0,
                "chrom_hetatm_name": "experimental",
                "status": "rcsb_copy",
            })

    log.info(f"[Copy] Copied {n_copied} RCSB PDB files to grafted/")

    # ── Step 4: Save summary ──────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"Summary saved to {SUMMARY_CSV}")
    log.info(f"  Total in grafted/: {n_grafted + n_copied}")
    log.info(f"  Grafted:           {n_grafted}")
    log.info(f"  RCSB copies:       {n_copied}")
    log.info(f"  Failed:            {n_failed}")
    log.info(f"\nStatus breakdown:")
    for status, count in summary["status"].value_counts().items():
        log.info(f"  {status:25s} {count}")


if __name__ == "__main__":
    main()
