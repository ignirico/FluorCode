"""
Chromophore Identification via MSA
===================================
Uses the existing MAFFT alignment to identify chromophore-forming residues
for each FP by mapping from avGFP's known chromophore (S65-Y66-G67).

Strategy:
  1. Load MAFFT alignment (990 sequences)
  2. Find avGFP and locate its chromophore columns (S65-Y66-G67, 1-indexed GFP numbering)
  3. For each FP, map those alignment columns back to ungapped sequence positions
  4. Flag cofactor-dependent FPs (biliverdin etc.) separately

Usage:
    python3 identify_chromophore.py

Output:
    data/chromophore_positions.csv  — slug, chrom_pos0, chrom_pos1, chrom_pos2, chrom_residues, is_cofactor
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
ALN_FILE = DATA_DIR / "fp_sequences_aligned.fasta"
META_CSV = DATA_DIR / "fp_cleaned.csv"
OUT_CSV  = DATA_DIR / "chromophore_positions.csv"

# avGFP chromophore: S65-Y66-G67 (1-indexed GFP numbering)
# In the raw avGFP sequence (MSKGEELFTG...), these are 0-indexed positions 64, 65, 66
# The tripeptide is SYG in ...F_SYG_VQC...
AVGFP_SLUG       = "avgfp"
AVGFP_CHROM_POS  = [64, 65, 66]  # 0-indexed in raw sequence

# Cofactor-dependent FPs use an external chromophore (e.g. biliverdin),
# not an autocatalytically formed tripeptide
COFACTOR_TYPES = {"bv", "fl", "br", "rl", "pc"}


def alignment_col_to_seq_pos(aligned_seq: str, target_cols: list[int]) -> list[int]:
    """Map alignment column indices to ungapped sequence positions.

    Returns -1 for columns where this sequence has a gap.
    """
    seq_pos = -1
    col_to_pos = {}
    for col_idx, char in enumerate(aligned_seq):
        if char != "-":
            seq_pos += 1
        col_to_pos[col_idx] = seq_pos if char != "-" else -1

    return [col_to_pos.get(c, -1) for c in target_cols]


def seq_pos_to_alignment_col(aligned_seq: str, target_positions: list[int]) -> list[int]:
    """Map ungapped sequence positions to alignment column indices."""
    pos_to_col = {}
    seq_pos = -1
    for col_idx, char in enumerate(aligned_seq):
        if char != "-":
            seq_pos += 1
            pos_to_col[seq_pos] = col_idx
    return [pos_to_col.get(p, -1) for p in target_positions]


def main():
    # Load metadata
    meta = pd.read_csv(META_CSV)
    cofactor_set = set(
        meta.loc[meta["cofactor"].isin(COFACTOR_TYPES), "slug"].tolist()
    )
    print(f"Cofactor-dependent FPs: {len(cofactor_set)}")

    # Load alignment
    print(f"Loading alignment from {ALN_FILE}...")
    aln = {rec.id: str(rec.seq) for rec in SeqIO.parse(ALN_FILE, "fasta")}
    print(f"  {len(aln)} sequences in alignment")

    # Find avGFP and its chromophore alignment columns
    if AVGFP_SLUG not in aln:
        raise ValueError(f"avGFP (slug '{AVGFP_SLUG}') not found in alignment")

    avgfp_aln = aln[AVGFP_SLUG]
    chrom_cols = seq_pos_to_alignment_col(avgfp_aln, AVGFP_CHROM_POS)
    print(f"\navGFP chromophore (0-indexed seq pos {AVGFP_CHROM_POS}):")
    print(f"  Alignment columns: {chrom_cols}")

    # Validate: extract avGFP residues at those columns
    avgfp_residues = "".join(avgfp_aln[c] for c in chrom_cols)
    print(f"  Residues at those columns: {avgfp_residues}")
    # Should be SYG (or TYG depending on variant)

    # Map chromophore columns to each FP
    rows = []
    n_found, n_gap, n_missing = 0, 0, 0
    for slug in meta["slug"]:
        is_cofactor = slug in cofactor_set

        if slug not in aln:
            rows.append({
                "slug": slug,
                "chrom_pos0": -1, "chrom_pos1": -1, "chrom_pos2": -1,
                "chrom_residues": "",
                "is_cofactor": is_cofactor,
                "status": "not_in_alignment",
            })
            n_missing += 1
            continue

        aligned_seq = aln[slug]
        positions = alignment_col_to_seq_pos(aligned_seq, chrom_cols)
        residues = "".join(
            aligned_seq[c] if c < len(aligned_seq) else "-"
            for c in chrom_cols
        )

        if -1 in positions:
            status = "gap_at_chromophore"
            n_gap += 1
        else:
            status = "ok"
            n_found += 1

        rows.append({
            "slug": slug,
            "chrom_pos0": positions[0],
            "chrom_pos1": positions[1],
            "chrom_pos2": positions[2],
            "chrom_residues": residues,
            "is_cofactor": is_cofactor,
            "status": status,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nResults:")
    print(f"  Chromophore found: {n_found}")
    print(f"  Gap at chromophore: {n_gap}")
    print(f"  Not in alignment: {n_missing}")
    print(f"  Cofactor-dependent: {len(cofactor_set)}")

    # Show chromophore residue distribution
    ok = df[df["status"] == "ok"]
    print(f"\nChromophore tripeptide distribution (top 15):")
    print(ok["chrom_residues"].value_counts().head(15).to_string())

    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
