"""Fetch FP records from the FPbase API, align with MAFFT, filter with trimAl.

Follows the FPredX (Tam & Zhang 2022) preprocessing recipe with their thresholds:
residue overlap >= 90% and sequence overlap >= 90% in trimAl.
"""

import os
import json
import requests
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

FPBASE_PROTEINS_URL = "https://www.fpbase.org/api/proteins/"
FPBASE_BASIC_URL    = "https://www.fpbase.org/api/proteins/basic/"

RAW_JSON        = OUT_DIR / "fpbase_raw.json"
RAW_BASIC_JSON  = OUT_DIR / "fpbase_basic_raw.json"
RAW_FASTA       = OUT_DIR / "fp_sequences_raw.fasta"
ALIGNED_FASTA   = OUT_DIR / "fp_sequences_aligned.fasta"
TRIMMED_FASTA   = OUT_DIR / "fp_sequences_trimmed.fasta"
CLEANED_CSV     = OUT_DIR / "fp_cleaned.csv"
LOG_FILE        = OUT_DIR / "pipeline.log"

RESIDUE_OVERLAP  = 0.9   # trimAl -resoverlap
SEQUENCE_OVERLAP = 0.9   # trimAl -seqoverlap (as percentage: 90)

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; FPbase-fetcher/1.0)",
}

def fetch_json(url: str, cache_path: Path) -> list | dict:
    if cache_path.exists():
        log(f"[cache] Loading {cache_path}")
        with open(cache_path) as f:
            return json.load(f)
    log(f"[fetch] GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"[fetch] Saved {len(data)} records → {cache_path}")
    return data


# ── Step 1: Fetch ─────────────────────────────────────────────────────────────

def fetch_all():
    log("\n=== Step 1: Fetch FPbase data ===")

    # Full protein records (includes seq, states, switch_type, etc.)
    proteins = fetch_json(FPBASE_PROTEINS_URL, RAW_JSON)

    # Basic flat properties (one row per non-switchable FP, default state assumed)
    basic    = fetch_json(FPBASE_BASIC_URL, RAW_BASIC_JSON)

    log(f"Full records   : {len(proteins)}")
    log(f"Basic records  : {len(basic)}")
    return proteins, basic


# ── Step 2: Parse → DataFrame + FASTA ────────────────────────────────────────

def parse_proteins(proteins: list, basic: list) -> tuple[pd.DataFrame, int]:
    log("\n=== Step 2: Parse records ===")

    # Build basic lookup keyed by slug for joining properties
    basic_lookup = {b["slug"]: b for b in basic}

    rows = []
    fasta_lines = []
    skipped_no_seq = 0

    for p in proteins:
        slug = p.get("slug", "")
        seq  = p.get("seq", "")

        if not seq:
            skipped_no_seq += 1
            continue

        # Flatten states: take first / default state properties
        states = p.get("states", [])
        state  = next((s for s in states if "default" in s.get("name","").lower()), states[0] if states else {})

        # Merge with basic flat record if available
        b = basic_lookup.get(slug, {})

        row = {
            "uuid"        : p.get("uuid"),
            "name"        : p.get("name"),
            "slug"        : slug,
            "seq"         : seq,
            "seq_len"     : len(seq),
            "ipg_id"      : p.get("ipg_id"),
            "genbank"     : p.get("genbank"),
            "uniprot"     : p.get("uniprot"),
            "pdb"         : "|".join(p.get("pdb", [])) if p.get("pdb") else None,
            "agg"         : p.get("agg") or b.get("agg"),
            "switch_type" : p.get("switch_type"),
            "cofactor"    : b.get("cofactor", ""),
            "ex_max"      : state.get("ex_max") or b.get("ex_max"),
            "em_max"      : state.get("em_max") or b.get("em_max"),
            "ext_coeff"   : state.get("ext_coeff") or b.get("ext_coeff"),
            "qy"          : state.get("qy") or b.get("qy"),
            "brightness"  : state.get("brightness") or b.get("brightness"),
            "pka"         : state.get("pka") or b.get("pka"),
            "maturation"  : state.get("maturation") or b.get("maturation"),
            "lifetime"    : state.get("lifetime") or b.get("lifetime"),
            "bleach"      : b.get("bleach"),
            "stokes"      : b.get("stokes"),
            "doi"         : p.get("doi"),
            "n_states"    : len(states),
        }
        rows.append(row)

        # FASTA: use slug as identifier (no spaces, unique)
        fasta_lines.append(f">{slug}")
        fasta_lines.append(seq)

    df = pd.DataFrame(rows)
    log(f"Records with sequence : {len(df)}")
    log(f"Records without seq   : {skipped_no_seq}")
    log(f"Unique sequences      : {df['seq'].nunique()}")

    # Deduplicate by sequence (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset="seq", keep="first")
    log(f"After dedup by seq    : {len(df)}  (removed {before - len(df)} duplicates)")

    # Write FASTA
    with open(RAW_FASTA, "w") as f:
        # Rebuild from deduplicated df
        for _, row in df.iterrows():
            f.write(f">{row['slug']}\n{row['seq']}\n")
    log(f"Raw FASTA written     : {RAW_FASTA}  ({len(df)} sequences)")

    return df, skipped_no_seq


# ── Step 3: MAFFT alignment ───────────────────────────────────────────────────

def run_mafft():
    log("\n=== Step 3: MAFFT alignment (default settings) ===")

    if ALIGNED_FASTA.exists():
        log(f"[cache] {ALIGNED_FASTA} already exists, skipping MAFFT.")
        return

    cmd = ["mafft", "--auto", "--quiet", str(RAW_FASTA)]
    log(f"Running: {' '.join(cmd)}")

    with open(ALIGNED_FASTA, "w") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{result.stderr}")

    # Count aligned sequences
    n_aligned = sum(1 for l in open(ALIGNED_FASTA) if l.startswith(">"))
    log(f"Aligned sequences     : {n_aligned}  → {ALIGNED_FASTA}")


# ── Step 4: trimAl filtering ──────────────────────────────────────────────────

def run_trimal():
    log("\n=== Step 4: trimAl filtering ===")
    log(f"  residue overlap  >= {RESIDUE_OVERLAP*100:.0f}%")
    log(f"  sequence overlap >= {SEQUENCE_OVERLAP*100:.0f}%")

    if TRIMMED_FASTA.exists():
        log(f"[cache] {TRIMMED_FASTA} already exists, skipping trimAl.")
        return

    cmd = [
        "trimal",
        "-in",  str(ALIGNED_FASTA),
        "-out", str(TRIMMED_FASTA),
        "-resoverlap", str(RESIDUE_OVERLAP),
        "-seqoverlap", str(int(SEQUENCE_OVERLAP * 100)),
    ]
    log(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trimAl failed:\n{result.stderr}\n{result.stdout}")

    n_trimmed = sum(1 for l in open(TRIMMED_FASTA) if l.startswith(">"))
    log(f"After trimAl          : {n_trimmed}  → {TRIMMED_FASTA}")
    log(result.stdout.strip() if result.stdout.strip() else "")


# ── Step 5: Merge trimmed sequences back with properties ─────────────────────

def merge_and_save(df: pd.DataFrame):
    log("\n=== Step 5: Merge filtered sequences with properties ===")

    # Read surviving slugs from trimmed FASTA
    kept_slugs = set()
    with open(TRIMMED_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                kept_slugs.add(line[1:].strip())

    log(f"Sequences kept by trimAl: {len(kept_slugs)}")

    # Filter DataFrame
    before = len(df)
    df_clean = df[df["slug"].isin(kept_slugs)].copy()
    log(f"Rows before filter    : {before}")
    log(f"Rows after filter     : {len(df_clean)}")

    # Save
    df_clean.to_csv(CLEANED_CSV, index=False)
    log(f"Cleaned CSV saved     : {CLEANED_CSV}")

    # Per-property summary (non-null counts)
    props = ["ex_max","em_max","ext_coeff","qy","brightness","pka","maturation","lifetime","bleach","agg","cofactor"]
    log("\nProperty coverage in cleaned dataset:")
    for col in props:
        if col in df_clean.columns:
            n = df_clean[col].replace("", pd.NA).notna().sum()
            log(f"  {col:<15}: {n}/{len(df_clean)} ({100*n/len(df_clean):.1f}%)")

    return df_clean


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Clear log
    LOG_FILE.unlink(missing_ok=True)
    log("FPbase Data Pipeline")
    log("=" * 60)

    proteins, basic = fetch_all()
    df, _ = parse_proteins(proteins, basic)

    try:
        run_mafft()
    except (RuntimeError, FileNotFoundError) as e:
        log(f"[ERROR] MAFFT step failed: {e}")
        log("Install MAFFT: conda install -c bioconda mafft")
        sys.exit(1)

    try:
        run_trimal()
    except (RuntimeError, FileNotFoundError) as e:
        log(f"[ERROR] trimAl step failed: {e}")
        log("Install trimAl: conda install -c bioconda trimal")
        sys.exit(1)

    df_clean = merge_and_save(df)

    log("\n=== Pipeline complete ===")
    log(f"Final dataset: {len(df_clean)} FP sequences with properties")
    log(f"Output: {CLEANED_CSV}")
