"""
Structure Collection Pipeline
==============================
Step 1 — Fetch experimental PDB structures from RCSB (priority).
Step 2 — Fold remaining sequences with Apple SimpleFold 100M + MLX.

All structures saved to: data/structures/<slug>.pdb

Usage:
    python3 fold_simplefold.py
"""

import os
import sys
import time
import shutil
import signal
import logging
import threading
import requests
import subprocess
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
STRUCT_DIR = DATA_DIR / "structures"
META_CSV   = DATA_DIR / "fp_cleaned.csv"
LOG_FILE   = DATA_DIR / "structure_collection.log"

STRUCT_DIR.mkdir(parents=True, exist_ok=True)

# SimpleFold must be run from this directory for its internal imports to work
# SimpleFold needs its own venv (conflicts with ESM3 used for embeddings)
SF_VENV     = Path("/Users/ekko/Desktop/FluorCode/ml-simplefold/.venv/bin/python3")
SF_WORKDIR  = Path("/Users/ekko/Desktop/FluorCode/ml-simplefold/src/simplefold")
SF_CKPT_DIR = ROOT / "artifacts"  # Valid checkpoints (6.3GB for 1.6B)
SF_MODEL    = "simplefold_100M"
SF_BACKEND  = "mlx"
SF_STEPS    = 500
SF_TAU      = 0.01

# Memory watchdog: kill SimpleFold if available system RAM drops below this
# Set low — macOS handles memory pressure well via swap; the loading spike
# temporarily reports low RAM before purgeable pages are reclaimed.
RAM_FLOOR_GB = 0.5
RAM_CHECK_INTERVAL = 10  # seconds

RCSB_URL   = "https://files.rcsb.org/download/{pdb_id}.pdb"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Memory helpers (macOS) ────────────────────────────────────────────────────

def get_available_memory_gb() -> float:
    """Available RAM in GB (free + inactive + purgeable pages) via vm_stat."""
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True)
        page_size = 16384  # Apple Silicon default
        free = inactive = purgeable = 0
        for line in r.stdout.splitlines():
            if "page size" in line:
                page_size = int("".join(c for c in line.split()[-2] if c.isdigit()))
            elif "Pages free" in line:
                free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive" in line:
                inactive = int(line.split()[-1].rstrip("."))
            elif "Pages purgeable" in line:
                purgeable = int(line.split()[-1].rstrip("."))
        return (free + inactive + purgeable) * page_size / (1024 ** 3)
    except Exception:
        return 99.0  # Fallback: assume plenty


def fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


# ── Progress bar ──────────────────────────────────────────────────────────────

def progress_bar(done: int, total: int, width: int = 30) -> str:
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {done}/{total} ({frac * 100:5.1f}%)"


# ── Step 1: RCSB fetch ────────────────────────────────────────────────────────

def fetch_rcsb(meta: pd.DataFrame):
    has_pdb  = meta["pdb"].notna() & (meta["pdb"].astype(str).str.strip() != "")
    meta_pdb = meta[has_pdb].copy()
    meta_pdb["pdb_id"] = meta_pdb["pdb"].astype(str).str.split("|").str[0].str.strip()

    done = {p.stem for p in STRUCT_DIR.glob("*.pdb")}
    todo = meta_pdb[~meta_pdb["slug"].isin(done)]

    log.info(f"[RCSB] FPs with PDB ID: {len(meta_pdb)}  |  Already fetched: {len(done & set(meta_pdb['slug']))}  |  To fetch: {len(todo)}")

    n_ok, n_fail = 0, 0
    for _, row in todo.iterrows():
        slug, pdb_id = row["slug"], row["pdb_id"].upper()
        try:
            resp = requests.get(RCSB_URL.format(pdb_id=pdb_id), timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                (STRUCT_DIR / f"{slug}.pdb").write_text(resp.text)
                n_ok += 1
            else:
                log.warning(f"  SKIP [{slug}] {pdb_id} — HTTP {resp.status_code}")
                n_fail += 1
        except Exception as e:
            log.warning(f"  ERROR [{slug}] {pdb_id} — {e}")
            n_fail += 1
        time.sleep(0.1)

    log.info(f"[RCSB] Done. Fetched: {n_ok}  |  Failed: {n_fail}")


# ── Step 2: SimpleFold ────────────────────────────────────────────────────────

def fold_simplefold(meta: pd.DataFrame):
    if not SF_WORKDIR.exists():
        log.error(f"SimpleFold not found at {SF_WORKDIR}")
        sys.exit(1)

    done = {p.stem for p in STRUCT_DIR.glob("*.pdb")}
    todo = meta[~meta["slug"].isin(done)].reset_index(drop=True)
    todo = todo.dropna(subset=["seq"])
    todo = todo[todo["seq"].str.len() <= 1024].reset_index(drop=True)
    n_total = len(todo)

    if n_total == 0:
        log.info("[SimpleFold] Nothing to fold — all sequences already have structures.")
        return

    log.info(f"[SimpleFold] Sequences to fold: {n_total}")

    # Write one FASTA file per protein (SimpleFold treats a multi-seq FASTA
    # as multiple chains of ONE structure — we need separate files).
    fasta_dir = DATA_DIR / "simplefold_fastas"
    fasta_dir.mkdir(exist_ok=True)
    out_dir = DATA_DIR / "simplefold_output"
    out_dir.mkdir(exist_ok=True)
    pred_dir = out_dir / f"predictions_{SF_MODEL}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for _, row in todo.iterrows():
        fp = fasta_dir / f"{row['slug']}.fasta"
        if not fp.exists():
            fp.write_text(f">{row['slug']}\n{row['seq']}\n")
    log.info(f"[SimpleFold] Prepared {n_total} individual FASTA files in {fasta_dir}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SF_WORKDIR.parent) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        str(SF_VENV),
        "-c", "from simplefold.cli import main; main()",
        "--simplefold_model",    SF_MODEL,
        "--ckpt_dir",            str(SF_CKPT_DIR),
        "--num_steps",           str(SF_STEPS),
        "--tau",                 str(SF_TAU),
        "--nsample_per_protein", "1",
        "--output_format",       "pdb",
        "--fasta_path",          str(fasta_dir),
        "--output_dir",          str(out_dir),
        "--backend",             SF_BACKEND,
    ]

    log.info(f"[SimpleFold] Launching: {SF_MODEL} / {SF_BACKEND} / {SF_STEPS} steps / tau={SF_TAU}")
    log.info(f"[SimpleFold] RAM floor: {RAM_FLOOR_GB:.1f} GB — will auto-kill if available RAM drops below")

    # Count PDBs that existed before we start (from prior partial runs)
    pre_existing = set(p.name for p in pred_dir.glob("*.pdb"))

    # ── Launch subprocess ─────────────────────────────────────────────────
    proc = subprocess.Popen(
        cmd, text=True, cwd=str(SF_WORKDIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    killed_by_watchdog = False
    kill_reason = ""

    # ── Stderr reader thread (prevents pipe deadlock) ─────────────────────
    stderr_lines = []
    def drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    # ── Stdout reader thread ──────────────────────────────────────────────
    stdout_lines = []
    def drain_stdout():
        for line in proc.stdout:
            stdout_lines.append(line.rstrip())
    stdout_thread = threading.Thread(target=drain_stdout, daemon=True)
    stdout_thread.start()

    # ── Watchdog + progress monitor ───────────────────────────────────────
    t_start = time.time()
    first_pdb_time = None
    model_loaded = False

    print()  # blank line before progress display
    try:
        while proc.poll() is None:
            time.sleep(RAM_CHECK_INTERVAL)

            # ── Check memory ──────────────────────────────────────────
            avail_gb = get_available_memory_gb()
            if avail_gb < RAM_FLOOR_GB:
                killed_by_watchdog = True
                kill_reason = f"Available RAM {avail_gb:.1f} GB < floor {RAM_FLOOR_GB:.1f} GB"
                log.error(f"[WATCHDOG] {kill_reason} — killing SimpleFold to protect your system")
                proc.kill()
                break

            # ── Count new PDBs ────────────────────────────────────────
            current_pdbs = set(p.name for p in pred_dir.glob("*.pdb"))
            new_pdbs = current_pdbs - pre_existing
            n_done = len(new_pdbs)
            elapsed = time.time() - t_start

            # Detect model loaded (first stdout activity after startup)
            if not model_loaded and any("loaded" in l.lower() for l in stdout_lines):
                model_loaded = True
                log.info(f"[SimpleFold] Model loaded in {fmt_time(elapsed)}")

            # Track timing from first PDB
            if n_done > 0 and first_pdb_time is None:
                first_pdb_time = time.time()

            # ── Build status line ─────────────────────────────────────
            bar = progress_bar(n_done, n_total)

            if n_done > 0 and first_pdb_time is not None:
                fold_elapsed = time.time() - first_pdb_time
                rate = fold_elapsed / n_done  # sec per sequence
                remaining = (n_total - n_done) * rate
                eta_str = fmt_time(remaining)
                rate_str = f"{rate:.1f}s/seq"
            elif model_loaded:
                eta_str = "calculating..."
                rate_str = "—"
            else:
                eta_str = "loading model..."
                rate_str = "—"

            status = (
                f"\r  [SimpleFold 100M]  {bar}  |  "
                f"{rate_str}  |  ETA: {eta_str}  |  "
                f"RAM avail: {avail_gb:.1f} GB  |  "
                f"elapsed: {fmt_time(elapsed)}"
            )
            print(status, end="", flush=True)

    except KeyboardInterrupt:
        log.warning("[SimpleFold] Interrupted by user — killing subprocess")
        proc.kill()
        print()
        return

    print()  # newline after progress bar

    # Wait for IO threads
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    # ── Log subprocess output ─────────────────────────────────────────────
    if stdout_lines:
        for line in stdout_lines:
            log.info(f"  [SF] {line}")
    if stderr_lines:
        for line in stderr_lines:
            log.warning(f"  [SF stderr] {line}")

    if killed_by_watchdog:
        log.error(f"[SimpleFold] KILLED by watchdog: {kill_reason}")
        log.info(f"[SimpleFold] Partial results preserved — re-run to continue from where it stopped")
    elif proc.returncode != 0:
        log.error(f"[SimpleFold] Failed (exit {proc.returncode})")
        return
    else:
        elapsed = time.time() - t_start
        log.info(f"[SimpleFold] Completed in {fmt_time(elapsed)}")

    # ── Move output PDBs to STRUCT_DIR ────────────────────────────────────
    pdbs = list(pred_dir.glob("*.pdb"))
    log.info(f"[SimpleFold] Output PDBs found: {len(pdbs)}")

    n_ok, n_fail = 0, 0
    slug_set = set(todo["slug"].tolist())
    for pdb in pdbs:
        slug = pdb.stem.replace("_sampled_0", "")
        if slug in slug_set:
            shutil.copy(pdb, STRUCT_DIR / f"{slug}.pdb")
            n_ok += 1
        else:
            log.warning(f"  Unrecognised output: {pdb.name}")
            n_fail += 1

    log.info(f"[SimpleFold] Copied: {n_ok}  |  Unmatched: {n_fail}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    meta = pd.read_csv(META_CSV)
    log.info(f"Loaded {len(meta)} FPs from {META_CSV.name}")

    log.info("=" * 60)
    log.info("  STEP 1 — Fetching RCSB experimental structures")
    log.info("=" * 60)
    fetch_rcsb(meta)

    log.info("=" * 60)
    log.info("  STEP 2 — Folding remaining with SimpleFold 100M + MLX")
    log.info("=" * 60)
    fold_simplefold(meta)

    total = len(list(STRUCT_DIR.glob("*.pdb")))
    log.info(f"\nAll done. Total structures in {STRUCT_DIR}: {total} / {len(meta)}")


if __name__ == "__main__":
    main()
