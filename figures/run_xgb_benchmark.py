"""
Run XGBoost benchmark for targets missing from clustered_cv_results.csv
(qy, ext_coeff, pka) under all CV schemes, for both A_onehot and B_lora.

Appends to xgb_extra_results.csv consumed by plot_comparison.py.
"""

import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sequence"
LORA_DIR = ROOT / "model" / "LoRA_ESM2"
BENCH_DIR = ROOT / "benchmark" / "clustered"
OUT_CSV = Path(__file__).resolve().parent / "xgb_extra_results.csv"

RANDOM_SEED = 42
N_FOLDS = 5
TARGETS = ["qy", "ext_coeff", "pka", "brightness"]
CLAMP_RANGES = {
    "qy": (0, 1), "ext_coeff": (0, 300000), "pka": (0, 14),
    "brightness": (0, 200),
}
SCHEMES = ["random", "90", "70", "50"]

# ── Load data ─────────────────────────────────────────────────────────
meta_full = pd.read_csv(DATA_DIR / "fp_embeddings_meta.csv")
meta = meta_full[meta_full["cofactor"].isna()].reset_index(drop=True)
meta["brightness"] = meta["qy"] * meta["ext_coeff"] / 1000.0
slugs = meta["slug"].tolist()

# LoRA embeddings
lora = np.load(LORA_DIR / "lora_embeddings_all_folds.npz")
X_lora = lora["embeddings"][0].astype(np.float32)

# One-hot from alignment
def load_alignment(fasta_path):
    aligned, slug, seq = {}, None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if slug: aligned[slug] = "".join(seq)
                slug, seq = line[1:].split()[0], []
            else: seq.append(line)
    if slug: aligned[slug] = "".join(seq)
    return aligned

def build_onehot(slugs, aligned):
    aln_len = len(next(iter(aligned.values())))
    pairs = []
    for pos in range(aln_len):
        chars = Counter()
        for s in slugs:
            if s in aligned: chars[aligned[s][pos]] += 1
        for ch in chars:
            if ch != "-": pairs.append((pos, ch))
    X = np.zeros((len(slugs), len(pairs)), dtype=np.float32)
    for i, s in enumerate(slugs):
        if s not in aligned: continue
        seq = aligned[s]
        for j, (pos, ch) in enumerate(pairs):
            if seq[pos] == ch: X[i, j] = 1.0
    keep = X.var(axis=0) > 0
    return X[:, keep]

aligned = load_alignment(DATA_DIR / "fp_sequences_aligned.fasta")
X_onehot = build_onehot(slugs, aligned)
print(f"One-hot: {X_onehot.shape}")
print(f"LoRA:    {X_lora.shape}")
print(f"FPs:     {len(slugs)}")

# Cluster groups
cluster_groups = {}
for scheme in ["90", "70", "50"]:
    tsv = BENCH_DIR / f"mmseqs_clusters_{scheme}.tsv"
    if tsv.exists():
        cl = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
        s2c = dict(zip(cl["member"], cl["rep"]))
        cluster_groups[scheme] = np.array([s2c.get(s, s) for s in slugs])

# Subsets
SUBSETS = {
    "A_onehot": X_onehot,
    "B_lora": X_lora,
}

# ── XGBoost CV ────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0, min_child_weight=2,
    random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
)

def run_xgb_cv(X, y, target, scheme):
    valid = ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    lo, hi = CLAMP_RANGES[target]

    if scheme == "random":
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(Xv)
    else:
        gv = cluster_groups[scheme][valid]
        n_folds = min(N_FOLDS, len(np.unique(gv)))
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(Xv, groups=gv)

    all_preds = np.full(len(yv), np.nan)
    for tr, va in split_iter:
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(Xv[tr], yv[tr], verbose=False)
        all_preds[va] = np.clip(m.predict(Xv[va]), lo, hi)

    mask = ~np.isnan(all_preds)
    pooled_mae = mean_absolute_error(yv[mask], all_preds[mask])
    r_val, _ = pearsonr(yv[mask], all_preds[mask])
    return {"pooled_mae": round(pooled_mae, 6), "r": round(r_val, 6), "n_eval": int(mask.sum())}


# ── Run ───────────────────────────────────────────────────────────────
rows = []
t0 = time.time()

for target in TARGETS:
    y = meta[target].values.astype(np.float32)
    n_valid = int((~np.isnan(y)).sum())
    if n_valid < 30:
        continue

    for sub_name, X in SUBSETS.items():
        for scheme in SCHEMES:
            if scheme != "random" and scheme not in cluster_groups:
                continue
            print(f"  {sub_name:12s} | {target:12s} | {scheme:6s}...", end=" ", flush=True)
            res = run_xgb_cv(X, y, target, scheme)
            print(f"MAE={res['pooled_mae']:.4f}  R={res['r']:.3f}")
            rows.append({"target": target, "subset": sub_name, "scheme": scheme, **res})

elapsed = time.time() - t0
print(f"\nDone in {elapsed / 60:.1f} min")

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")
