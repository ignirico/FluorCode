"""
Run MLP benchmark across all CV schemes (random, 90, 70, 50) for all targets.
Produces mlp_benchmark_results.csv consumed by plot_comparison.py.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sequence"
LORA_DIR = ROOT / "model" / "LoRA_ESM2"
BENCH_DIR = ROOT / "benchmark" / "clustered"
OUT_CSV = Path(__file__).resolve().parent / "mlp_benchmark_results.csv"

RANDOM_SEED = 42
N_FOLDS = 5
STRUCT_DIR = ROOT / "model" / "LoRA_ESM2_Structure"
TARGETS = ["ex_max", "em_max", "qy", "ext_coeff", "pka", "brightness"]
CLAMP_RANGES = {
    "ex_max": (300, 700), "em_max": (350, 750),
    "qy": (0, 1), "ext_coeff": (0, 300000), "pka": (0, 14),
    "brightness": (0, 200),
}
SCHEMES = ["random", "90", "70", "50"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────
meta_full = pd.read_csv(DATA_DIR / "fp_embeddings_meta.csv")
meta = meta_full[meta_full["cofactor"].isna()].reset_index(drop=True)
meta["brightness"] = meta["qy"] * meta["ext_coeff"] / 1000.0
slugs = meta["slug"].tolist()
print(f"FPs: {len(slugs)}")

lora = np.load(LORA_DIR / "lora_embeddings_all_folds.npz")
X_lora = lora["embeddings"][0].astype(np.float32)
print(f"LoRA: {X_lora.shape}")

# Pocket3D structural features
p3d_path = STRUCT_DIR / "pocket3d_features.npz"
X_struct = None
if p3d_path.exists():
    p3d = np.load(p3d_path, allow_pickle=True)
    p3d_names = list(p3d["feature_names"])
    slug2p3d = {s: p3d["features"][i] for i, s in enumerate(p3d["slugs"])}
    slug2hp = {s: int(p3d["has_pocket"][i]) for i, s in enumerate(p3d["slugs"])}
    D_p = p3d["features"].shape[1]

    TOP_FEATURES = [
        "A__tau_cos", "A__tau_sin", "A__inter_ring_dihedral_cos",
        "A__phenol_planarity_rmsd", "A__imid_planarity_rmsd",
        "B__oh_nearest_his_nitrogen", "B__oh_electrostatic_proxy_8A",
        "C__imid_electrostatic_proxy_8A",
    ]
    top_idx = [p3d_names.index(f) for f in TOP_FEATURES]
    X_p3d_full = np.stack(
        [slug2p3d.get(s, np.zeros(D_p, dtype=np.float32)) for s in slugs]
    ).astype(np.float32)
    has_p3d = np.array([slug2hp.get(s, 0) for s in slugs], dtype=np.float32)
    X_p3d_full[~np.isfinite(X_p3d_full)] = 0.0
    X_struct = np.hstack([X_p3d_full[:, top_idx], has_p3d[:, None]]).astype(np.float32)
    X_fused = np.hstack([X_lora, X_struct])
    print(f"Struct: {X_struct.shape}  ->  Fused: {X_fused.shape}")
else:
    print("Struct: pocket3d_features.npz not found, skipping MLP+Struct")

# Cluster groups
cluster_groups = {}
for scheme in ["90", "70", "50"]:
    tsv = BENCH_DIR / f"mmseqs_clusters_{scheme}.tsv"
    if tsv.exists():
        cl = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
        s2c = dict(zip(cl["member"], cl["rep"]))
        cluster_groups[scheme] = np.array([s2c.get(s, s) for s in slugs])
        print(f"Clusters@{scheme}%: {len(np.unique(cluster_groups[scheme]))}")


# ── Model ─────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, d_in, hidden1=512, hidden2=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden1), nn.BatchNorm1d(hidden1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_eval(model, X_tr, y_tr, X_va, y_va, target,
                   lr=1e-3, wd=1e-2, epochs=300, patience=30, batch_size=64):
    lo, hi = CLAMP_RANGES[target]
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    y_tr_z = ((y_tr - y_mean) / y_std).astype(np.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr_z, device=device)
    X_va_t = torch.tensor(X_va, device=device)

    best_mae, best_state, wait = 999.0, None, 0
    n = len(y_tr)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_z = model(X_va_t).cpu().numpy()
            pred = np.clip(pred_z * y_std + y_mean, lo, hi)
            mae = mean_absolute_error(y_va, pred)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = np.clip(model(X_va_t).cpu().numpy() * y_std + y_mean, lo, hi)
    return best_mae, final_pred


def run_cv(X, y, target, scheme, d_in):
    valid = ~np.isnan(y)
    Xv, yv = X[valid], y[valid]

    if scheme == "random":
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(Xv)
    else:
        gv = cluster_groups[scheme][valid]
        n_unique = len(np.unique(gv))
        n_folds = min(N_FOLDS, n_unique)
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(Xv, groups=gv)

    all_preds = np.full(len(yv), np.nan)
    fold_maes = []

    for fi, (tr, va) in enumerate(split_iter):
        sc = StandardScaler()
        X_tr = sc.fit_transform(Xv[tr]).astype(np.float32)
        X_va = sc.transform(Xv[va]).astype(np.float32)

        torch.manual_seed(RANDOM_SEED + fi)
        model = SimpleMLP(d_in=d_in).to(device)
        mae, pred = train_and_eval(model, X_tr, yv[tr], X_va, yv[va], target)
        fold_maes.append(mae)
        all_preds[va] = pred

    mask = ~np.isnan(all_preds)
    pooled_mae = mean_absolute_error(yv[mask], all_preds[mask])
    r_val, _ = pearsonr(yv[mask], all_preds[mask])

    return {
        "pooled_mae": round(pooled_mae, 6),
        "r": round(r_val, 6),
        "mean_fold": round(float(np.mean(fold_maes)), 6),
        "n_eval": int(mask.sum()),
    }


# ── Run all benchmarks ────────────────────────────────────────────────
rows = []
t0 = time.time()

for target in TARGETS:
    y = meta[target].values.astype(np.float32)
    n_valid = int((~np.isnan(y)).sum())
    if n_valid < 30:
        print(f"  Skipping {target} (only {n_valid} samples)")
        continue

    for scheme in SCHEMES:
        if scheme != "random" and scheme not in cluster_groups:
            continue

        print(f"  MLP_lora | {target:12s} | {scheme:6s}...", end=" ", flush=True)
        res = run_cv(X_lora, y, target, scheme, d_in=X_lora.shape[1])
        print(f"MAE={res['pooled_mae']:.4f}  R={res['r']:.3f}  (n={res['n_eval']})")
        rows.append({
            "target": target, "subset": "MLP_lora", "scheme": scheme,
            "n_eval": res["n_eval"], **res,
        })

        # MLP+Struct
        if X_struct is not None:
            print(f"  MLP_struct | {target:12s} | {scheme:6s}...", end=" ", flush=True)
            res2 = run_cv(X_fused, y, target, scheme, d_in=X_fused.shape[1])
            print(f"MAE={res2['pooled_mae']:.4f}  R={res2['r']:.3f}  (n={res2['n_eval']})")
            rows.append({
                "target": target, "subset": "MLP_lora+struct", "scheme": scheme,
                "n_eval": res2["n_eval"], **res2,
            })

elapsed = time.time() - t0
print(f"\nDone in {elapsed / 60:.1f} min")

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")
print(df_out.to_string(index=False))
