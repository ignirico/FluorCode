"""
Scatter plot: predicted vs actual em_max for LoRA-ESM2 (MLP) under clustered-50 CV.
Saves to figures/scatter_em_max.png
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sequence"
LORA_DIR = ROOT / "model" / "LoRA_ESM2"
BENCH_DIR = ROOT / "benchmark" / "clustered"
OUT = Path(__file__).resolve().parent

C_TEXT = "#2D2D2D"
C_GRID = "#E5E5E5"
C_MLP = "#D97757"

RANDOM_SEED = 42
N_FOLDS = 5

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Load data ─────────────────────────────────────────────────────────
meta = pd.read_csv(DATA_DIR / "fp_embeddings_meta.csv")
meta = meta[meta["cofactor"].isna()].reset_index(drop=True)
slugs = meta["slug"].tolist()

lora = np.load(LORA_DIR / "lora_embeddings_all_folds.npz")
X_lora = lora["embeddings"][0].astype(np.float32)

# Cluster groups at 50%
tsv = BENCH_DIR / "mmseqs_clusters_50.tsv"
cl = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
s2c = dict(zip(cl["member"], cl["rep"]))
groups = np.array([s2c.get(s, s) for s in slugs])

target = "em_max"
y_all = meta[target].values.astype(np.float32)
valid = ~np.isnan(y_all)
X, y = X_lora[valid], y_all[valid]
g = groups[valid]

print(f"em_max: {valid.sum()} samples, {len(np.unique(g))} clusters")


# ── MLP ───────────────────────────────────────────────────────────────
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


def train_and_predict(X_tr, y_tr, X_va, d_in, fold_seed):
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    y_tr_z = ((y_tr - y_mean) / y_std).astype(np.float32)

    torch.manual_seed(fold_seed)
    model = SimpleMLP(d_in=d_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = nn.SmoothL1Loss()

    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr_z, device=device)
    X_va_t = torch.tensor(X_va, device=device)

    best_mae, best_state, wait = 999.0, None, 0
    n = len(y_tr)

    for epoch in range(300):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = perm[i:i+64]
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
            pred = np.clip(model(X_va_t).cpu().numpy() * y_std + y_mean, 350, 750)
            mae = np.mean(np.abs(pred - np.mean(y_tr)))  # proxy
        if best_state is None:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 30:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.clip(model(X_va_t).cpu().numpy() * y_std + y_mean, 350, 750)


# ── Run clustered-50 CV ──────────────────────────────────────────────
splitter = GroupKFold(n_splits=N_FOLDS)
all_preds = np.full(len(y), np.nan)

for fi, (tr, va) in enumerate(splitter.split(X, groups=g)):
    print(f"  Fold {fi+1}/{N_FOLDS}...", end=" ", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr]).astype(np.float32)
    X_va = sc.transform(X[va]).astype(np.float32)
    pred = train_and_predict(X_tr, y[tr], X_va, d_in=X.shape[1], fold_seed=RANDOM_SEED + fi)
    all_preds[va] = pred
    print(f"MAE={mean_absolute_error(y[va], pred):.2f}")

mask = ~np.isnan(all_preds)
pooled_mae = mean_absolute_error(y[mask], all_preds[mask])
r_val, _ = pearsonr(y[mask], all_preds[mask])
print(f"\nPooled: MAE={pooled_mae:.2f} nm, R={r_val:.3f}")

# ── Random CV ────────────────────────────────────────────────────────
from sklearn.model_selection import KFold
splitter_rand = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
rand_preds = np.full(len(y), np.nan)

print("\nRandom CV:")
for fi, (tr, va) in enumerate(splitter_rand.split(X)):
    print(f"  Fold {fi+1}/{N_FOLDS}...", end=" ", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr]).astype(np.float32)
    X_va = sc.transform(X[va]).astype(np.float32)
    pred = train_and_predict(X_tr, y[tr], X_va, d_in=X.shape[1], fold_seed=RANDOM_SEED + fi)
    rand_preds[va] = pred
    print(f"MAE={mean_absolute_error(y[va], pred):.2f}")

rand_mask = ~np.isnan(rand_preds)
rand_mae = mean_absolute_error(y[rand_mask], rand_preds[rand_mask])
rand_r, _ = pearsonr(y[rand_mask], rand_preds[rand_mask])
print(f"\nRandom Pooled: MAE={rand_mae:.2f} nm, R={rand_r:.3f}")


# ── Scatter plots: side by side ──────────────────────────────────────
def plot_scatter(ax, y_true, y_pred, title, mae_val, r_val):
    lo, hi = 370, 730
    ax.scatter(y_true, y_pred, c=C_MLP, s=28, alpha=0.55,
               edgecolor=C_TEXT, linewidth=0.15, zorder=3)
    ax.plot([lo, hi], [lo, hi], color="#A89B91", lw=1.5, ls="--", zorder=1,
            label="y = x")
    ax.fill_between([lo, hi], [lo - mae_val, hi - mae_val],
                    [lo + mae_val, hi + mae_val],
                    color=C_MLP, alpha=0.08, zorder=0,
                    label=f"\u00b1MAE ({mae_val:.1f} nm)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Emission (nm)", fontsize=14, color=C_TEXT)
    ax.set_ylabel("Predicted Emission (nm)", fontsize=14, color=C_TEXT)
    ax.set_title(f"{title}\nMAE = {mae_val:.1f} nm    R = {r_val:.2f}",
                 fontsize=15, fontweight="bold", color=C_TEXT, pad=12)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12, colors=C_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=11, frameon=False, loc="upper left")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

plot_scatter(ax1, y[rand_mask], rand_preds[rand_mask],
             "(a) Random CV", rand_mae, rand_r)
plot_scatter(ax2, y[mask], all_preds[mask],
             "(b) Clustered CV (50% Identity)", pooled_mae, r_val)

fig.suptitle("LoRA-ESM2 (MLP) — Emission Wavelength Prediction",
             fontsize=17, fontweight="bold", color=C_TEXT, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "scatter_em_max.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved scatter_em_max.png")
