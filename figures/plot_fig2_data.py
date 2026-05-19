"""
Figure 2: FPbase dataset statistics for ICML AI4Science workshop paper.

Layout (2 rows, 3 cols):
  Top:    (a) Property correlation heatmap  |  (b) FP landscape: ex vs em  |  (c) Pipeline attrition
  Bottom: (d) ex_max KDE by stage  |  (e) em_max KDE by stage  |  (f) QY KDE by stage
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "sequence"
STRUCT_DIR = ROOT / "data" / "structure"
OUT = Path(__file__).resolve().parent

# ── Consistent color scheme with other paper figures ──────────────────
C_TEXT = "#2D2D2D"
C_GRID = "#E5E5E5"

# Warm palette matching fig3/fig4
STAGE_COLORS = {
    "raw": "#C4A882",       # same brown as XGBoost bars
    "cleaned": "#D97757",   # same coral as MLP bars
    "train_exp": "#A89B91", # muted warm grey
    "train_comp": "#E8D5B7",# same beige as FPredX bars
}
STAGE_LABELS = {
    "raw": "Raw FPbase",
    "cleaned": "Cleaned",
    "train_exp": "Train (experimental)",
    "train_comp": "Train (computational)",
}
STAGE_ORDER = ["raw", "cleaned", "train_exp", "train_comp"]

PROPS = ["ex_max", "em_max", "qy", "ext_coeff", "pka"]

# ── Load data ─────────────────────────────────────────────────────────
def load_raw():
    with open(SEQ_DIR / "fpbase_raw.json") as f:
        data = json.load(f)
    rows = []
    for prot in data:
        states = prot.get("states") or []
        if not states:
            rows.append({"slug": prot.get("slug"), **{p: None for p in PROPS}})
            continue
        default_slug = prot.get("default_state") or ""
        chosen = None
        for st in states:
            if st.get("slug") == default_slug:
                chosen = st
                break
        if chosen is None:
            chosen = states[0]
        rows.append({
            "slug": prot.get("slug"),
            "ex_max": chosen.get("ex_max"), "em_max": chosen.get("em_max"),
            "qy": chosen.get("qy"), "ext_coeff": chosen.get("ext_coeff"),
            "pka": chosen.get("pka"),
        })
    return pd.DataFrame(rows)

df_raw = load_raw()
df_clean = pd.read_csv(SEQ_DIR / "fp_cleaned.csv")

graft = pd.read_csv(STRUCT_DIR / "graft_summary.csv")
graft["source"] = np.where(
    graft["chrom_hetatm_name"] == "experimental", "experimental", "computational"
)
clean_props = df_clean[["slug"] + PROPS].rename(columns={"slug": "target_slug"})
train = graft.merge(clean_props, on="target_slug", how="left")
df_train_exp = train[train["source"] == "experimental"]
df_train_comp = train[train["source"] == "computational"]

for p in PROPS:
    df_raw[p] = pd.to_numeric(df_raw[p], errors="coerce")
    df_clean[p] = pd.to_numeric(df_clean[p], errors="coerce")

stage_dfs = {
    "raw": df_raw, "cleaned": df_clean,
    "train_exp": df_train_exp, "train_comp": df_train_comp,
}

n = {k: len(v) for k, v in stage_dfs.items()}
print(f"Raw: {n['raw']}, Cleaned: {n['cleaned']}, "
      f"Train exp: {n['train_exp']}, Train comp: {n['train_comp']}")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=12)


# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

# ═══════════════════════════════════════════════════════════════════════
# (a) Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])

df_corr = df_clean[PROPS].copy()
df_corr["stokes"] = df_clean["em_max"] - df_clean["ex_max"]
df_corr["brightness"] = df_clean["ext_coeff"] * df_clean["qy"]
corr = df_corr.corr(method="pearson")

im = ax_a.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
labels_corr = ["ex_max", "em_max", "QY", "ext_coeff", "pKa", "Stokes", "brightness"]
ax_a.set_xticks(range(len(labels_corr)))
ax_a.set_yticks(range(len(labels_corr)))
ax_a.set_xticklabels(labels_corr, rotation=45, ha="right", fontsize=12)
ax_a.set_yticklabels(labels_corr, fontsize=12)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                  color="white" if abs(val) > 0.5 else "black",
                  fontsize=11, fontweight="medium")
cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label("Pearson r", fontsize=13)
cb.ax.tick_params(labelsize=11)
ax_a.set_title("(a) Property Correlation", fontsize=15,
               fontweight="bold", color=C_TEXT, pad=12)

# ═══════════════════════════════════════════════════════════════════════
# (b) FP landscape: ex vs em
# ═══════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

m = df_clean[["ex_max", "em_max", "qy", "ext_coeff"]].dropna()
sc = ax_b.scatter(
    m["ex_max"], m["em_max"],
    c=m["qy"], cmap="YlOrBr",  # warm cmap consistent with palette
    s=np.clip(np.log10(m["ext_coeff"].clip(lower=1)) * 14, 5, 90),
    alpha=0.75, edgecolor=C_TEXT, linewidth=0.15, zorder=3,
)
ax_b.plot([280, 750], [280, 750], color="#A89B91", lw=1, ls="--",
          label="em = ex", zorder=1)
cbar = plt.colorbar(sc, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label("Quantum Yield", fontsize=13)
cbar.ax.tick_params(labelsize=11)
ax_b.set_xlabel("Excitation (nm)", fontsize=13, color=C_TEXT)
ax_b.set_ylabel("Emission (nm)", fontsize=13, color=C_TEXT)
ax_b.set_title(f"(b) FP Landscape (n={len(m)})", fontsize=15,
               fontweight="bold", color=C_TEXT, pad=12)
ax_b.legend(fontsize=11, loc="upper left", frameon=False)
ax_b.tick_params(labelsize=12)
style_ax(ax_b)

# ═══════════════════════════════════════════════════════════════════════
# (c) Pipeline attrition — horizontal bars
# ═══════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[0, 2])

counts = [n["raw"], n["cleaned"], n["train_exp"], n["train_comp"]]
labels_bar = [STAGE_LABELS[s] for s in STAGE_ORDER]
colors_bar = [STAGE_COLORS[s] for s in STAGE_ORDER]

y_pos = np.arange(len(STAGE_ORDER))[::-1]
bars = ax_c.barh(y_pos, counts, color=colors_bar, edgecolor="white", linewidth=1.2,
                 height=0.6)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(labels_bar, fontsize=12)
for y, c in zip(y_pos, counts):
    ax_c.text(c + max(counts) * 0.02, y, f"n = {c}", va="center", fontsize=12,
              color=C_TEXT, fontweight="medium")
ax_c.set_xlabel("Number of Proteins", fontsize=13, color=C_TEXT)
ax_c.set_title("(c) Pipeline Attrition", fontsize=15, fontweight="bold",
               color=C_TEXT, pad=12)
ax_c.set_xlim(0, max(counts) * 1.18)
style_ax(ax_c)
ax_c.yaxis.grid(False)
ax_c.xaxis.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
ax_c.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════════════════
# Bottom row: (d) ex_max, (e) em_max, (f) QY — KDE by stage
# ═══════════════════════════════════════════════════════════════════════
dist_props = [
    ("ex_max", "Excitation (nm)", (280, 720), "(d)"),
    ("em_max", "Emission (nm)", (330, 780), "(e)"),
    ("qy", "Quantum Yield", (-0.05, 1.05), "(f)"),
]

for col_idx, (prop, xlabel, xlim, letter) in enumerate(dist_props):
    ax = fig.add_subplot(gs[1, col_idx])
    grid = np.linspace(xlim[0], xlim[1], 400)

    for s_key in STAGE_ORDER:
        df = stage_dfs[s_key]
        v = pd.to_numeric(df[prop], errors="coerce").dropna().to_numpy()
        if len(v) < 3:
            continue
        label = f"{STAGE_LABELS[s_key]} (n={len(v)})"
        try:
            kde = gaussian_kde(v, bw_method=0.25)
            ax.fill_between(grid, kde(grid), alpha=0.18, color=STAGE_COLORS[s_key])
            ax.plot(grid, kde(grid), color=STAGE_COLORS[s_key], lw=2.2, label=label)
        except Exception:
            pass

    ax.set_xlabel(xlabel, fontsize=13, color=C_TEXT)
    if col_idx == 0:
        ax.set_ylabel("Density", fontsize=13, color=C_TEXT)
    ax.set_title(f"{letter} {prop} Distribution", fontsize=15,
                 fontweight="bold", color=C_TEXT, pad=12)
    ax.set_xlim(xlim)
    style_ax(ax)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    if col_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=False)

fig.suptitle("FPbase Dataset Statistics", fontsize=20, fontweight="bold",
             color=C_TEXT, y=0.99)
fig.savefig(OUT / "fig2_data_statistics.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved fig2_data_statistics.png")
