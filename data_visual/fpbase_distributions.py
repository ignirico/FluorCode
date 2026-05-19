"""
FPbase data graphical analysis — raw vs. cleaned vs. training splits.

Produces four PNGs + one CSV in data_visual/:
  1. fpbase_raw_vs_clean.png        — pipeline attrition funnel + coverage
  2. fpbase_ex_em_by_stage.png      — ex_max & em_max across 4 data stages
  3. fpbase_supporting_distributions.png — qy/pka/ext_coeff raw vs cleaned
  4. fpbase_correlation.png         — correlation heatmap + ex-vs-em landscape
  5. fpbase_property_stats.csv      — summary statistics table (long-form)

Stages:
  raw                 — 1040 proteins from fpbase_raw.json
  cleaned             — ~986 proteins from fp_cleaned.csv
  train_experimental  — 162 training proteins with RCSB-donor structure
  train_computational — 751 training proteins with SimpleFold structure
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "sequence"
STRUCT_DIR = ROOT / "data" / "structure"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROPS_ALL = ["ex_max", "em_max", "qy", "ext_coeff", "pka"]
PROPS_PRIMARY = ["ex_max", "em_max"]
PROPS_SUPPORT = ["qy", "ext_coeff", "pka"]

STAGE_ORDER = ["raw", "cleaned", "train_experimental", "train_computational"]
STAGE_COLORS = {
    "raw": "#888888",
    "cleaned": "#1f77b4",
    "train_experimental": "#2ca02c",
    "train_computational": "#ff7f0e",
}
STAGE_LABELS = {
    "raw": "Raw FPbase",
    "cleaned": "Cleaned",
    "train_experimental": "Train (experimental)",
    "train_computational": "Train (computational)",
}

# ── Section 1 ── Load stage membership ───────────────────────────────────────

def load_raw():
    """Flatten fpbase_raw.json into a flat DataFrame (one row per protein,
    using default state or first state, mirroring fetch_fpbase.py logic)."""
    with open(SEQ_DIR / "fpbase_raw.json") as f:
        data = json.load(f)

    rows = []
    for prot in data:
        states = prot.get("states") or []
        if not states:
            rows.append({"slug": prot.get("slug"), **{p: None for p in PROPS_ALL}})
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
            "ex_max": chosen.get("ex_max"),
            "em_max": chosen.get("em_max"),
            "qy": chosen.get("qy"),
            "ext_coeff": chosen.get("ext_coeff"),
            "pka": chosen.get("pka"),
        })
    return pd.DataFrame(rows)


def load_cleaned():
    df = pd.read_csv(SEQ_DIR / "fp_cleaned.csv")
    return df[["slug"] + PROPS_ALL].copy()


def load_training_with_source():
    """Return graft_summary joined with cleaned-property values and a
    structure_source column ∈ {experimental, computational}."""
    graft = pd.read_csv(STRUCT_DIR / "graft_summary.csv")
    graft["structure_source"] = np.where(
        graft["chrom_hetatm_name"] == "experimental",
        "experimental", "computational",
    )
    clean = load_cleaned().rename(columns={"slug": "target_slug"})
    train = graft.merge(clean, on="target_slug", how="left")
    train = train.rename(columns={"target_slug": "slug"})
    return train[["slug", "structure_source"] + PROPS_ALL].copy()


print("Loading data …")
df_raw = load_raw()
df_clean = load_cleaned()
df_train = load_training_with_source()

n_exp = int((df_train["structure_source"] == "experimental").sum())
n_comp = int((df_train["structure_source"] == "computational").sum())
print(f"  raw:                 {len(df_raw)}")
print(f"  cleaned:             {len(df_clean)}")
print(f"  training total:      {len(df_train)}")
print(f"  training experimental: {n_exp}")
print(f"  training computational: {n_comp}")

# Long-form frame for stage-stratified plots
def tag(df, stage):
    d = df.copy()
    d["stage"] = stage
    return d[["slug", "stage"] + PROPS_ALL]

df_all = pd.concat([
    tag(df_raw, "raw"),
    tag(df_clean, "cleaned"),
    tag(df_train[df_train["structure_source"] == "experimental"], "train_experimental"),
    tag(df_train[df_train["structure_source"] == "computational"], "train_computational"),
], ignore_index=True)

# Numeric coercion (some raw values arrive as strings or None)
for p in PROPS_ALL:
    df_all[p] = pd.to_numeric(df_all[p], errors="coerce")
    df_raw[p] = pd.to_numeric(df_raw[p], errors="coerce")
    df_clean[p] = pd.to_numeric(df_clean[p], errors="coerce")
    df_train[p] = pd.to_numeric(df_train[p], errors="coerce")


# ── Section 2 ── fpbase_raw_vs_clean.png ─────────────────────────────────────

print("\nPlotting raw-vs-clean attrition …")
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A — funnel
stages = ["raw", "cleaned", "train_experimental", "train_computational"]
counts = [len(df_raw), len(df_clean), n_exp, n_comp]
labels = [STAGE_LABELS[s] for s in stages]
colors = [STAGE_COLORS[s] for s in stages]

y_pos = np.arange(len(stages))[::-1]
axA.barh(y_pos, counts, color=colors, edgecolor="black", alpha=0.9)
axA.set_yticks(y_pos)
axA.set_yticklabels(labels)
for y, c in zip(y_pos, counts):
    axA.text(c + max(counts) * 0.01, y, f"n={c}", va="center", fontsize=10)
axA.set_xlabel("Number of proteins")
axA.set_title("Pipeline attrition: Raw → Cleaned → Training set")
axA.set_xlim(0, max(counts) * 1.15)

# Panel B — coverage per property across stages
stage_dfs = {
    "raw": df_raw,
    "cleaned": df_clean,
    "train_experimental": df_train[df_train["structure_source"] == "experimental"],
    "train_computational": df_train[df_train["structure_source"] == "computational"],
}
width = 0.2
x = np.arange(len(PROPS_ALL))
for i, stage in enumerate(stages):
    d = stage_dfs[stage]
    cov = [d[p].notna().sum() for p in PROPS_ALL]
    offset = (i - 1.5) * width
    axB.bar(x + offset, cov, width=width, color=STAGE_COLORS[stage],
            edgecolor="black", linewidth=0.4, alpha=0.9,
            label=f"{STAGE_LABELS[stage]} (total={len(d)})")
    for xi, c in zip(x + offset, cov):
        axB.text(xi, c + 10, str(c), ha="center", fontsize=7)
axB.set_xticks(x)
axB.set_xticklabels(PROPS_ALL)
axB.set_ylabel("Non-null count")
axB.set_title("Property coverage by stage")
axB.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "fpbase_raw_vs_clean.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {out}")


# ── Section 3 ── fpbase_ex_em_by_stage.png ───────────────────────────────────

print("Plotting ex_max & em_max by stage …")
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

for row, prop in enumerate(PROPS_PRIMARY):
    # Gather per-stage arrays
    stage_vals = {}
    for s in STAGE_ORDER:
        v = df_all[df_all["stage"] == s][prop].dropna().to_numpy()
        stage_vals[s] = v

    xmin = min(v.min() for v in stage_vals.values() if len(v))
    xmax = max(v.max() for v in stage_vals.values() if len(v))
    grid = np.linspace(xmin - 10, xmax + 10, 400)

    # Col 1 — overlaid KDE + histogram (density)
    ax = axes[row, 0]
    for s in STAGE_ORDER:
        v = stage_vals[s]
        if len(v) < 2:
            continue
        ax.hist(v, bins=40, density=True, alpha=0.18,
                color=STAGE_COLORS[s], edgecolor="none")
        try:
            kde = gaussian_kde(v)
            ax.plot(grid, kde(grid), color=STAGE_COLORS[s], lw=1.6,
                    label=f"{STAGE_LABELS[s]} (n={len(v)})")
        except Exception:
            pass
    ax.set_xlabel(f"{prop} (nm)", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title(f"{prop} — overlaid KDE across stages", fontsize=16, fontweight="bold")
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=13)

    # Col 2 — violin + strip
    ax = axes[row, 1]
    data_list = [stage_vals[s] for s in STAGE_ORDER]
    parts = ax.violinplot(data_list, showmedians=True, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(STAGE_COLORS[STAGE_ORDER[i]])
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
    # jittered strip
    rng = np.random.default_rng(42)
    for i, s in enumerate(STAGE_ORDER):
        v = stage_vals[s]
        if len(v) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(np.full(len(v), i + 1) + jitter, v, s=4,
                   color=STAGE_COLORS[s], alpha=0.35, edgecolor="none")
    ax.set_xticks(np.arange(1, len(STAGE_ORDER) + 1))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER],
                       rotation=20, ha="right", fontsize=13)
    ax.set_ylabel(f"{prop} (nm)", fontsize=15)
    ax.set_title(f"{prop} — violin + jitter per stage", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=13)

    # Col 3 — ECDF
    ax = axes[row, 2]
    for s in STAGE_ORDER:
        v = np.sort(stage_vals[s])
        if len(v) == 0:
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, color=STAGE_COLORS[s], lw=1.8,
                label=f"{STAGE_LABELS[s]} (n={len(v)})")
    ax.set_xlabel(f"{prop} (nm)", fontsize=15)
    ax.set_ylabel("Cumulative fraction", fontsize=15)
    ax.set_title(f"{prop} — ECDF across stages", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, loc="lower right")
    ax.tick_params(labelsize=13)

plt.suptitle("ex_max & em_max distributions across pipeline stages", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / "fpbase_ex_em_by_stage.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {out}")


# ── Section 4 ── fpbase_supporting_distributions.png ─────────────────────────

print("Plotting supporting distributions (qy, ext_coeff, pka) …")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, prop in zip(axes, PROPS_SUPPORT):
    v_raw = df_raw[prop].dropna().to_numpy()
    v_clean = df_clean[prop].dropna().to_numpy()
    if len(v_raw) == 0 or len(v_clean) == 0:
        continue
    lo = min(v_raw.min(), v_clean.min())
    hi = max(v_raw.max(), v_clean.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(v_raw, bins=bins, alpha=0.5, color=STAGE_COLORS["raw"],
            label=f"Raw (n={len(v_raw)})", edgecolor="black", linewidth=0.3)
    ax.hist(v_clean, bins=bins, alpha=0.55, color=STAGE_COLORS["cleaned"],
            label=f"Cleaned (n={len(v_clean)})", edgecolor="black", linewidth=0.3)
    ax.axvline(np.median(v_clean), color=STAGE_COLORS["cleaned"],
               ls="--", lw=1, label=f"cleaned median={np.median(v_clean):.2f}")
    ax.set_xlabel(prop)
    ax.set_ylabel("Count")
    ax.set_title(f"{prop}")
    ax.legend(fontsize=8)
plt.suptitle("Supporting property distributions — raw vs. cleaned", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = OUT_DIR / "fpbase_supporting_distributions.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {out}")


# ── Section 5 ── fpbase_correlation.png ──────────────────────────────────────

print("Plotting correlation heatmap + landscape …")

# Claude warm color scheme
from matplotlib.colors import LinearSegmentedColormap
C_TEXT = "#2D2D2D"
C_GRID = "#E5E5E5"

# Custom warm diverging colormap: beige → white → coral
warm_div = LinearSegmentedColormap.from_list("warm_div", [
    "#D97757",   # coral (negative)
    "#F2E0D0",   # light warm
    "#FFFFFF",   # white (zero)
    "#D4C4A8",   # light tan
    "#8B6F47",   # dark brown (positive)
])

df_corr = df_clean[PROPS_ALL].copy()
df_corr["stokes"] = df_clean["em_max"] - df_clean["ex_max"]
df_corr["brightness"] = df_clean["ext_coeff"] * df_clean["qy"]
corr = df_corr.corr(method="pearson")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 7))

im = axA.imshow(corr.values, cmap=warm_div, vmin=-1, vmax=1)
labels_corr = ["ex_max", "em_max", "QY", "ext_coeff", "pKa", "Stokes", "brightness"]
axA.set_xticks(range(len(labels_corr)))
axA.set_yticks(range(len(labels_corr)))
axA.set_xticklabels(labels_corr, rotation=45, ha="right", fontsize=14)
axA.set_yticklabels(labels_corr, fontsize=14)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        axA.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="white" if abs(val) > 0.65 else C_TEXT,
                 fontsize=12, fontweight="medium")
cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
cb.set_label("Pearson r", fontsize=14)
cb.ax.tick_params(labelsize=12)
axA.set_title("(a) Property Correlation (cleaned set)", fontsize=16,
              fontweight="bold", color=C_TEXT, pad=12)

m = df_clean[["ex_max", "em_max", "qy", "ext_coeff"]].dropna()
sc = axB.scatter(
    m["ex_max"], m["em_max"],
    c=m["qy"], cmap="YlOrBr",
    s=np.clip(np.log10(m["ext_coeff"].clip(lower=1)) * 14, 5, 90),
    alpha=0.75, edgecolor=C_TEXT, linewidth=0.15, zorder=3,
)
axB.plot([280, 750], [280, 750], color="#A89B91", lw=1, ls="--",
         label="em = ex", zorder=1)
cbar = plt.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
cbar.set_label("Quantum Yield", fontsize=14)
cbar.ax.tick_params(labelsize=12)
axB.set_xlabel("Excitation (nm)", fontsize=14, color=C_TEXT)
axB.set_ylabel("Emission (nm)", fontsize=14, color=C_TEXT)
axB.set_title(f"(b) FP Landscape (n={len(m)}, size \u221d log ext_coeff)",
              fontsize=16, fontweight="bold", color=C_TEXT, pad=12)
axB.legend(fontsize=13, loc="upper left", frameon=False)
axB.tick_params(labelsize=13, colors=C_TEXT)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.spines["left"].set_color(C_GRID)
axB.spines["bottom"].set_color(C_GRID)
axB.grid(alpha=0.2)

plt.tight_layout()
out = OUT_DIR / "fpbase_correlation.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved {out}")


# ── Section 6 ── fpbase_property_stats.csv ───────────────────────────────────

print("Writing stats CSV …")
rows = []
def summarize(prop, stage, values):
    v = pd.Series(values).dropna()
    if len(v) == 0:
        return {
            "property": prop, "stage": stage, "n": 0,
            "mean": None, "std": None, "min": None,
            "q25": None, "median": None, "q75": None, "max": None,
            "n_outliers_1p5iqr": 0,
        }
    q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((v < lo) | (v > hi)).sum())
    return {
        "property": prop, "stage": stage, "n": int(len(v)),
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "q25": float(q1), "median": float(q2),
        "q75": float(q3), "max": float(v.max()),
        "n_outliers_1p5iqr": n_out,
    }

for prop in PROPS_PRIMARY:
    for s in STAGE_ORDER:
        rows.append(summarize(prop, s, df_all[df_all["stage"] == s][prop]))

for prop in PROPS_SUPPORT:
    rows.append(summarize(prop, "raw", df_raw[prop]))
    rows.append(summarize(prop, "cleaned", df_clean[prop]))

stats = pd.DataFrame(rows)
out = OUT_DIR / "fpbase_property_stats.csv"
stats.to_csv(out, index=False, float_format="%.4f")
print(f"  saved {out}")
print("\nStats preview:")
print(stats.to_string(index=False))
print("\nDone.")
