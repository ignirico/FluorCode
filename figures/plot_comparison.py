"""
Publication-quality comparison figures for ICML AI4Science workshop paper.

Models: FPredX (baseline), LoRA-ESM2+XGBoost, LoRA-ESM2+MLP
Outputs:
  - fig3_random_cv.png           — Random CV: Pearson R + MAE (all 6 properties)
  - fig4_clustered_cv.png        — Clustered CV: (a) Pearson R, (b) MAE degradation, (c) MAE bar
  - mae_degradation_across_thresholds.png — standalone version of Fig 4b

Data sources:
  - FPredX & LoRA+XGBoost (ex/em/brightness): benchmark/clustered/clustered_cv_results.csv
  - FPredX & LoRA+XGBoost (qy/ext_coeff/pka): figures/xgb_extra_results.csv
  - LoRA+MLP (all targets): figures/mlp_benchmark_results.csv

Usage: python figures/plot_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSV_XGB = ROOT / "benchmark" / "clustered" / "clustered_cv_results.csv"
CSV_XGB_EXTRA = Path(__file__).resolve().parent / "xgb_extra_results.csv"
CSV_MLP = Path(__file__).resolve().parent / "mlp_benchmark_results.csv"
OUT     = Path(__file__).resolve().parent

# ── Color scheme ──────────────────────────────────────────────────────
C_FPREDX  = "#E8D5B7"
C_XGB     = "#C4A882"
C_MLP     = "#D97757"
C_TEXT    = "#2D2D2D"
C_GRID    = "#E5E5E5"
COLORS    = [C_FPREDX, C_XGB, C_MLP]
MODELS    = ["FPredX", "LoRA-ESM2\n(XGBoost)", "LoRA-ESM2\n(MLP)"]

# ── Load data ──────────────────────────────────────────────────────────
df_xgb = pd.read_csv(CSV_XGB)
df_xgb_extra = pd.read_csv(CSV_XGB_EXTRA) if CSV_XGB_EXTRA.exists() else pd.DataFrame()
df_mlp = pd.read_csv(CSV_MLP)


def get_val(target, scheme, model, metric):
    """Get metric value for a model/target/scheme combination."""
    if "MLP" in model and "XGBoost" not in model:
        row = df_mlp[(df_mlp["target"] == target) & (df_mlp["subset"] == "MLP_lora")
                     & (df_mlp["scheme"] == scheme)]
        if row.empty:
            return None
        return row.iloc[0][metric]

    # FPredX or LoRA+XGBoost
    subset = "A_onehot" if "FPredX" in model else "B_lora"

    # Try main CSV first
    row = df_xgb[(df_xgb["target"] == target) & (df_xgb["subset"] == subset)
                 & (df_xgb["scheme"] == scheme)]
    if not row.empty:
        return row.iloc[0][metric]

    # Fall back to extra XGB results
    if not df_xgb_extra.empty:
        row = df_xgb_extra[(df_xgb_extra["target"] == target)
                           & (df_xgb_extra["subset"] == subset)
                           & (df_xgb_extra["scheme"] == scheme)]
        if not row.empty:
            return row.iloc[0][metric]

    return None


def style_ax(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, fontsize=13, color=C_TEXT, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=14, color=C_TEXT, fontweight="bold", pad=10)
    ax.tick_params(colors=C_TEXT, labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


# ── Target groups ─────────────────────────────────────────────────────
main_targets = ["ex_max", "em_max"]
main_labels  = ["Excitation\n(nm)", "Emission\n(nm)"]

all_targets = ["ex_max", "em_max", "qy", "ext_coeff", "pka", "brightness"]
all_labels  = ["Excitation\n(nm)", "Emission\n(nm)", "QY", "Ext. Coeff.\n(M\u207b\xb9cm\u207b\xb9)",
               "pKa", "Brightness\n(%)"]


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Random CV — Pearson R + MAE across all 6 properties
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
x = np.arange(len(all_targets))
w = 0.25

# (a) Pearson R
for i, model in enumerate(MODELS):
    vals = [get_val(t, "random", model, "r") for t in all_targets]
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax1.bar(x + i * w, plot_vals, w, label=model, color=COLORS[i],
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=C_TEXT,
                     fontweight="medium")

ax1.set_xticks(x + w)
ax1.set_xticklabels(all_labels, fontsize=11)
ax1.set_ylim(0, 1.15)
style_ax(ax1, "Pearson R", "(a) Pearson Correlation")

# (b) MAE — ex_max and em_max only (other targets have different units)
x2 = np.arange(len(main_targets))
w2 = 0.25

for i, model in enumerate(MODELS):
    vals = [get_val(t, "random", model, "pooled_mae") for t in main_targets]
    bars = ax2.bar(x2 + i * w2, vals, w2, label=model, color=COLORS[i],
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=11, color=C_TEXT,
                     fontweight="medium")

ax2.set_xticks(x2 + w2)
ax2.set_xticklabels(main_labels, fontsize=12)
style_ax(ax2, "MAE (nm)", "(b) Mean Absolute Error")

fig.legend(*ax1.get_legend_handles_labels(), frameon=False, fontsize=12,
           loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
fig.suptitle("Random Cross-Validation", fontsize=16, color=C_TEXT, fontweight="bold", y=1.02)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.savefig(OUT / "fig3_random_cv.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig3_random_cv.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Clustered CV — 3 subfigures
#   (a) Pearson R at 50% identity, 6 properties
#   (b) MAE degradation across clustering thresholds (ex_max, em_max)
#   (c) MAE at 50% identity (ex_max, em_max)
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))

# Top row: (a) Pearson R — spans full width
ax_a = fig.add_subplot(2, 1, 1)

x = np.arange(len(all_targets))
w = 0.25

for i, model in enumerate(MODELS):
    vals = [get_val(t, "50", model, "r") for t in all_targets]
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax_a.bar(x + i * w, plot_vals, w, label=model, color=COLORS[i],
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f"{v:.2f}", ha="center", va="bottom", fontsize=12, color=C_TEXT,
                      fontweight="medium")

ax_a.set_xticks(x + w)
ax_a.set_xticklabels(all_labels, fontsize=14)
ax_a.set_ylim(0, 1.18)
style_ax(ax_a, "Pearson R", "(a) Pearson Correlation — 50% Seq. Identity Clustering")
ax_a.yaxis.label.set_size(15)
ax_a.title.set_size(16)
ax_a.tick_params(labelsize=13)
ax_a.legend(frameon=False, fontsize=13, loc="upper center",
            bbox_to_anchor=(0.5, -0.12), ncol=3)

# Bottom row: (b) MAE degradation + (c) MAE bar
ax_b = fig.add_subplot(2, 2, 3)
ax_c = fig.add_subplot(2, 2, 4)

# (b) MAE degradation across thresholds
schemes = ["random", "90", "70", "50"]
scheme_labels = ["Random", "90%", "70%", "50%"]
markers = ["s", "D", "o"]
linestyles = ["-", "-", "-"]

for target, ls_style in zip(main_targets, ["-", "--"]):
    t_label = "Ex" if target == "ex_max" else "Em"
    for i, model in enumerate(MODELS):
        vals = [get_val(target, s, model, "pooled_mae") for s in schemes]
        if any(v is None for v in vals):
            continue
        label = f"{model} ({t_label})" if target == "ex_max" else None
        ax_b.plot(scheme_labels, vals, marker=markers[i], color=COLORS[i],
                  linewidth=2.2, markersize=7, label=label,
                  markeredgecolor="white", markeredgewidth=1.2,
                  linestyle=ls_style, zorder=3,
                  alpha=1.0 if target == "ex_max" else 0.55)

style_ax(ax_b, "MAE (nm)", "(b) MAE Degradation Across Thresholds")
ax_b.set_xlabel("Clustering Threshold", fontsize=11, color=C_TEXT)
ax_b.legend(frameon=False, fontsize=8, loc="upper center",
            bbox_to_anchor=(0.5, -0.18), ncol=3)

# (c) MAE bar at 50%
x3 = np.arange(len(main_targets))
w3 = 0.25

for i, model in enumerate(MODELS):
    vals = [get_val(t, "50", model, "pooled_mae") for t in main_targets]
    bars = ax_c.bar(x3 + i * w3, vals, w3, label=model, color=COLORS[i],
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                      f"{v:.1f}", ha="center", va="bottom", fontsize=11, color=C_TEXT,
                      fontweight="medium")

ax_c.set_xticks(x3 + w3)
ax_c.set_xticklabels(main_labels, fontsize=12)
style_ax(ax_c, "MAE (nm)", "(c) MAE — 50% Seq. Identity")

fig.suptitle("Clustered Cross-Validation", fontsize=16, color=C_TEXT,
             fontweight="bold", y=1.01)
fig.tight_layout()
fig.subplots_adjust(hspace=0.45)
fig.savefig(OUT / "fig4_clustered_cv.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved fig4_clustered_cv.png")


# ══════════════════════════════════════════════════════════════════════
# Standalone: MAE degradation across clustering thresholds
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
schemes = ["random", "90", "70", "50"]
scheme_labels = ["Random", "90%", "70%", "50%"]
markers = ["s", "D", "o"]

for j, (target, label) in enumerate(zip(main_targets, ["Excitation (nm)", "Emission (nm)"])):
    ax = axes[j]
    for i, model in enumerate(MODELS):
        vals = [get_val(target, s, model, "pooled_mae") for s in schemes]
        if any(v is None for v in vals):
            continue
        ax.plot(scheme_labels, vals, marker=markers[i], color=COLORS[i],
                linewidth=2.5, markersize=9, label=model,
                markeredgecolor="white", markeredgewidth=1.2,
                linestyle="-", zorder=3)
        # annotate each point
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=10, color=C_TEXT, fontweight="medium")
    style_ax(ax, "MAE (nm)" if j == 0 else "", label)
    ax.title.set_size(15)
    ax.set_xlabel("Clustering Threshold", fontsize=13, color=C_TEXT)
    ax.tick_params(labelsize=12)

fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=13,
           loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
fig.suptitle("MAE Degradation Under Increasing Clustering Stringency",
             fontsize=16, color=C_TEXT, fontweight="bold", y=1.02)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.savefig(OUT / "mae_degradation_across_thresholds.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved mae_degradation_across_thresholds.png")


print(f"\nAll figures saved to {OUT}")
