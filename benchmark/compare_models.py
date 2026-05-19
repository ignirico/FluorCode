"""
Head-to-head comparison of the 3 models:
  1. FPredX   (baseline)     — XGBoost on alignment one-hot
  2. LoRA-ESM2 + XGBoost     — LoRA fine-tuned ESM2-650M embeddings + XGBoost
  3. LoRA-ESM2 + MLP         — LoRA fine-tuned ESM2-650M embeddings + 2-layer MLP

Compares random CV and clustered CV (50% identity) across 6 targets.

Data sources:
  benchmark/clustered/clustered_cv_results.csv   (FPredX & XGBoost: ex/em/brightness)
  figures/xgb_extra_results.csv                  (FPredX & XGBoost: qy/ext_coeff/pka)
  figures/mlp_benchmark_results.csv              (MLP: all targets)

Outputs (all under benchmark/):
  head_to_head_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# ── Load data ────────────────────────────────────────────────────────────────
df_xgb = pd.read_csv(ROOT / "benchmark" / "clustered" / "clustered_cv_results.csv")
xgb_extra = ROOT / "figures" / "xgb_extra_results.csv"
if xgb_extra.exists():
    df_xgb = pd.concat([df_xgb, pd.read_csv(xgb_extra)], ignore_index=True)
df_mlp = pd.read_csv(ROOT / "figures" / "mlp_benchmark_results.csv")

TARGETS = ["ex_max", "em_max", "qy", "ext_coeff", "pka", "brightness"]
SCHEMES = ["random", "50"]
SCHEME_LABELS = {"random": "Random CV", "50": "Clustered 50%"}


def get_val(df, target, subset, scheme, metric):
    row = df[(df["target"] == target) & (df["subset"] == subset) & (df["scheme"] == scheme)]
    if row.empty:
        return None
    return row.iloc[0][metric]


# ── Build comparison table ───────────────────────────────────────────────────
rows = []
for scheme in SCHEMES:
    for target in TARGETS:
        # FPredX
        r = get_val(df_xgb, target, "A_onehot", scheme, "r")
        mae = get_val(df_xgb, target, "A_onehot", scheme, "pooled_mae")
        rows.append({"scheme": scheme, "target": target, "model": "FPredX",
                     "pearson_r": r, "mae": mae})

        # LoRA + XGBoost
        r = get_val(df_xgb, target, "B_lora", scheme, "r")
        mae = get_val(df_xgb, target, "B_lora", scheme, "pooled_mae")
        rows.append({"scheme": scheme, "target": target, "model": "LoRA+XGBoost",
                     "pearson_r": r, "mae": mae})

        # LoRA + MLP
        r = get_val(df_mlp, target, "MLP_lora", scheme, "r")
        mae = get_val(df_mlp, target, "MLP_lora", scheme, "pooled_mae")
        rows.append({"scheme": scheme, "target": target, "model": "LoRA+MLP",
                     "pearson_r": r, "mae": mae})

df = pd.DataFrame(rows)
df.to_csv(OUT / "head_to_head_comparison.csv", index=False, float_format="%.4f")

# ── Print summary ────────────────────────────────────────────────────────────
for scheme in SCHEMES:
    print(f"\n=== {SCHEME_LABELS[scheme]} ===")
    sub = df[df["scheme"] == scheme].pivot(index="target", columns="model", values="pearson_r")
    sub = sub.reindex(columns=["FPredX", "LoRA+XGBoost", "LoRA+MLP"])
    sub = sub.reindex(TARGETS)
    print(sub.to_string(float_format="%.3f"))

print(f"\nSaved → {OUT / 'head_to_head_comparison.csv'}")
