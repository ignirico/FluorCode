"""
Benchmark: Our Models vs FPredX (Tam & Zhang, 2021)
====================================================
Replicates FPredX's evaluation protocol (20-fold random CV)
and compares alignment-based OneHot + XGBoost.

FPredX used: 672 FPs, 3362 one-hot features, XGBoost, 20-fold random CV
We use:      986 FPs, 1271 one-hot features, Optuna-tuned XGBoost, 20-fold random CV

FPredX results (from paper):
  ex_max best single fold MAE: 11.23 nm
  em_max best single fold MAE:  7.72 nm

Usage:
    python3 ML_Algorithms/Improved_FPredX/benchmark_vs_fprex.py

Output:
    ML_Algorithms/Improved_FPredX/benchmark_results/
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from Bio import SeqIO
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = Path(__file__).resolve().parent / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPREX_RESULTS = {
    "ex_max": {"best_fold_mae": 11.23, "naive_mae": 14.55},
    "em_max": {"best_fold_mae": 7.72,  "naive_mae": 9.19},
}

TARGETS = ["ex_max", "em_max", "qy", "ext_coeff", "pka"]


def build_onehot_features(meta, aligned, min_freq=0.02):
    """Build alignment-based one-hot features (FPredX style)."""
    aln_len = len(next(iter(aligned.values())))
    n_seqs = len(meta)

    position_chars = []
    for pos in range(aln_len):
        chars = Counter()
        for slug in meta["slug"]:
            if slug in aligned:
                chars[aligned[slug][pos]] += 1
        for char, count in chars.items():
            if count / n_seqs >= min_freq and char != "-":
                position_chars.append((pos, char))

    X = np.zeros((n_seqs, len(position_chars)), dtype=np.float32)
    for i, slug in enumerate(meta["slug"]):
        if slug not in aligned:
            continue
        seq = aligned[slug]
        for j, (pos, char) in enumerate(position_chars):
            if seq[pos] == char:
                X[i, j] = 1.0

    # Remove zero-variance features
    var = X.var(axis=0)
    X = X[:, var > 0]
    return X


def tune_xgboost(X, y, n_trials=30):
    """Quick Optuna tuning with 3-fold CV."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42, "n_jobs": -1, "tree_method": "hist",
        }
        kf3 = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        for tr, te in kf3.split(X):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr], verbose=False)
            maes.append(mean_absolute_error(y[te], m.predict(X[te])))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update({"random_state": 42, "n_jobs": -1, "tree_method": "hist"})
    return best


def evaluate_20fold(X, y, params):
    """20-fold random CV, returns per-fold and pooled metrics."""
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    fold_maes = []
    all_preds = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx], verbose=False)
        pred = m.predict(X[test_idx])
        fold_maes.append(mean_absolute_error(y[test_idx], pred))
        all_preds[test_idx] = pred

    pooled_mae = mean_absolute_error(y, all_preds)
    pooled_rmse = np.sqrt(mean_squared_error(y, all_preds))
    pooled_r, _ = pearsonr(y, all_preds)
    pooled_r2 = 1 - np.sum((y - all_preds)**2) / np.sum((y - y.mean())**2)

    return {
        "pooled_mae": float(pooled_mae),
        "pooled_rmse": float(pooled_rmse),
        "pooled_r": float(pooled_r),
        "pooled_r2": float(pooled_r2),
        "mean_fold_mae": float(np.mean(fold_maes)),
        "std_fold_mae": float(np.std(fold_maes)),
        "best_fold_mae": float(min(fold_maes)),
        "worst_fold_mae": float(max(fold_maes)),
        "all_fold_maes": [float(m) for m in fold_maes],
        "predictions": all_preds,
        "true_values": y,
    }


def main():
    # ── Load data ─────────────────────────────────────────────
    aligned = {}
    for rec in SeqIO.parse(DATA_DIR / "fp_sequences_aligned.fasta", "fasta"):
        aligned[rec.id] = str(rec.seq)

    meta = pd.read_csv(DATA_DIR / "fp_embeddings_meta.csv")
    print(f"Total FPs: {len(meta)}")
    print(f"FPs with alignment: {sum(s in aligned for s in meta['slug'])}")

    # ── Build features ────────────────────────────────────────
    print("\nBuilding alignment-based one-hot features...")
    X_onehot = build_onehot_features(meta, aligned)
    print(f"One-hot features: {X_onehot.shape[1]}")

    # ── Benchmark each target ─────────────────────────────────
    all_results = []

    for target in TARGETS:
        mask = meta[target].notna() & meta["slug"].isin(aligned)
        df = meta[mask]
        X = X_onehot[df["emb_idx"].values]
        y = df[target].values

        if len(y) < 50:
            print(f"\n{target}: too few samples ({len(y)}), skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Target: {target} (n={len(y)})")
        print(f"{'='*60}")

        # Tune
        print("  Tuning XGBoost (30 Optuna trials)...")
        best_params = tune_xgboost(X, y, n_trials=30)

        # Evaluate
        print("  Running 20-fold CV...")
        result = evaluate_20fold(X, y, best_params)
        preds = result.pop("predictions")
        true = result.pop("true_values")

        # Compare with FPredX
        fprex = FPREX_RESULTS.get(target, {})
        result["target"] = target
        result["n_samples"] = int(len(y))
        result["fprex_best_fold_mae"] = fprex.get("best_fold_mae")
        result["fprex_naive_mae"] = fprex.get("naive_mae")
        result["beats_fprex_best_fold"] = (
            result["best_fold_mae"] < fprex["best_fold_mae"]
            if "best_fold_mae" in fprex else None
        )
        result["beats_fprex_pooled"] = (
            result["pooled_mae"] < fprex["best_fold_mae"]
            if "best_fold_mae" in fprex else None
        )
        result["xgb_params"] = best_params

        all_results.append(result)

        # Print
        print(f"\n  {'Metric':<25s} {'Ours':>10s} {'FPredX':>10s}")
        print(f"  {'-'*47}")
        print(f"  {'Pooled MAE':<25s} {result['pooled_mae']:10.2f} {'—':>10s}")
        print(f"  {'Mean fold MAE':<25s} {result['mean_fold_mae']:10.2f} {'—':>10s}")
        if fprex.get("best_fold_mae"):
            print(f"  {'Best fold MAE':<25s} {result['best_fold_mae']:10.2f} {fprex['best_fold_mae']:10.2f}")
            beat = "YES" if result["beats_fprex_best_fold"] else "no"
            print(f"  {'Beats FPredX best fold?':<25s} {beat:>10s}")
        else:
            print(f"  {'Best fold MAE':<25s} {result['best_fold_mae']:10.2f} {'N/A':>10s}")
        print(f"  {'Pearson r':<25s} {result['pooled_r']:10.4f}")
        print(f"  {'R²':<25s} {result['pooled_r2']:10.4f}")

        # Save per-target
        save_result = {k: v for k, v in result.items() if k != "all_fold_maes"}
        with open(OUT_DIR / f"{target}_results.json", "w") as f:
            json.dump(save_result, f, indent=2)

        # Save predictions for plotting
        np.savez(OUT_DIR / f"{target}_predictions.npz",
                 predictions=preds, true_values=true,
                 slugs=df["slug"].values)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — 20-Fold Random CV Benchmark")
    print(f"{'='*60}")
    print(f"\n{'Target':>12s} {'n':>6s} {'Pooled MAE':>12s} {'Best Fold':>12s} {'FPredX':>12s} {'Beat?':>8s} {'r':>8s} {'R²':>8s}")
    print("-" * 80)
    for r in all_results:
        fprex_val = f"{r['fprex_best_fold_mae']:.2f}" if r["fprex_best_fold_mae"] else "N/A"
        beat = "YES" if r.get("beats_fprex_best_fold") else ("no" if r.get("beats_fprex_best_fold") is not None else "—")
        print(f"{r['target']:>12s} {r['n_samples']:6d} {r['pooled_mae']:12.4f} {r['best_fold_mae']:12.4f} {fprex_val:>12s} {beat:>8s} {r['pooled_r']:8.4f} {r['pooled_r2']:8.4f}")

    # Save summary
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ["all_fold_maes", "xgb_params"]}
                            for r in all_results])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "all_fold_maes"} for r in all_results], f, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets_to_plot = [t for t in TARGETS if (OUT_DIR / f"{t}_predictions.npz").exists()]
        n_plots = len(targets_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for idx, target in enumerate(targets_to_plot):
            ax = axes[idx]
            data = np.load(OUT_DIR / f"{target}_predictions.npz")
            yt, yp = data["true_values"], data["predictions"]

            ax.scatter(yt, yp, alpha=0.3, s=15)
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            m = (hi - lo) * 0.05
            ax.plot([lo-m, hi+m], [lo-m, hi+m], "r--", alpha=0.5)

            mae = mean_absolute_error(yt, yp)
            r, _ = pearsonr(yt, yp)
            r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)

            unit = "nm" if target in ["ex_max", "em_max"] else ""
            ax.set_title(f"{target}\nMAE={mae:.2f}{unit}  r={r:.3f}  R²={r2:.3f}")
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "scatter_plots.png", dpi=150)
        print(f"\nSaved scatter_plots.png")
    except ImportError:
        pass

    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
