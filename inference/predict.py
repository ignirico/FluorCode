"""Predict ex_max, em_max, qy, ext_coeff, pka for a sequence or FASTA.

Examples:
    python predict.py -s MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK -c fold_0/best.pt
    python predict.py -f my_fps.fasta -d checkpoints/ --ensemble
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import build_model, load_checkpoint, TARGETS

CLAMP_RANGES = {
    "ex_max": (300, 800), "em_max": (300, 800),
    "qy": (0.0, 1.0), "ext_coeff": (0, 300000), "pka": (0, 14),
}


def parse_fasta(fasta_path):
    """Parse a FASTA file into list of (name, sequence) tuples."""
    entries = []
    current_name = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    entries.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_name:
        entries.append((current_name, "".join(current_seq)))
    return entries


def predict_single(model, alphabet, sequence, target_stats, device="cpu",
                   name="query", chrom_positions=None):
    """Predict properties for a single sequence. Returns dict of predictions."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([(name, sequence)])
    tokens = tokens.to(device)
    seq_lens = torch.tensor([len(sequence)], dtype=torch.long, device=device)

    chrom_pos = None
    if chrom_positions is not None:
        chrom_pos = torch.tensor([chrom_positions], dtype=torch.long, device=device)

    with torch.no_grad():
        preds, _ = model(tokens, seq_lens, chrom_pos)

    results = {}
    for t in TARGETS:
        mu, sd = target_stats[t]
        val = float(preds[t].cpu().numpy()[0]) * sd + mu
        lo, hi = CLAMP_RANGES[t]
        val = max(lo, min(hi, val))
        results[t] = round(val, 3)

    return results


def predict_batch(model, alphabet, sequences, target_stats, device="cpu",
                  batch_size=8):
    """Predict properties for multiple sequences. Returns list of dicts."""
    batch_converter = alphabet.get_batch_converter()
    all_results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        data = [(name, seq) for name, seq in batch]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        seq_lens = torch.tensor([len(seq) for _, seq in batch],
                                dtype=torch.long, device=device)

        with torch.no_grad():
            preds, _ = model(tokens, seq_lens)

        for j in range(len(batch)):
            results = {"name": batch[j][0]}
            for t in TARGETS:
                mu, sd = target_stats[t]
                val = float(preds[t].cpu().numpy()[j]) * sd + mu
                lo, hi = CLAMP_RANGES[t]
                val = max(lo, min(hi, val))
                results[t] = round(val, 3)
            all_results.append(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="FluorCode: Predict FP photophysical properties from sequence"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sequence", type=str, help="Amino acid sequence")
    group.add_argument("-f", "--fasta", type=str, help="Path to FASTA file")

    parser.add_argument("-c", "--checkpoint", type=str,
                        help="Path to a single fold checkpoint (best.pt)")
    parser.add_argument("-d", "--checkpoint_dir", type=str,
                        help="Path to directory with fold_*/best.pt checkpoints")
    parser.add_argument("--ensemble", action="store_true",
                        help="Average predictions across all available fold checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto (default: auto)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    parser.add_argument("--name", type=str, default="query",
                        help="Name for single sequence prediction")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Resolve checkpoints
    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths = [Path(args.checkpoint)]
    elif args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        if args.ensemble:
            ckpt_paths = sorted(ckpt_dir.glob("fold_*/best.pt"))
        else:
            ckpt_paths = [ckpt_dir / "fold_0" / "best.pt"]

    if not ckpt_paths:
        print("Error: no checkpoints found. Provide --checkpoint or --checkpoint_dir.")
        sys.exit(1)

    for p in ckpt_paths:
        if not p.exists():
            print(f"Error: checkpoint not found: {p}")
            sys.exit(1)

    print(f"Checkpoints: {len(ckpt_paths)}")

    # Build model
    print("Loading ESM2-650M + LoRA...")
    model, alphabet = build_model(device=device)

    # Load sequences
    if args.sequence:
        sequences = [(args.name, args.sequence)]
    else:
        sequences = parse_fasta(args.fasta)
    print(f"Sequences: {len(sequences)}")

    # Predict (ensemble if multiple checkpoints)
    all_fold_results = []
    for ckpt_path in ckpt_paths:
        target_stats = load_checkpoint(model, str(ckpt_path), device=device)
        fold_results = predict_batch(model, alphabet, sequences, target_stats,
                                     device=device, batch_size=args.batch_size)
        all_fold_results.append(fold_results)
        print(f"  Predicted with {ckpt_path.parent.name}")

    # Average across folds
    final_results = []
    for i in range(len(sequences)):
        result = {"name": sequences[i][0], "sequence_length": len(sequences[i][1])}
        for t in TARGETS:
            vals = [fold[i][t] for fold in all_fold_results]
            result[t] = round(float(np.mean(vals)), 3)
        final_results.append(result)

    # Output
    print(f"\n{'Name':<30s} {'ex_max':>8s} {'em_max':>8s} {'QY':>6s} {'ext_coeff':>10s} {'pKa':>6s}")
    print("-" * 75)
    for r in final_results:
        print(f"{r['name']:<30s} {r['ex_max']:>8.1f} {r['em_max']:>8.1f} "
              f"{r['qy']:>6.3f} {r['ext_coeff']:>10.0f} {r['pka']:>6.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\nSaved → {args.output}")

    return final_results


if __name__ == "__main__":
    main()
