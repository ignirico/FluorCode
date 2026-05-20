# FluorCode Inference

Predict fluorescent protein photophysical properties from amino acid sequence using LoRA-fine-tuned ESM2-650M.

## Predicted Properties

| Property | Unit | Description |
|----------|------|-------------|
| ex_max | nm | Excitation maximum wavelength |
| em_max | nm | Emission maximum wavelength |
| qy | 0-1 | Quantum yield |
| ext_coeff | M⁻¹cm⁻¹ | Molar extinction coefficient |
| pka | - | Acid dissociation constant |

## Requirements

```bash
pip install torch "numpy<2" fair-esm
```

> **Important:** Install `fair-esm`, **not** `esm`. The PyPI name `esm` is a different
> package (EvolutionaryScale) that does not expose `esm.pretrained.esm2_t33_650M_UR50D`.
> If you see `AttributeError: module 'esm' has no attribute 'pretrained'`, run:
> `pip uninstall -y esm fair-esm && pip install fair-esm`.

> **NumPy 2.x note:** prebuilt PyTorch wheels are typically compiled against NumPy 1.x.
> If you see `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`,
> pin NumPy: `pip install "numpy<2"`.

## Usage

### Single sequence

```bash
python predict.py \
    --sequence MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK \
    --checkpoint path/to/fold_0/best.pt
```

### FASTA file

```bash
python predict.py \
    --fasta my_proteins.fasta \
    --checkpoint path/to/fold_0/best.pt \
    --output predictions.json
```

### Ensemble (all 20 folds)

```bash
python predict.py \
    --fasta my_proteins.fasta \
    --checkpoint_dir model/LoRA_ESM2/checkpoints/ \
    --ensemble \
    --output predictions.json
```

### Python API

```python
from model import build_model_for_checkpoint
from predict import predict_single

# Auto-detects LoRA rank / pool layout from the checkpoint and builds a matching model.
model, alphabet, target_stats = build_model_for_checkpoint("fold_0/best.pt", device="cuda")

result = predict_single(
    model, alphabet,
    sequence="MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK",
    target_stats=target_stats,
    device="cuda",
)
print(result)
# {'ex_max': 488.5, 'em_max': 509.2, 'qy': 0.65, 'ext_coeff': 55000, 'pka': 6.0}
```

## Model Architecture

```
Sequence → ESM2-650M (frozen) + LoRA (layers 27-32)
         → ChromophoreAwareAttentionPooling (4 heads)
         → 5 MLP prediction heads
```

- **Base model**: ESM2-650M (`esm2_t33_650M_UR50D`, 650M parameters)
- **LoRA adapters**: rank-16, alpha=32, applied to q/k/v/out_proj in last 6 layers (kept scaling = 2.0)
- **Trainable parameters**: ~500K (0.08% of ESM2)
- **Pooling**: 4-head attention with learned chromophore position bias

The bundled `fold_0/best.pt` (in `model/LoRA_ESM2/checkpoints/`) is an earlier-generation
checkpoint with rank-8 LoRA on q/v_proj only and a simpler pool projection. Inference
introspects the checkpoint and builds a matching architecture, so both generations
load without manual configuration.

## Checkpoints

Each fold checkpoint (`fold_*/best.pt`) contains:
- LoRA adapter weights
- Attention pooling weights
- Prediction head weights
- Target normalization statistics (mean/std per target)

Ensemble prediction averages across all 20 fold checkpoints for best accuracy.

## Citation

If you use this model, please cite our ICML AI4Science workshop paper (2026).
