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
pip install torch fair-esm numpy
```

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
from model import build_model, load_checkpoint
from predict import predict_single

model, alphabet = build_model(device="cuda")
target_stats = load_checkpoint(model, "fold_0/best.pt", device="cuda")

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
Sequence → ESM2-650M (frozen) + LoRA rank-8 (layers 27-32)
         → ChromophoreAwareAttentionPooling (4 heads)
         → 5 MLP prediction heads
```

- **Base model**: ESM2-650M (`esm2_t33_650M_UR50D`, 650M parameters)
- **LoRA adapters**: rank=16, alpha=32, applied to q/v_proj in last 6 layers
- **Trainable parameters**: ~500K (0.08% of ESM2)
- **Pooling**: 4-head attention with learned chromophore position bias

## Checkpoints

Each fold checkpoint (`fold_*/best.pt`) contains:
- LoRA adapter weights
- Attention pooling weights
- Prediction head weights
- Target normalization statistics (mean/std per target)

Ensemble prediction averages across all 20 fold checkpoints for best accuracy.

## Citation

If you use this model, please cite our ICML AI4Science workshop paper (2026).
