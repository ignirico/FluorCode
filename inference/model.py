"""
FluorCode — LoRA-ESM2 model architecture for fluorescent protein property prediction.

Architecture:
    ESM2-650M (frozen) + LoRA rank-16 adapters (layers 27-32, q/k/v/out_proj)
    → ChromophoreAwareAttentionPooling (4 heads, 1280-dim output)
    → 5 MLP prediction heads (ex_max, em_max, qy, ext_coeff, pka)

Usage:
    from model import build_model, load_checkpoint
    model, alphabet = build_model()
    load_checkpoint(model, "path/to/best.pt")
"""

import torch
import torch.nn as nn

TARGETS = ["ex_max", "em_max", "qy", "ext_coeff", "pka"]
EMBED_DIM = 1280
LORA_RANK = 16
LORA_ALPHA = 32
LORA_LAYERS = list(range(27, 33))


class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.orig_linear = orig_linear
        self.scaling = alpha / rank
        in_f = orig_linear.in_features
        out_f = orig_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        orig_linear.weight.requires_grad = False
        if orig_linear.bias is not None:
            orig_linear.bias.requires_grad = False

    def forward(self, x):
        return self.orig_linear(x) + self.scaling * (x @ self.lora_A @ self.lora_B)


def apply_lora(model, target_layers, rank=16, alpha=32.0):
    for layer_idx in target_layers:
        layer = model.layers[layer_idx]
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            orig = getattr(attn, proj_name)
            setattr(attn, proj_name, LoRALinear(orig, rank=rank, alpha=alpha))


class ChromophoreAwareAttentionPooling(nn.Module):
    def __init__(self, hidden_dim=1280, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1, bias=True) for _ in range(n_heads)
        ])
        self.chrom_bias = nn.Parameter(torch.tensor(3.0))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, hidden_states, seq_lens, chrom_positions=None):
        B, L, D = hidden_states.shape
        mask = torch.zeros(B, L, device=hidden_states.device)
        for i in range(B):
            mask[i, 1: seq_lens[i] + 1] = 1.0

        chrom_mask = torch.zeros(B, L, device=hidden_states.device)
        if chrom_positions is not None:
            for i in range(B):
                for p in chrom_positions[i]:
                    if 0 <= p < seq_lens[i]:
                        chrom_mask[i, p + 1] = 1.0

        head_outputs = []
        for head in self.heads:
            scores = head(hidden_states).squeeze(-1)
            scores = scores + self.chrom_bias * chrom_mask
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1)
            head_outputs.append((hidden_states * weights.unsqueeze(-1)).sum(dim=1))

        pooled = self.proj(torch.cat(head_outputs, dim=-1))
        return pooled


class LoRAESM2MultiTask(nn.Module):
    def __init__(self, esm_backbone, n_heads=4, dropout=0.1):
        super().__init__()
        self.esm = esm_backbone
        self.pool = ChromophoreAwareAttentionPooling(EMBED_DIM, n_heads=n_heads)
        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.LayerNorm(EMBED_DIM),
                nn.Dropout(dropout),
                nn.Linear(EMBED_DIM, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            for t in TARGETS
        })

    def forward(self, tokens, seq_lens, chrom_positions=None):
        results = self.esm(tokens, repr_layers=[33], return_contacts=False)
        hidden = results["representations"][33]
        pooled = self.pool(hidden, seq_lens, chrom_positions)
        preds = {t: head(pooled).squeeze(-1) for t, head in self.heads.items()}
        return preds, pooled


def build_model(device="cpu"):
    """Build the full LoRA-ESM2 model and return (model, alphabet)."""
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    for p in esm_model.parameters():
        p.requires_grad = False
    apply_lora(esm_model, LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA)
    model = LoRAESM2MultiTask(esm_model, n_heads=4, dropout=0.1)
    model = model.to(device)
    model.eval()
    return model, alphabet


def load_checkpoint(model, ckpt_path, device="cpu"):
    """Load a fold checkpoint. Returns target_stats dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    model.eval()
    return ckpt["target_stats"]
