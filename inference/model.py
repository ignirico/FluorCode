"""LoRA-ESM2 model: ESM2-650M backbone with LoRA on the last 6 layers,
chromophore-aware attention pooling, and 5 MLP heads (ex_max, em_max, qy, ext_coeff, pka).

The exact LoRA configuration (rank, target projections) and pooling-projection layout
vary between checkpoint generations. `build_model_for_checkpoint` introspects the
checkpoint's state_dict and constructs a matching architecture so old and new
checkpoints both load cleanly.
"""

import torch
import torch.nn as nn

TARGETS = ["ex_max", "em_max", "qy", "ext_coeff", "pka"]
EMBED_DIM = 1280
LORA_LAYERS = list(range(27, 33))

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_PROJS = ["q_proj", "k_proj", "v_proj", "out_proj"]
DEFAULT_POOL_STYLE = "bottleneck"

# Kept for backward compatibility with code that imports the old constants.
LORA_RANK = DEFAULT_LORA_RANK
LORA_ALPHA = DEFAULT_LORA_ALPHA


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


def apply_lora(model, target_layers, rank=DEFAULT_LORA_RANK, alpha=DEFAULT_LORA_ALPHA,
               projs=None):
    projs = projs if projs is not None else DEFAULT_LORA_PROJS
    for layer_idx in target_layers:
        layer = model.layers[layer_idx]
        attn = layer.self_attn
        for proj_name in projs:
            orig = getattr(attn, proj_name)
            setattr(attn, proj_name, LoRALinear(orig, rank=rank, alpha=alpha))


def _build_pool_proj(hidden_dim, n_heads, style):
    if style == "simple":
        # Old checkpoints: Linear(D*H -> D) -> LayerNorm(D)
        return nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    if style == "bottleneck":
        # New checkpoints: Linear(D*H -> D/2) -> GELU -> LayerNorm(D/2) -> Linear(D/2 -> D)
        return nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
    raise ValueError(f"Unknown pool style: {style!r}")


class ChromophoreAwareAttentionPooling(nn.Module):
    def __init__(self, hidden_dim=1280, n_heads=4, pool_style=DEFAULT_POOL_STYLE):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1, bias=True) for _ in range(n_heads)
        ])
        self.chrom_bias = nn.Parameter(torch.tensor(3.0))
        self.proj = _build_pool_proj(hidden_dim, n_heads, pool_style)

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
    def __init__(self, esm_backbone, n_heads=4, dropout=0.1,
                 pool_style=DEFAULT_POOL_STYLE):
        super().__init__()
        self.esm = esm_backbone
        self.pool = ChromophoreAwareAttentionPooling(EMBED_DIM, n_heads=n_heads,
                                                    pool_style=pool_style)
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


def _load_esm2_650m():
    """Import `esm` and load ESM2-650M, with a clear error if the wrong package is installed.

    The PyPI name `esm` refers to EvolutionaryScale's package, which does NOT expose
    `esm.pretrained.esm2_t33_650M_UR50D`. The package we need is `fair-esm`, which
    installs to the same `esm` namespace and conflicts with it. If both are installed,
    the EvolutionaryScale one usually wins.
    """
    try:
        import esm
    except ImportError as e:
        raise ImportError(
            "The `esm` module is not installed. Install Facebook's ESM with:\n"
            "    pip install fair-esm\n"
            "(Do not `pip install esm` — that is a different package.)"
        ) from e

    if not hasattr(esm, "pretrained") or not hasattr(getattr(esm, "pretrained", None), "esm2_t33_650M_UR50D"):
        esm_file = getattr(esm, "__file__", "<unknown>")
        raise ImportError(
            "The installed `esm` package does not expose `esm.pretrained.esm2_t33_650M_UR50D`.\n"
            "This usually means EvolutionaryScale's `esm` package is installed instead of `fair-esm`.\n"
            f"  Installed esm: {esm_file}\n"
            "Fix:\n"
            "    pip uninstall -y esm fair-esm\n"
            "    pip install fair-esm\n"
        )

    return esm.pretrained.esm2_t33_650M_UR50D()


def detect_checkpoint_arch(state_dict):
    """Infer LoRA rank/projections and pool style from a checkpoint state_dict.

    Returns a dict with keys: lora_rank, lora_alpha, lora_projs, pool_style.
    """
    projs = sorted({k.split(".")[-2] for k in state_dict if "lora_A" in k or "lora_B" in k})
    if not projs:
        raise ValueError("No LoRA parameters found in checkpoint state_dict.")
    ranks = {state_dict[k].shape[1] for k in state_dict if k.endswith("lora_A")}
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent LoRA ranks across layers: {ranks}")
    rank = ranks.pop()

    pool_idx = sorted({k.split(".")[2] for k in state_dict if k.startswith("pool.proj.")})
    if pool_idx == ["0", "1"]:
        pool_style = "simple"
    elif pool_idx == ["0", "2", "3"]:
        pool_style = "bottleneck"
    else:
        raise ValueError(f"Unrecognized pool projection layout (indices {pool_idx}).")

    # Both checkpoint generations were trained with effective scaling = alpha/rank = 2.0
    alpha = float(rank * 2)

    return {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_projs": projs,
        "pool_style": pool_style,
    }


def build_model(device="cpu", lora_rank=DEFAULT_LORA_RANK, lora_alpha=DEFAULT_LORA_ALPHA,
                lora_projs=None, pool_style=DEFAULT_POOL_STYLE):
    """Build the full LoRA-ESM2 model and return (model, alphabet)."""
    esm_model, alphabet = _load_esm2_650m()
    for p in esm_model.parameters():
        p.requires_grad = False
    apply_lora(esm_model, LORA_LAYERS, rank=lora_rank, alpha=lora_alpha,
               projs=lora_projs)
    model = LoRAESM2MultiTask(esm_model, n_heads=4, dropout=0.1, pool_style=pool_style)
    model = model.to(device)
    model.eval()
    return model, alphabet


def build_model_for_checkpoint(ckpt_path, device="cpu"):
    """Build a model whose architecture matches the given checkpoint, then load it.

    Returns (model, alphabet, target_stats). Use this when you don't know in advance
    which checkpoint generation (rank-8 q/v-only vs rank-16 q/k/v/out) you're loading.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = detect_checkpoint_arch(ckpt["trainable_state"])
    model, alphabet = build_model(
        device=device,
        lora_rank=arch["lora_rank"],
        lora_alpha=arch["lora_alpha"],
        lora_projs=arch["lora_projs"],
        pool_style=arch["pool_style"],
    )
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(
            f"Checkpoint contains unexpected keys not present in model: {unexpected}"
        )
    model.eval()
    return model, alphabet, ckpt["target_stats"]


def load_checkpoint(model, ckpt_path, device="cpu"):
    """Load a fold checkpoint into an already-built model. Returns target_stats.

    Prefer `build_model_for_checkpoint` when loading checkpoints whose architecture
    you don't know in advance — this function will fail with a shape mismatch if
    `model` was built with a different LoRA rank or pool style than the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    model.eval()
    return ckpt["target_stats"]


def find_chromophore_positions(seq: str) -> list:
    """Locate the chromophore triad XYG in an FP sequence.

    Returns 0-indexed [x, y, g] for the candidate Y closest to position 64
    (the avGFP-trimmed convention) with G immediately downstream. Returns
    [-1, -1, -1] when no XYG match is found, which disables the chromophore
    bias for that sequence.
    """
    candidates = [
        (i - 1, i, i + 1)
        for i in range(1, len(seq) - 1)
        if seq[i] == "Y" and seq[i + 1] == "G"
    ]
    if not candidates:
        return [-1, -1, -1]
    return list(min(candidates, key=lambda c: abs(c[1] - 64)))
