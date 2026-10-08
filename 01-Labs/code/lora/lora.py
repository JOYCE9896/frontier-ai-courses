"""LoRA: freeze the original weight matrix W and train only two small matrices A and B added next to it."""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """output = W x + (alpha / r) * B (A x). W is frozen; A is r x in_features, B is out_features x r."""
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False          # freeze the original weight
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B starts at 0, so the model is unchanged at first

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self):
        """After training, B A can be added into W, so inference is as fast as a plain linear layer."""
        return self.base.weight + self.scale * self.B @ self.A


def apply_lora(model, r, alpha=None):
    """Replace the 7 linear layers in every Transformer block (attention q, k, v, o; feed-forward w1, w2, w3)
    with LoRA versions, and freeze every other parameter (embedding, norms)."""
    alpha = alpha if alpha is not None else 2 * r
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        for parent, names in [(block.attn, ["wq", "wk", "wv", "wo"]), (block.ffn, ["w1", "w2", "w3"])]:
            for n in names:
                setattr(parent, n, LoRALinear(getattr(parent, n), r, alpha))
    return model
