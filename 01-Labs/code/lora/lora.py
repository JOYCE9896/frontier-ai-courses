"""LoRA：冻结原来的权重矩阵 W，只训练旁边加上的两个小矩阵 A 和 B。"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """输出 = W x + (alpha / r) * B (A x)。W 冻结不动；A 是 r×输入维度，B 是输出维度×r。"""
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False          # 原权重冻结
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B 从 0 开始：刚加上时模型和原来完全一样

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self):
        """训练完可以把 B A 直接加进 W，推理时就和普通线性层一样快。"""
        return self.base.weight + self.scale * self.B @ self.A


def apply_lora(model, r, alpha=None):
    """把每个 Transformer 块里的 7 个线性层（注意力的 q、k、v、o，前馈的 w1、w2、w3）换成 LoRA 版本，
    其余参数（词嵌入、归一化）全部冻结。"""
    alpha = alpha if alpha is not None else 2 * r
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        for parent, names in [(block.attn, ["wq", "wk", "wv", "wo"]), (block.ffn, ["w1", "w2", "w3"])]:
            for n in names:
                setattr(parent, n, LoRALinear(getattr(parent, n), r, alpha))
    return model
