"""一个从零写的小 LLaMA：RMSNorm + RoPE + 因果自注意力 + SwiGLU 前馈层。"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    dim: int = 384          # 每个 token 的向量长度
    n_layers: int = 6       # Transformer 块的层数
    n_heads: int = 6        # 注意力头数，每个头 384/6 = 64 维
    hidden_dim: int = 1024  # SwiGLU 前馈层的中间维度
    max_seq_len: int = 256  # 一次最多看多少个 token


class RMSNorm(nn.Module):
    """把向量除以它的均方根，再乘一个可学习的缩放。比 LayerNorm 少了减均值这一步。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def rope_tables(head_dim, max_seq_len, base=10000.0):
    """预先算好每个位置、每对维度的旋转角的 cos 和 sin。"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    pos = torch.arange(max_seq_len).float()                                   # (T,)
    angles = torch.outer(pos, freqs)                                          # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """把 x 的维度两两配对，每一对在二维平面上按位置旋转。x: (B, heads, T, head_dim)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # 因果遮盖：位置 i 只能看位置 <= i
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        # 1. 算 query、key、value，并拆成多个头: (B, heads, T, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # 2. 只对 q 和 k 加旋转位置编码
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # 3. 缩放点积注意力：每个位置对之前所有位置打分，softmax 后对 value 加权平均
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v                                             # (B, heads, T, head_dim)
        # 4. 把各个头拼回去，再做一次线性变换
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class SwiGLU(nn.Module):
    """前馈层：silu(x W1) 逐元素乘以 (x W3)，再乘 W2 投影回去。"""
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """一个 Transformer 块：先归一化再注意力，加残差；先归一化再前馈，加残差。"""
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 输入嵌入和输出层共用一个矩阵
        cos, sin = rope_tables(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)                     # (B, T) -> (B, T, dim)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self.lm_head(self.norm(x))       # (B, T, vocab)
        loss = None
        if targets is not None:
            # 每个位置都在预测下一个 token，用交叉熵衡量预测得多准
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eot_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)          # 贪心
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)              # 按概率抽一个
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and nxt.item() == eot_id:
                break
        return idx
