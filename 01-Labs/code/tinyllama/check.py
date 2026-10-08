"""Checks before training: parameter count, tensor shapes, the relative-position property of RoPE,
the loss at random initialization, and training speed."""
import math, time, torch
from model import Config, TinyLlama, rope_tables, apply_rope

torch.manual_seed(0)
cfg = Config()
model = TinyLlama(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"total parameters: {n:,} (about {n/1e6:.1f}M)")
print(f"  token embedding (shared with the output layer): {model.tok_emb.weight.numel():,}")
print(f"  each Transformer block: {sum(p.numel() for p in model.blocks[0].parameters()):,}")

idx = torch.randint(0, cfg.vocab_size, (2, 16))
logits, loss = model(idx, idx)
print(f"\ninput shape {tuple(idx.shape)} -> output logits shape {tuple(logits.shape)}")
print(f"loss at random initialization: {loss.item():.3f}, theory ln(4096) = {math.log(4096):.3f}")

# RoPE: with q at position m and k at position n, the dot product depends only on m - n
cos, sin = rope_tables(64, 256)
q, k = torch.randn(64), torch.randn(64)
def score(m, n):
    qm = apply_rope(q.view(1, 1, 1, 64).expand(1, 1, 256, 64), cos, sin)[0, 0, m]
    kn = apply_rope(k.view(1, 1, 1, 64).expand(1, 1, 256, 64), cos, sin)[0, 0, n]
    return (qm @ kn).item()
print(f"\nRoPE: score at positions (10, 11) = {score(11, 10):.4f}, at (100, 101) = {score(101, 100):.4f}")
print(f"      score at positions (10, 13) = {score(13, 10):.4f}, at (200, 203) = {score(203, 200):.4f}")

dev = "mps"
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, cfg.vocab_size, (32, 256), device=dev)
for i in range(13):
    if i == 3: torch.mps.synchronize(); t0 = time.time()
    _, loss = model(x, x); opt.zero_grad(); loss.backward(); opt.step()
torch.mps.synchronize()
dt = (time.time() - t0) / 10
print(f"\non the Apple M5 GPU (MPS): one step of 32x256 = 8192 tokens takes {dt*1000:.0f} ms, about {8192/dt:,.0f} tokens/s")
