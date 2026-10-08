"""训练前的检查：参数量、张量形状、RoPE 的相对位置性质、随机初始化时的损失、训练速度。"""
import math, time, torch
from model import Config, TinyLlama, rope_tables, apply_rope

torch.manual_seed(0)
cfg = Config()
model = TinyLlama(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"参数总量: {n:,}（约 {n/1e6:.1f}M）")
print(f"  其中词嵌入（与输出层共用）: {model.tok_emb.weight.numel():,}")
print(f"  每个 Transformer 块: {sum(p.numel() for p in model.blocks[0].parameters()):,}")

idx = torch.randint(0, cfg.vocab_size, (2, 16))
logits, loss = model(idx, idx)
print(f"\n输入形状 {tuple(idx.shape)} -> 输出 logits 形状 {tuple(logits.shape)}")
print(f"随机初始化时的损失: {loss.item():.3f}，理论值 ln(4096) = {math.log(4096):.3f}")

# RoPE：q 在位置 m、k 在位置 n 时，点积只取决于 m - n
cos, sin = rope_tables(64, 256)
q, k = torch.randn(64), torch.randn(64)
def score(m, n):
    qm = apply_rope(q.view(1, 1, 1, 64).expand(1, 1, 256, 64), cos, sin)[0, 0, m]
    kn = apply_rope(k.view(1, 1, 1, 64).expand(1, 1, 256, 64), cos, sin)[0, 0, n]
    return (qm @ kn).item()
print(f"\nRoPE 检查：位置 (10, 11) 的打分 {score(11, 10):.4f}，位置 (100, 101) 的打分 {score(101, 100):.4f}")
print(f"           位置 (10, 13) 的打分 {score(13, 10):.4f}，位置 (200, 203) 的打分 {score(203, 200):.4f}")

dev = "mps"
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, cfg.vocab_size, (32, 256), device=dev)
for i in range(13):
    if i == 3: torch.mps.synchronize(); t0 = time.time()
    _, loss = model(x, x); opt.zero_grad(); loss.backward(); opt.step()
torch.mps.synchronize()
dt = (time.time() - t0) / 10
print(f"\n在 Apple M5 GPU (MPS) 上：每步 32×256 = 8192 个 token，用时 {dt*1000:.0f} 毫秒，约 {8192/dt:,.0f} token/秒")
