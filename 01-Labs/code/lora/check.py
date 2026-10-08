"""加上 LoRA 之后：输出变了吗？要训练的参数有多少？"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama
from lora import apply_lora

torch.manual_seed(0)
x = torch.randint(0, 4096, (1, 16))
total = sum(p.numel() for p in TinyLlama(Config()).parameters())
print(f"原模型参数总量：{total:,}")
print(f"一个 384×384 的注意力矩阵 W：{384*384:,} 个数")
for r in [1, 4, 16, 64]:
    torch.manual_seed(0)
    m = TinyLlama(Config()); before = m(x)[0]
    apply_lora(m, r); after = m(x)[0]
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"r={r:<3} 每个 384×384 矩阵旁边的 A、B 共 {384*r + r*384:>6,} 个数；全模型要训练 {n:>9,} 个参数（原来的 {n/total:.2%}）；"
          f"加上 LoRA 后输出不变：{torch.allclose(before, after)}")
