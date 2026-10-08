"""After adding LoRA: does the output change? How many parameters are trained?"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama
from lora import apply_lora

torch.manual_seed(0)
x = torch.randint(0, 4096, (1, 16))
total = sum(p.numel() for p in TinyLlama(Config()).parameters())
print(f"parameters in the original model: {total:,}")
print(f"one 384x384 attention matrix W: {384*384:,} numbers")
for r in [1, 4, 16, 64]:
    torch.manual_seed(0)
    m = TinyLlama(Config()); before = m(x)[0]
    apply_lora(m, r); after = m(x)[0]
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"r={r:<3} A and B next to one 384x384 matrix: {384*r + r*384:>6,} numbers; trainable in the whole model: {n:>9,} "
          f"({n/total:.2%} of the original); output unchanged after adding LoRA: {torch.allclose(before, after)}")
