"""LoRA 训练完要存多少东西？合并进原权重后，输出是否不变？"""
import os, sys, io, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama
from lora import apply_lora, LoRALinear

def nbytes(state):
    buf = io.BytesIO(); torch.save(state, buf); return buf.tell()

m = TinyLlama(Config())
full = nbytes(m.state_dict())
print(f"完整模型存成文件：{full/1e6:.1f} MB")
for r in [1, 4, 16, 64]:
    m = apply_lora(TinyLlama(Config()), r)
    lora_only = {k: v for k, v in m.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    print(f"LoRA r={r:<3} 只需要存 A 和 B：{nbytes(lora_only)/1e6:6.2f} MB（完整模型的 {nbytes(lora_only)/full:.1%}）")

# 合并：把 W + scale * B A 算出来，换回普通线性层，看输出是否一样
torch.manual_seed(0)
m = apply_lora(TinyLlama(Config()), 4)
for mod in m.modules():
    if isinstance(mod, LoRALinear):
        torch.nn.init.normal_(mod.B, std=0.02)       # 随便给 B 一些非零值，模拟训练过的状态
x = torch.randint(0, 4096, (1, 32))
before = m(x)[0]
for block in m.blocks:
    for parent, names in [(block.attn, ["wq", "wk", "wv", "wo"]), (block.ffn, ["w1", "w2", "w3"])]:
        for n in names:
            lo = getattr(parent, n)
            lin = torch.nn.Linear(lo.base.in_features, lo.base.out_features, bias=False)
            lin.weight.data = lo.merged_weight().detach()
            setattr(parent, n, lin)
after = m(x)[0]
print(f"\n合并前后输出的最大差别：{(before - after).abs().max().item():.2e}（只是浮点舍入误差）")
