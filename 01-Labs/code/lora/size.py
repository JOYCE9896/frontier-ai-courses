"""How much needs to be saved after LoRA training? Does merging into the original weights change the output?"""
import os, sys, io, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinyllama"))
from model import Config, TinyLlama
from lora import apply_lora, LoRALinear

def nbytes(state):
    buf = io.BytesIO(); torch.save(state, buf); return buf.tell()

m = TinyLlama(Config())
full = nbytes(m.state_dict())
print(f"full model saved to a file: {full/1e6:.1f} MB")
for r in [1, 4, 16, 64]:
    m = apply_lora(TinyLlama(Config()), r)
    lora_only = {k: v for k, v in m.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    print(f"LoRA r={r:<3} only A and B: {nbytes(lora_only)/1e6:6.2f} MB ({nbytes(lora_only)/full:.1%} of the full model)")

# merge: compute W + scale * B A, swap in a plain linear layer, compare outputs
torch.manual_seed(0)
m = apply_lora(TinyLlama(Config()), 4)
for mod in m.modules():
    if isinstance(mod, LoRALinear):
        torch.nn.init.normal_(mod.B, std=0.02)       # give B some non-zero values, as if it had been trained
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
print(f"\nlargest difference in output before vs. after merging: {(before - after).abs().max().item():.2e}")
