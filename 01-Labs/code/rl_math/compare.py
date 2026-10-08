"""Evaluate every model on the same 500 test problems: greedy accuracy, sampled accuracy, and pass@8."""
from arith import *

models = {
    "SFT (2000 labeled problems)": "sft_math.pt",
    "SFT, 600 more steps on the same data": "sftmore_math.pt",
    "GRPO, 600 steps": "grpo_math.pt",
    "DPO, lr 2e-5": "dpo_math_lr2e-05_nll0.pt",
    "DPO, lr 2e-6": "dpo_math_lr2e-06_nll0.pt",
    "DPO + SFT loss on chosen": "dpo_math_lr2e-05_nll1.pt",
    "SFT on own correct answers only": "dpo_math_lr2e-05_nll1_nodpo.pt",
}
print(f"{'model':<38}{'greedy':>8}{'sampled':>9}{'pass@8':>8}")
for name, f in models.items():
    m = load_model("~/.labs_data/" + f)
    g = accuracy(m, TEST)
    s = accuracy(m, TEST, temperature=1.0, k=1, seed=1)
    p8 = accuracy(m, TEST, temperature=1.0, k=8, seed=1)
    print(f"{name:<38}{g:>8.1%}{s:>9.1%}{p8:>8.1%}", flush=True)
