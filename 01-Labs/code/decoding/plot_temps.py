"""画图：同一个位置，不同温度下概率最高的 12 个 token 的概率。"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decoding import tok, next_logprobs

prompt = "The cat was very"
lp = next_logprobs(tok.encode(prompt).ids)
top = torch.topk(lp, 12).indices.tolist()
labels = [tok.decode([i]).strip() for i in top]
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150, sharey=True)
for ax, t in zip(axes, [0.5, 1.0, 1.8]):
    p = torch.softmax(lp / t, -1)
    vals = [p[i].item() for i in top]
    rest = 1 - sum(vals)
    ax.bar(range(12), vals, color="#2e5c7a")
    ax.set_xticks(range(12)); ax.set_xticklabels(labels, rotation=60, fontsize=8)
    ax.set_title(f"温度 {t}（前 12 名以外合计 {rest:.0%}）", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("概率")
fig.suptitle(f'"{prompt} ___" 的下一个 token', fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "decoding_temps.png")
fig.savefig(out)
for t in [0.5, 1.0, 1.8]:
    p = torch.softmax(lp / t, -1)
    print(f"温度 {t}: " + "，".join(f"{tok.decode([i]).strip()} {p[i].item():.1%}" for i in top[:5]) + f"；前 12 名以外合计 {1 - sum(p[i].item() for i in top):.1%}")
