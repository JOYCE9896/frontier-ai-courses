"""Plot: probabilities of the 12 most likely next tokens at different temperatures."""
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
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150, sharey=True)
for ax, t in zip(axes, [0.5, 1.0, 1.8]):
    p = torch.softmax(lp / t, -1)
    vals = [p[i].item() for i in top]
    rest = 1 - sum(vals)
    ax.bar(range(12), vals, color="#2e5c7a")
    ax.set_xticks(range(12)); ax.set_xticklabels(labels, rotation=60, fontsize=8)
    ax.set_title(f"temperature {t} (outside top 12: {rest:.0%})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("probability")
fig.suptitle(f'next token after "{prompt} ___"', fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "decoding_temps.png")
fig.savefig(out)
for t in [0.5, 1.0, 1.8]:
    p = torch.softmax(lp / t, -1)
    print(f"temperature {t}: " + ", ".join(f"{tok.decode([i]).strip()} {p[i].item():.1%}" for i in top[:5]) + f"; outside the top 12: {1 - sum(p[i].item() for i in top):.1%}")
