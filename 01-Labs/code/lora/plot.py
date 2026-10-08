"""画两张小图：微调过程中，龙故事损失和普通故事损失的变化。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(os.path.expanduser("~/.labs_data/lora_results.json")))
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
colors = {"全量微调": "#b03a2e", "LoRA r=1": "#9ec5e0", "LoRA r=4": "#5b9bc8", "LoRA r=16": "#2e6f9e", "LoRA r=64": "#163f5c"}
base = R["原模型（不微调）"]
for ax, k, title in [(axes[0], 1, "没见过的龙故事上的损失（越低越好）"), (axes[1], 2, "没见过的普通故事上的损失（越低越好）")]:
    for name, c in colors.items():
        cur = R[name]["curve"]
        ax.plot([s for s, *_ in cur], [row[k] for row in cur], "o-", ms=3, color=c, label=name)
    ax.axhline(base["dragon" if k == 1 else "general"], color="#999", ls="--", lw=1, label="原模型")
    ax.set_title(title, fontsize=10); ax.set_xlabel("微调步数"); ax.grid(alpha=0.3)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "img", "lora_curves.png"))
print("saved")
