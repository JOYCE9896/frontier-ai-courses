"""把训练日志画成损失曲线。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = json.load(open(os.path.expanduser("~/.labs_data/tinyllama_run/log.json")))
out = os.path.join(os.path.dirname(__file__), "..", "..", "img")
os.makedirs(out, exist_ok=True)

tr = log["train"]; va = log["val"]
# 训练损失每步都在抖，取 50 步滑动平均看得清楚些
w = 50
smooth = [sum(l for _, l in tr[max(0, i-w+1):i+1]) / len(tr[max(0, i-w+1):i+1]) for i in range(len(tr))]

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot([s for s, _ in tr], [l for _, l in tr], color="#c9c4b8", lw=0.6, label="训练损失（每一步）")
ax.plot([s for s, _ in tr], smooth, color="#7a5c2e", lw=1.5, label="训练损失（50 步平均）")
ax.plot([s for s, _ in va], [l for _, l in va], "o-", color="#2e5c7a", ms=3, lw=1.2, label="验证损失")
ax.set_xlabel("训练步数"); ax.set_ylabel("交叉熵损失")
ax.set_ylim(1, 8.6); ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(out, "tinyllama_loss.png"))
print("saved", os.path.abspath(os.path.join(out, "tinyllama_loss.png")))
